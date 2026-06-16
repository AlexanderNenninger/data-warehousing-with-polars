"""
incremental.py

Public API: the :func:`incremental` decorator and :class:`IncrementalPipeline`.
Coordinates file listing, watermark tracking, transform dispatch, and compaction.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Protocol, cast, runtime_checkable

import joblib
import polars as pl
from deltalake import DeltaTable, WriterProperties, write_deltalake
from deltalake.exceptions import TableNotFoundError

from .maintenance import _load_run_count, _save_run_count, maintain
from .record_batch_source import make_lazy
from .scd import _sink_scd2, _sink_scd4, _upsert_overwrite

logger = logging.getLogger(__name__)

FileFormat = Literal["parquet", "csv", "ndjson", "delta"]

_SUFFIXES: dict[str, str] = {
    "parquet": ".parquet",
    "csv": ".csv",
    "ndjson": ".ndjson",
}

_CDF_CONFIG = {"delta.enableChangeDataFeed": "true"}


# ── Internal helpers ──────────────────────────────────────────────────────────


def _list_local_files(source: str, suffixes: tuple[str, ...]) -> list[str]:
    """Return sorted absolute paths of files under *source* matching *suffixes*."""
    root = Path(source)
    if not root.exists():
        return []
    return sorted(str(p.resolve()) for p in root.rglob("*") if p.is_file() and p.suffix in suffixes)


def _list_s3_files(source: str, suffixes: tuple[str, ...]) -> list[str]:
    """Return sorted S3 URIs of files under *source* matching *suffixes*."""
    from pyarrow import fs as pa_fs  # noqa: PLC0415

    filesystem, base_path = pa_fs.FileSystem.from_uri(source)
    selector = pa_fs.FileSelector(base_path.rstrip("/"), recursive=True)
    try:
        file_infos = filesystem.get_file_info(selector)
    except FileNotFoundError:
        return []
    return sorted(
        f"s3://{info.path}"
        for info in file_infos
        if info.type == pa_fs.FileType.File and any(info.path.endswith(s) for s in suffixes)
    )


def _load_cursors(store: str) -> dict[int, object]:
    """Return the per-slot cursors recorded in the watermark table.

    Returns ``{}`` on first run, or when the store holds an incompatible
    (pre-fan-in) schema — in which case the pipeline does a fresh load.
    """
    try:
        result = cast(
            pl.DataFrame,
            pl.scan_delta(store).select("slot", "cursor_json").collect(),
        )
        return {
            int(row["slot"]): json.loads(row["cursor_json"]) for row in result.iter_rows(named=True)
        }
    except Exception:
        return {}


def _save_cursors(store: str, cursors: dict[int, object]) -> None:
    """Overwrite the watermark table with one row per slot cursor."""
    slots = sorted(cursors)
    now = datetime.now(timezone.utc)
    rows = pl.DataFrame({
        "slot": slots,
        "cursor_json": [json.dumps(cursors[s], default=str) for s in slots],
        "saved_at": [now] * len(slots),
    })
    write_deltalake(store, rows, mode="overwrite", schema_mode="overwrite")


def _scan_files(
    paths: list[str],
    fmt: FileFormat,
    reader_kwargs: dict | None = None,
    concat_options: dict | None = None,
) -> pl.LazyFrame:
    """Lazily scan *paths* into a single LazyFrame, tagging rows with ``_source_file``
    and ``_ingested_at``.
    """
    kwargs = reader_kwargs or {}
    now = datetime.now(timezone.utc)
    frames: list[pl.LazyFrame] = []

    for path in paths:
        if fmt == "parquet":
            lf = pl.scan_parquet(path, **kwargs)
        elif fmt == "csv":
            lf = pl.scan_csv(path, **kwargs)
        else:  # ndjson
            lf = pl.scan_ndjson(path, **kwargs)
        lf = lf.with_columns(
            pl.lit(path).alias("_source_file"),
            pl.lit(now).alias("_ingested_at"),
        )
        frames.append(lf)

    cc = concat_options or {}
    return pl.concat(frames, **cc)


def _read_delta_source(
    source: str,
    from_version: int | None,
) -> tuple[pl.LazyFrame, int]:
    """Return ``(lf, current_version)`` from *source* via CDF, or full scan on first run."""
    dt = DeltaTable(source)
    current_version = dt.version()

    if from_version is None:
        return pl.scan_delta(source), current_version

    reader = dt.load_cdf(
        starting_version=from_version + 1,
        ending_version=current_version,
    )
    lf = (
        make_lazy(reader)
        .filter(pl.col("_change_type").is_in(["insert", "update_postimage"]))
        .drop(["_change_type", "_commit_version", "_commit_timestamp"])
    )
    return lf, current_version


def _sink_target(
    target: str,
    lf: pl.LazyFrame,
    merge_on: str | list[str] | None,
    partition_by: str | list[str] | None = None,
) -> None:
    """Write *lf* to *target*: append when ``merge_on`` is ``None``, upsert otherwise."""
    keys = [merge_on] if isinstance(merge_on, str) else (list(merge_on) if merge_on else [])
    partition_list = (
        [partition_by]
        if isinstance(partition_by, str)
        else (list(partition_by) if partition_by else None)
    )

    first_write = False
    try:
        DeltaTable(target)
    except TableNotFoundError:
        first_write = True

    if not keys:
        if first_write:
            _df = cast(pl.DataFrame, lf.collect())
            write_deltalake(
                target,
                _df.to_arrow(),
                mode="overwrite",
                configuration=_CDF_CONFIG,
                partition_by=partition_list,
                writer_properties=WriterProperties(),
            )
        else:
            # Streaming append: data flows chunk-by-chunk without full materialisation.
            lf.sink_delta(target, mode="append")
        return

    _df = cast(pl.DataFrame, lf.collect(engine="streaming"))

    if first_write:
        write_deltalake(
            target,
            _df.to_arrow(),
            mode="overwrite",
            configuration=_CDF_CONFIG,
            partition_by=partition_list,
            writer_properties=WriterProperties(),
        )
        return

    # Upsert via full-table rewrite (anti-join + overwrite) rather than Delta MERGE,
    # which delta-rs 1.6 executes unreliably on S3 targets at scale. Match on the same
    # columns the MERGE predicate used: merge keys plus any partition columns.
    join_cols = keys + (partition_list or [])
    _upsert_overwrite(target, _df, join_cols, partition_list)


# ── Source protocol and built-in implementations ──────────────────────────────


@dataclass(frozen=True)
class Batch:
    """A batch of data returned by a :class:`Source`, paired with its cursor.

    Attributes:
        frame:  Lazy representation of the new data.
        cursor: JSON-serialisable value representing what was consumed.
                Passed back as ``since`` on the next :meth:`Source.poll` call.
                ``None`` means the source always does a full refresh.
    """

    frame: pl.LazyFrame
    cursor: object


@runtime_checkable
class Source(Protocol):
    """Protocol for incremental data sources.

    Implement ``poll`` to teach :class:`IncrementalPipeline` how to fetch new
    data and what cursor to store so the next run picks up where this one left
    off.
    """

    def poll(self, since: object | None) -> Batch | None:
        """Return new data since *since*, or ``None`` when up to date.

        Args:
            since: The cursor saved by the previous run, or ``None`` on first run.

        Returns:
            A :class:`Batch` with the new data and the next cursor,
            or ``None`` if nothing has changed since *since*.
        """
        ...


class QuerySource:
    """Source driven by a user function and a column high-water-mark cursor.

    The function receives the last cursor value (or ``None`` on first run) and
    should return a :class:`~polars.LazyFrame` containing only rows that are
    new since that cursor, or ``None`` if there is nothing new.  The next
    cursor is computed as the maximum value of *cursor_on* in the returned
    frame.

    .. note::
        Only the cursor maximum is materialised here; the frame itself stays
        lazy and is streamed by the downstream sink.  For large file-based
        sources prefer implementing :class:`Source` directly.
    """

    def __init__(
        self,
        fn: Callable[[object | None], pl.LazyFrame | None],
        cursor_on: str,
    ) -> None:
        self._fn = fn
        self._cursor_on = cursor_on

    def poll(self, since: object | None) -> Batch | None:
        lf = self._fn(since)
        if lf is None:
            return None
        _cursor_df = cast(
            pl.DataFrame,
            lf.select(pl.col(self._cursor_on).max()).collect(engine="streaming"),
        )
        cursor = _cursor_df.item() if _cursor_df.height else None
        if cursor is None:
            return None
        return Batch(frame=lf, cursor=cursor)


class FrameSource:
    """Source that returns the result of a no-argument factory on every run.

    The cursor is never advanced, so ``run()`` always ingests the full result.
    Requires ``merge_on`` — without it every run appends a duplicate copy.
    """

    def __init__(self, fn: Callable[[], pl.LazyFrame]) -> None:
        self._fn = fn

    def poll(self, since: object | None) -> Batch | None:  # noqa: ARG002
        return Batch(frame=self._fn(), cursor=None)


class _DirSource:
    """Internal :class:`Source` adapter for a single directory of files.

    Cursor is the sorted list of file paths already processed from this
    directory.  Each ``poll`` lists the directory, drops paths already in the
    cursor, and returns only the new files — so a list of ``_DirSource`` slots
    fans in exactly like the rest of the source types.
    """

    def __init__(
        self,
        root: str,
        file_format: FileFormat,
        suffixes: tuple[str, ...],
        reader_kwargs: dict | None,
        concat_options: dict | None,
    ) -> None:
        self._root = root
        self._file_format = file_format
        self._suffixes = suffixes
        self._reader_kwargs = reader_kwargs
        self._concat_options = concat_options

    def poll(self, since: object | None) -> Batch | None:
        processed = set(since) if isinstance(since, list) else set()
        if self._root.startswith("s3://"):
            all_files = _list_s3_files(self._root, self._suffixes)
        else:
            all_files = _list_local_files(self._root, self._suffixes)
        new_files = sorted(p for p in all_files if p not in processed)
        if not new_files:
            return None
        frame = _scan_files(new_files, self._file_format, self._reader_kwargs, self._concat_options)
        return Batch(frame=frame, cursor=sorted(processed | set(new_files)))


class _DeltaCdfSource:
    """Internal :class:`Source` adapter for a single Delta table read via CDF.

    Cursor is the last processed Delta table version.  First poll reads the full
    table; later polls read only the Change Data Feed since the stored version.
    """

    def __init__(self, source_path: str) -> None:
        self._source_path = source_path

    def poll(self, since: object | None) -> Batch | None:
        from_version = int(since) if isinstance(since, int) else None
        current_version = DeltaTable(self._source_path).version()
        if from_version is not None and from_version >= current_version:
            return None
        frame, current_version = _read_delta_source(self._source_path, from_version)
        return Batch(frame=frame, cursor=current_version)


def from_query(
    fn: Callable[[object | None], pl.LazyFrame | None],
    cursor_on: str,
) -> QuerySource:
    """Create a :class:`QuerySource` driven by *fn* and a column high-water-mark.

    Args:
        fn:        Called with the last cursor (or ``None``) each run.
                   Return a :class:`~polars.LazyFrame` of new rows, or ``None``
                   when there is nothing new.
        cursor_on: Column whose ``max`` value becomes the next cursor.

    Example:
        HTTP source that skips re-processing the same month::

            def fetch_prices(since: str | None) -> pl.LazyFrame | None:
                stamp = datetime.now().strftime("%Y%m")
                if since == stamp:
                    return None
                resp = httpx.get("https://api.example.com/prices.csv")
                return pl.scan_csv(io.BytesIO(resp.content)).with_columns(
                    pl.lit(stamp).alias("_stamp")
                )

            @incremental(
                source=from_query(fetch_prices, cursor_on="_stamp"),
                target="/data/delta/prices",
                merge_on="id",
            )
            def prices(lf: pl.LazyFrame) -> pl.LazyFrame:
                return lf.drop("_stamp")
    """
    return QuerySource(fn, cursor_on)


def from_frame(fn: Callable[[], pl.LazyFrame]) -> FrameSource:
    """Create a :class:`FrameSource` that calls *fn* on every run (full refresh).

    Requires ``merge_on`` — without it every run appends a duplicate copy of
    the data.

    Args:
        fn: No-argument factory that returns a :class:`~polars.LazyFrame`.
    """
    return FrameSource(fn)


# ── Public API ────────────────────────────────────────────────────────────────


def incremental(
    source: str | Source | list[str | Source],
    target: str,
    merge_on: str | list[str] | None = None,
    fail_on_version_mismatch: bool = False,
    file_format: FileFormat = "parquet",
    watermark_store: str | None = None,
    partition_by: str | list[str] | None = None,
    scd_type: Literal[1, 2, 4] = 1,
    history_target: str | None = None,
    compact_every: int | None = None,
    compute_context: object | None = None,
    staging: str | None = None,
    reader_kwargs: dict | None = None,
    concat_options: dict | None = None,
) -> Callable:
    """Wrap a ``LazyFrame → LazyFrame`` function as an :class:`IncrementalPipeline`.

    Each call to :meth:`IncrementalPipeline.run` polls every source for new data,
    passes the active frames to the transform, writes the output Delta table, and
    saves a per-source cursor so the next run picks up exactly where this one left
    off.

    Args:
        source: Where to read from. Accepts:

            * **A directory path** (``str``) — scans for new files matching
              *file_format* on every run.
            * **A Delta table path** (``str``) with ``file_format="delta"`` — reads
              only changed rows via the Change Data Feed.
            * **A** :class:`Source` **object** (e.g. from :func:`from_query` or
              :func:`from_frame`) — delegates listing and cursor tracking to the
              object's :meth:`~Source.poll` method.
            * **A list of any of the above** — fans in multiple sources; each slot
              keeps an independent cursor and the transform receives one frame per
              active source as positional arguments.

        target:          Output Delta table path (local or ``s3://``).
        merge_on:        Upsert key column(s). ``None`` for append-only. Upserts
                         rewrite only the data a batch touches (never via Delta
                         ``MERGE``); on a partitioned target only the affected
                         partitions are rewritten, so partitioned tables scale to
                         arbitrary size.
        fail_on_version_mismatch: If ``True``, the pipeline will raise a
            ``RuntimeError`` at run start when the stored pipeline hash
            (from a previous run) differs from the current code hash. When
            ``False`` (default) a warning is emitted instead. The pipeline
            hash is computed with ``joblib.hash`` and persisted to
            ``<watermark>/.pipeline_version`` on successful runs.
        file_format:     ``"parquet"`` (default) | ``"csv"`` | ``"ndjson"`` | ``"delta"``.
        watermark_store: Watermark table path. Defaults to ``target + "/.watermark"``.
        partition_by:    Partition column(s). Fixed at table creation. Also bounds
                         upsert write cost to the touched partitions — partition the
                         target to upsert efficiently into a large table.
        scd_type:        ``1`` (default, latest state), ``2`` (``valid_from``/``valid_to``
                         history), or ``4`` (separate history table).
        history_target:  History table path. Required when ``scd_type=4``.
        compact_every:   Run OPTIMIZE + VACUUM every N successful runs. ``None`` to
                         disable.
        compute_context: ``polars_cloud.ComputeContext`` for remote execution.
        staging:         Unused. Kept for backwards compatibility.
        reader_kwargs:   Forwarded to the file scanner (``pl.scan_parquet`` etc.).
        concat_options:  Forwarded to ``pl.concat`` when combining files.

    Examples:
        Basic upsert — process new Parquet files and merge on ``id``::

            @incremental(source="/data/uploads/", target="/data/delta/clean",
                         merge_on="id")
            def clean(lf: pl.LazyFrame) -> pl.LazyFrame:
                return lf.filter(pl.col("value") > 0)

            new_files = clean.run()

        Append-only (no deduplication)::

            @incremental(source="/data/logs/", target="/data/delta/events")
            def ingest(lf: pl.LazyFrame) -> pl.LazyFrame:
                return lf

        Delta source — read only changed rows via CDF::

            @incremental(source="/data/delta/source", target="/data/delta/sink",
                         merge_on="id", file_format="delta")
            def propagate(lf: pl.LazyFrame) -> pl.LazyFrame:
                return lf

        HTTP source via :func:`from_query` — cursor is the max of a column::

            def fetch(since: str | None) -> pl.LazyFrame | None:
                stamp = datetime.now().strftime("%Y%m")
                if since == stamp:
                    return None
                return pl.read_csv(download()).with_columns(
                    pl.lit(stamp).alias("_stamp")
                ).lazy()

            @incremental(
                source=from_query(fetch, cursor_on="_stamp"),
                target="/data/delta/dataset",
                merge_on="id",
            )
            def dataset(lf: pl.LazyFrame) -> pl.LazyFrame:
                return lf.drop("_stamp")

        Full-refresh reference table via :func:`from_frame`::

            @incremental(
                source=from_frame(lambda: pl.scan_csv("https://example.com/ref.csv")),
                target="/data/delta/reference",
                merge_on="code",
            )
            def reference(lf: pl.LazyFrame) -> pl.LazyFrame:
                return lf

        Fan-in — multiple sources, each with its own cursor; use ``*lfs`` because
        idle sources are omitted from the argument list::

            @incremental(
                source=["/data/region_a/", "/data/region_b/"],
                target="/data/delta/combined",
                merge_on="id",
            )
            def combine(*lfs: pl.LazyFrame) -> pl.LazyFrame:
                return pl.concat(list(lfs))

        Mixed fan-in — directory and HTTP source in one pipeline::

            @incremental(
                source=["/data/uploads/", from_query(fetch, cursor_on="_stamp")],
                target="/data/delta/combined",
                merge_on="id",
            )
            def combine(*lfs: pl.LazyFrame) -> pl.LazyFrame:
                return pl.concat(list(lfs))

        SCD Type 2 — keep full row history with ``valid_from``/``valid_to``::

            @incremental(source="/data/uploads/", target="/data/delta/history",
                         merge_on="id", scd_type=2)
            def track(lf: pl.LazyFrame) -> pl.LazyFrame:
                return lf

        SCD Type 4 — current state in one table, history in another::

            @incremental(
                source="/data/uploads/",
                target="/data/delta/current",
                history_target="/data/delta/history",
                merge_on="id",
                scd_type=4,
            )
            def track(lf: pl.LazyFrame) -> pl.LazyFrame:
                return lf

        Auto-compact every 10 runs::

            @incremental(source="/data/uploads/", target="/data/delta/clean",
                         merge_on="id", compact_every=10)
            def clean(lf: pl.LazyFrame) -> pl.LazyFrame:
                return lf
    """
    _watermark = watermark_store or target.rstrip("/") + "/.watermark"

    def decorator(fn: Callable) -> "IncrementalPipeline":
        return IncrementalPipeline(
            fn=fn,
            source=source,
            target=target,
            merge_on=merge_on,
            fail_on_version_mismatch=fail_on_version_mismatch,
            file_format=file_format,
            watermark_store=_watermark,
            partition_by=partition_by,
            scd_type=scd_type,
            history_target=history_target,
            compact_every=compact_every,
            compute_context=compute_context,
            staging=staging,
            reader_kwargs=reader_kwargs,
            concat_options=concat_options,
        )

    return decorator


class IncrementalPipeline:
    """Incremental Polars pipeline. Returned by :func:`incremental`; do not instantiate directly.

    Examples:
        Basic usage::

            @incremental(source="/data/uploads/", target="/data/delta/clean",
                         merge_on="id", compact_every=10)
            def clean(lf: pl.LazyFrame) -> pl.LazyFrame:
                return lf.filter(pl.col("value") > 0)

            clean.run()                   # ingest new files
            clean.run(dry_run=True)       # preview new files without writing
            clean.status()                # watermark table as a DataFrame
            clean.maintain(vacuum=True)   # compact small files and vacuum
            clean.reset()                 # clear watermark; next run reprocesses all

        Fan-in from multiple sources::

            @incremental(
                source=["/data/region_a/", "/data/region_b/"],
                target="/data/delta/combined",
                merge_on="id",
            )
            def combine(lf_a: pl.LazyFrame, lf_b: pl.LazyFrame) -> pl.LazyFrame:
                return pl.concat([lf_a, lf_b])
    """

    def __init__(
        self,
        fn: Callable[..., pl.LazyFrame],
        source: str | Source | list[str | Source],
        target: str,
        merge_on: str | list[str] | None = None,
        file_format: FileFormat = "parquet",
        watermark_store: str = "",
        partition_by: str | list[str] | None = None,
        scd_type: Literal[1, 2, 4] = 1,
        history_target: str | None = None,
        compact_every: int | None = None,
        compute_context: object | None = None,
        staging: str | None = None,
        reader_kwargs: dict | None = None,
        concat_options: dict | None = None,
        fail_on_version_mismatch: bool = False,
    ) -> None:
        """Validate config and set attributes.

        Raises:
            ValueError: If ``scd_type=4`` and ``history_target`` is not set.
            ValueError: If ``merge_on`` is an empty list.
        """
        if isinstance(merge_on, list) and len(merge_on) == 0:
            raise ValueError("merge_on must not be an empty list")
        if scd_type == 4 and history_target is None:
            raise ValueError("history_target is required when scd_type=4")

        self.fn = fn
        self.source = source
        self.target = target
        self.merge_on = merge_on
        self.file_format = file_format
        self.watermark_store = watermark_store or target.rstrip("/") + "/.watermark"
        self.partition_by = partition_by
        self.scd_type = scd_type
        self.history_target = history_target
        self.compact_every = compact_every
        self.compute_context = compute_context
        self.staging = staging
        self.reader_kwargs = reader_kwargs
        self.concat_options = concat_options
        self.fail_on_version_mismatch = fail_on_version_mismatch

        # Normalize every source — directory, Delta table, or custom Source —
        # into a uniform list of Source adapters so all source types fan in.
        suffix = _SUFFIXES.get(file_format, "")
        suffixes = (suffix,) if suffix else ()

        def _as_source(item: str | Source) -> Source:
            if isinstance(item, Source):
                return item
            if file_format == "delta":
                return _DeltaCdfSource(item)
            return _DirSource(item, file_format, suffixes, reader_kwargs, concat_options)

        items = cast("list[str | Source]", source if isinstance(source, list) else [source])
        self._sources: list[Source] = [_as_source(it) for it in items]

        # Preserve wrapped function metadata.
        self.__wrapped__ = fn
        self.__name__ = getattr(fn, "__name__", repr(fn))
        self.__doc__ = fn.__doc__

        # Pipeline version hash (joblib.hash covers function code and closure).
        try:
            self.pipeline_hash = joblib.hash(fn)
        except Exception:
            # Fallback: best-effort using function name if hashing fails.
            self.pipeline_hash = getattr(fn, "__name__", repr(fn))
        # honor passed parameter (already set above)

    def __call__(self, *lfs: pl.LazyFrame) -> pl.LazyFrame:
        """Call the wrapped transform function directly, bypassing the incremental machinery."""
        return self.fn(*lfs)

    def _labels(self, slot: int, before: object, batch: Batch) -> list[str]:
        """Return human-readable identifiers of what *batch* processed for *slot*.

        Preserves the historical per-source return contract: new file paths for a
        directory source, ``v{version}`` for a Delta source, the cursor repr otherwise.
        """
        src = self._sources[slot]
        if isinstance(src, _DirSource):
            before_set = set(before) if isinstance(before, list) else set()
            cursor = batch.cursor if isinstance(batch.cursor, list) else []
            return [str(p) for p in cursor if p not in before_set]
        if isinstance(src, _DeltaCdfSource):
            return [f"v{batch.cursor}"]
        return [str(batch.cursor)[:100]]

    def run(self, dry_run: bool = False) -> list[str]:
        """Ingest new data, apply the transform, write to the target, and save the watermark.

        Polls every source, omits those with nothing new (so the transform's
        argument count matches the number of active sources), runs the transform
        on the remaining frames, writes the target, and saves all cursors.

        Args:
            dry_run: Log what would be processed without reading or writing.

        Returns:
            Identifiers of what was processed: new file paths for directory
            sources, ``["v{version}"]`` per Delta source, ``[cursor_repr]`` per
            custom :class:`Source`. ``[]`` when every source is up to date.

        Notes:
            This method performs a pipeline-version check before executing
            writes. The computed pipeline hash is persisted under
            ``<watermark>/.pipeline_version``; if a previous value exists and
            differs from the current hash the pipeline will either log a
            warning or raise ``RuntimeError`` depending on the
            ``fail_on_version_mismatch`` setting supplied to
            :func:`incremental`.
        """
        cursors = _load_cursors(self.watermark_store)
        batches: list[tuple[int, Batch]] = []
        for i, src in enumerate(self._sources):
            batch = src.poll(cursors.get(i))
            if batch is not None:
                batches.append((i, batch))

        if not batches:
            logger.info("No new data — nothing to do.")
            return []

        labels = [lbl for i, b in batches for lbl in self._labels(i, cursors.get(i), b)]
        logger.info("%d source(s) with new data.", len(batches))

        if dry_run:
            for lbl in labels:
                logger.info("  [dry_run] %s", lbl)
            return labels

        # Check stored pipeline version and warn if it differs from current.
        try:
            stored = self._load_pipeline_version()
            if stored is not None and stored != self.pipeline_hash:
                msg = (
                    f"Pipeline version changed (stored={stored} current={self.pipeline_hash}). "
                    "This may make existing watermark/cursor incompatible."
                )
                if self.fail_on_version_mismatch:
                    raise RuntimeError(msg)
                logger.warning(msg)
        except Exception as exc:
            # If user asked to fail on mismatch, propagate RuntimeError; otherwise log debug.
            if isinstance(exc, RuntimeError):
                raise
            logger.debug("Unable to read stored pipeline version.")

        frames = [b.frame for _, b in batches]
        result_lf = self.fn(*frames)

        if self.compute_context is not None:
            from .cloud import _sink_target_remote  # noqa: PLC0415

            _sink_target_remote(
                self.target,
                result_lf,
                self.merge_on,
                self.compute_context,
                self.partition_by,
            )
        elif self.scd_type == 2:
            assert self.merge_on is not None, "merge_on is required for scd_type=2"
            _sink_scd2(self.target, result_lf, self.merge_on, self.partition_by)
        elif self.scd_type == 4:
            assert self.merge_on is not None, "merge_on is required for scd_type=4"
            assert self.history_target is not None
            _sink_scd4(
                self.target, self.history_target, result_lf, self.merge_on, self.partition_by
            )
        else:
            _sink_target(self.target, result_lf, self.merge_on, self.partition_by)

        new_cursors = {**cursors, **{i: b.cursor for i, b in batches}}
        _save_cursors(self.watermark_store, new_cursors)

        # Persist pipeline version info alongside watermark.
        try:
            self._save_pipeline_version()
        except Exception:
            logger.exception("Failed to persist pipeline version.")

        if self.compact_every is not None:
            count = _load_run_count(self.watermark_store) + 1
            _save_run_count(self.watermark_store, count)
            if count % self.compact_every == 0:
                maintain(self.target)

        logger.info("Processed %d source(s).", len(batches))
        return labels

    def reset(self) -> None:
        """Delete all watermark rows so the next ``run()`` reprocesses all files."""
        try:
            dt = DeltaTable(self.watermark_store)
            dt.delete()
            logger.info("Watermark cleared.")
        except Exception:
            logger.info("No watermark found — nothing to reset.")

    def status(self) -> pl.DataFrame:
        """Return the watermark table as an eager DataFrame."""
        result = cast(pl.DataFrame, pl.scan_delta(self.watermark_store).collect())
        return result

    def maintain(
        self,
        compact: bool = True,
        z_order_by: str | list[str] | None = None,
        vacuum: bool = True,
        retention_hours: int = 168,
    ) -> None:
        """Compact and/or vacuum the target Delta table.

        Delegates to :func:`maintenance.maintain`.
        """
        maintain(
            self.target,
            compact=compact,
            z_order_by=z_order_by,
            vacuum=vacuum,
            retention_hours=retention_hours,
        )

    # ---------------------
    # Pipeline version helpers
    # ---------------------
    def _pipeline_version_path(self) -> str:
        return self.watermark_store.rstrip("/") + "/.pipeline_version"

    def _save_pipeline_version(self) -> None:
        # Write a tiny Delta table with a single row containing pipeline hash

        path = self._pipeline_version_path()
        df = pl.DataFrame({"pipeline_hash": [self.pipeline_hash]})
        # Use write_deltalake and overwrite any previous value.
        write_deltalake(path, df.to_arrow(), mode="overwrite", schema_mode="overwrite")

    def _load_pipeline_version(self) -> str | None:
        try:
            dt = DeltaTable(self._pipeline_version_path())
            tbl = dt.to_pyarrow_table()
            df = pl.from_arrow(tbl)
            # Ensure a DataFrame (pyarrow may produce a Series for single-column tables)
            df = pl.DataFrame(df)
            if len(df) == 0:
                return None
            vals = df["pipeline_hash"].to_list()
            return str(vals[0])
        except Exception:
            return None
