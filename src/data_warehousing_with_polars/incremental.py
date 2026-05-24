"""
incremental.py

Public API: the :func:`incremental` decorator and :class:`IncrementalPipeline`.
Coordinates file listing, watermark tracking, transform dispatch, and compaction.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

import polars as pl
from deltalake import DeltaTable, WriterProperties, write_deltalake
from deltalake.exceptions import TableNotFoundError

from .maintenance import _load_run_count, _save_run_count, maintain
from .record_batch_source import make_lazy
from .scd import _sink_scd2, _sink_scd4

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


def _load_watermark(store: str) -> set[str]:
    """Return file paths recorded in the watermark table. Returns ``set()`` on first run."""
    try:
        result = pl.scan_delta(store).select("file_path").collect()
        assert isinstance(result, pl.DataFrame)
        return set(result["file_path"].to_list())
    except Exception:
        return set()


def _save_watermark(store: str, paths: list[str]) -> None:
    """Append *paths* with an ``ingested_at`` timestamp to the watermark table."""
    rows = pl.DataFrame(
        {
            "file_path": paths,
            "ingested_at": [datetime.now(timezone.utc)] * len(paths),
        }
    )
    write_deltalake(store, rows, mode="append")


def _load_delta_watermark(store: str) -> int | None:
    """Return the last processed Delta source version, or ``None`` on first run."""
    try:
        result = pl.scan_delta(store).select(pl.col("version").max()).collect()
        assert isinstance(result, pl.DataFrame)
        val = result["version"][0]
        return int(val) if val is not None else None
    except Exception:
        return None


def _save_delta_watermark(store: str, version: int) -> None:
    """Append *version* with a ``committed_at`` timestamp to the watermark table."""
    rows = pl.DataFrame(
        {
            "version": [version],
            "committed_at": [datetime.now(timezone.utc)],
        }
    )
    write_deltalake(store, rows, mode="append")


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
            _df = lf.collect()
            assert isinstance(_df, pl.DataFrame)
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

    _df = lf.collect(engine="streaming")
    assert isinstance(_df, pl.DataFrame)

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

    predicate = " AND ".join(f"target.{k} = source.{k}" for k in keys)
    if partition_list:
        partition_pred = " AND ".join(f"target.{k} = source.{k}" for k in partition_list)
        predicate = f"{predicate} AND {partition_pred}"

    dt = DeltaTable(target)
    (
        dt.merge(_df.to_arrow(), predicate=predicate, source_alias="source", target_alias="target")
        .when_matched_update_all()
        .when_not_matched_insert_all()
        .execute()
    )


# ── Public API ────────────────────────────────────────────────────────────────


def incremental(
    source: str | list[str],
    target: str,
    merge_on: str | list[str] | None = None,
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

    Args:
        source:          Source directory or list of directories.
        target:          Output Delta table path.
        merge_on:        Upsert key column(s). ``None`` for append-only.
        file_format:     ``"parquet"`` | ``"csv"`` | ``"ndjson"`` | ``"delta"``.
        watermark_store: Watermark table path. Defaults to ``target + "/.watermark"``.
        partition_by:    Partition column(s). Fixed at table creation.
        scd_type:        ``1`` (default), ``2`` (valid_from/valid_to), or ``4``
                         (separate history table).
        history_target:  History table path. Required when ``scd_type=4``.
        compact_every:   Run compaction every N successful runs. ``None`` to disable.
        compute_context: ``polars_cloud.ComputeContext`` for remote execution.
        staging:         S3 staging path. Required when ``compute_context`` is set.
        reader_kwargs:   Forwarded to the file scanner.
        concat_options:  Forwarded to ``pl.concat``.

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

        Fan-in from multiple source directories — each source is a separate argument::

            @incremental(
                source=["/data/region_a/", "/data/region_b/"],
                target="/data/delta/combined",
                merge_on="id",
            )
            def combine(lf_a: pl.LazyFrame, lf_b: pl.LazyFrame) -> pl.LazyFrame:
                return pl.concat([lf_a, lf_b])

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

        Delta source — read only changed rows via CDF::

            @incremental(source="/data/delta/source", target="/data/delta/sink",
                         merge_on="id", file_format="delta")
            def propagate(lf: pl.LazyFrame) -> pl.LazyFrame:
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

    Example::

        @incremental(source="/data/uploads/", target="/data/delta/clean",
                     merge_on="id", compact_every=10)
        def clean(lf: pl.LazyFrame) -> pl.LazyFrame:
            return lf.filter(pl.col("value") > 0)

        clean.run()                   # ingest new files
        clean.run(dry_run=True)       # preview new files without writing
        clean.status()                # watermark table as a DataFrame
        clean.maintain(vacuum=True)   # compact small files and vacuum
        clean.reset()                 # clear watermark; next run reprocesses all

    Fan-in example::

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
        source: str | list[str],
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
    ) -> None:
        """Validate config and set attributes.

        Raises:
            ValueError: If ``scd_type=4`` and ``history_target`` is not set.
            ValueError: If ``compute_context`` is set and ``staging`` is not.
            ValueError: If ``merge_on`` is an empty list.
        """
        if isinstance(merge_on, list) and len(merge_on) == 0:
            raise ValueError("merge_on must not be an empty list")
        if scd_type == 4 and history_target is None:
            raise ValueError("history_target is required when scd_type=4")
        if compute_context is not None and staging is None:
            raise ValueError("staging is required when compute_context is set")

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

        # Derived attributes computed once.
        self._sources: list[str] = [source] if isinstance(source, str) else list(source)
        self._suffix: str = _SUFFIXES.get(file_format, "")
        self._suffixes: tuple[str, ...] = (self._suffix,) if self._suffix else ()

        # Preserve wrapped function metadata.
        self.__wrapped__ = fn
        self.__name__ = getattr(fn, "__name__", repr(fn))
        self.__doc__ = fn.__doc__

    def __call__(self, *lfs: pl.LazyFrame) -> pl.LazyFrame:
        """Call the wrapped transform function directly, bypassing the incremental machinery."""
        return self.fn(*lfs)

    def _new_files(self) -> list[str]:
        """Return sorted paths of source files not yet in the watermark."""
        processed = _load_watermark(self.watermark_store)
        all_files: list[str] = []
        for src in self._sources:
            all_files.extend(_list_local_files(src, self._suffixes))
        return sorted(p for p in all_files if p not in processed)

    def _scan_new(self, new_files: list[str]) -> list[pl.LazyFrame]:
        """Return one LazyFrame per source directory that has files in *new_files*."""
        frames: list[pl.LazyFrame] = []
        for src in self._sources:
            src_root = str(Path(src).resolve())
            src_files = [f for f in new_files if f.startswith(src_root)]
            if src_files:
                frames.append(
                    _scan_files(
                        src_files, self.file_format, self.reader_kwargs, self.concat_options
                    )
                )
        return frames

    def run(self, dry_run: bool = False) -> list[str]:
        """Ingest new files, apply the transform, write to the target, and save the watermark.

        Args:
            dry_run: Log new files without reading or writing. Returns the same file list.

        Returns:
            Processed file paths, or ``["v{version}"]`` for ``file_format="delta"``.
        """
        if self.file_format == "delta":
            return self._run_delta(dry_run)

        new_files = self._new_files()
        if not new_files:
            logger.info("No new files — nothing to do.")
            return []

        logger.info("%d new file(s) found.", len(new_files))
        if dry_run:
            for f in new_files:
                logger.info("  [dry_run] %s", f)
            return new_files

        frames = self._scan_new(new_files)
        result_lf = self.fn(*frames)

        if self.compute_context is not None:
            from .cloud import _sink_target_remote  # noqa: PLC0415

            assert self.staging is not None
            _sink_target_remote(
                self.target,
                result_lf,
                self.merge_on,
                self.compute_context,
                self.staging,
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

        _save_watermark(self.watermark_store, new_files)

        if self.compact_every is not None:
            count = _load_run_count(self.watermark_store) + 1
            _save_run_count(self.watermark_store, count)
            if count % self.compact_every == 0:
                maintain(self.target)

        logger.info("Processed %d file(s).", len(new_files))
        return new_files

    def _run_delta(self, dry_run: bool) -> list[str]:
        """Run one Delta CDF increment; return ``["v{version}"]`` or ``[]`` when up to date."""
        source_path = self._sources[0]
        from_version = _load_delta_watermark(self.watermark_store)
        dt = DeltaTable(source_path)
        current_version = dt.version()

        if from_version is not None and from_version >= current_version:
            logger.info("Delta source at version %d — nothing to do.", current_version)
            return []

        version_label = f"v{current_version}"
        range_desc = (
            f"{from_version + 1}..{current_version}"
            if from_version is not None
            else f"0..{current_version} (initial load)"
        )
        logger.info("Reading Delta CDF %s.", range_desc)

        if dry_run:
            logger.info("  [dry_run] %s", version_label)
            return [version_label]

        lf, current_version = _read_delta_source(source_path, from_version)
        result_lf = self.fn(lf)

        if self.scd_type == 2:
            assert self.merge_on is not None
            _sink_scd2(self.target, result_lf, self.merge_on, self.partition_by)
        elif self.scd_type == 4:
            assert self.merge_on is not None
            assert self.history_target is not None
            _sink_scd4(
                self.target, self.history_target, result_lf, self.merge_on, self.partition_by
            )
        else:
            _sink_target(self.target, result_lf, self.merge_on, self.partition_by)

        _save_delta_watermark(self.watermark_store, current_version)

        if self.compact_every is not None:
            count = _load_run_count(self.watermark_store) + 1
            _save_run_count(self.watermark_store, count)
            if count % self.compact_every == 0:
                maintain(self.target)

        return [version_label]

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
        result = pl.scan_delta(self.watermark_store).collect()
        assert isinstance(result, pl.DataFrame)
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
