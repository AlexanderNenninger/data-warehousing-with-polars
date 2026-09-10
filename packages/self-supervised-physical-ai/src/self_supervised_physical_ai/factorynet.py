"""Streams FactoryNet straight from Hugging Face into Delta tables on S3, one per
schema group, via `data_warehousing_with_polars`'s `@incremental` + `@schema`.

FactoryNet (huggingface.co/datasets/factorynet/factorynet) unifies several public
industrial-robotics datasets — VORAUS-AD, AURSAD, a CNC run, the FactoryWave UR/KUKA
captures, and paired sim/real episodes — into one repo, each source as its own
parquet shard(s) under `data/`. This was verified directly against the published
files (not just the paper's schema description): columns follow a Setpoint /
Effort / Feedback / Context (S-E-F-C) naming convention (`setpoint_*`, `effort_*`,
`feedback_*`, `ctx_*`), plus `auxiliary_*` for extra sensors (e.g. accelerometers).

Sources don't share a schema — each file only populates the prefixed columns its
underlying instrumentation actually recorded, and files that look related by
filename or column *count* aren't guaranteed to match exactly (verified: 120 files
sort into 10 distinct schemas, not one per apparent "family" — see
`group_files_by_schema`). Never `pl.concat`/`scan_parquet` files across groups;
`build_incremental_pipeline` ingests one group at a time, each into its own table.

Every source file also carries `episode_id`, `time_s` (seconds since episode
start — there's no separate sample-rate field; derive it from `time_s` diffs),
`dataset_source`, and `machine_type`; most (not `factorywave_data_episode_metadata`
/ `factorywave_kukadata_episode_metadata`, which are metadata-only, and the
`simulations_*` shards, which lack `ctx_anomaly_label`) also carry an
`ctx_anomaly_label` / `ctx_anomaly_category` pair from the underlying
anomaly-detection benchmarks (currently populated by voraus/aursad).
"""

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import polars as pl
from data_warehousing_with_polars import Batch, incremental
from data_warehousing_with_polars.incremental import IncrementalPipeline
from data_warehousing_with_polars.schema import schema as validate_schema

HF_DATASET_REPO = "factorynet/factorynet"
HF_RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main/data/"

# Full `data/*.parquet` listing as of 2026-07 (120 files, ~13.7 GB total). Source:
# `curl https://huggingface.co/api/datasets/factorynet/factorynet/tree/main/data?recursive=true`
FACTORYNET_FILES: tuple[str, ...] = (
    "aursad_001.parquet",
    "aursad_002.parquet",
    "aursad_003.parquet",
    "aursad_004.parquet",
    "aursad_005.parquet",
    "aursad_006.parquet",
    "aursad_007.parquet",
    "aursad_008.parquet",
    "aursad_009.parquet",
    "aursad_010.parquet",
    "aursad_011.parquet",
    "aursad_012.parquet",
    "aursad_013.parquet",
    "aursad_014.parquet",
    "aursad_015.parquet",
    "aursad_016.parquet",
    "aursad_017.parquet",
    "aursad_018.parquet",
    "aursad_019.parquet",
    "aursad_020.parquet",
    "aursad_021.parquet",
    "aursad_022.parquet",
    "aursad_023.parquet",
    "aursad_024.parquet",
    "aursad_025.parquet",
    "aursad_026.parquet",
    "aursad_027.parquet",
    "aursad_028.parquet",
    "aursad_029.parquet",
    "aursad_030.parquet",
    "aursad_031.parquet",
    "aursad_032.parquet",
    "cnc_000.parquet",
    "factorywave_data_episode_metadata.parquet",
    "factorywave_data_flow_metadata.parquet",
    "factorywave_data_ur_screwdriver.parquet",
    "factorywave_kukadata_episode_metadata.parquet",
    "factorywave_kukadata_flow_metadata.parquet",
    "factorywave_ur_consolidated.parquet",
    "simulations_baseline_part_000.parquet",
    "simulations_baseline_part_001.parquet",
    "simulations_baseline_part_002.parquet",
    "simulations_baseline_part_003.parquet",
    "simulations_baseline_part_004.parquet",
    "simulations_baseline_part_005.parquet",
    "simulations_baseline_part_006.parquet",
    "simulations_baseline_part_007.parquet",
    "simulations_baseline_part_008.parquet",
    "simulations_baseline_part_009.parquet",
    "simulations_baseline_part_010.parquet",
    "simulations_counterfactual_part_000.parquet",
    "simulations_counterfactual_part_001.parquet",
    "simulations_counterfactual_part_002.parquet",
    "simulations_counterfactual_part_003.parquet",
    "simulations_counterfactual_part_004.parquet",
    "simulations_counterfactual_part_005.parquet",
    "simulations_counterfactual_part_006.parquet",
    "simulations_counterfactual_part_007.parquet",
    "simulations_counterfactual_part_008.parquet",
    "simulations_counterfactual_part_009.parquet",
    "voraus_001.parquet",
    "voraus_002.parquet",
    "voraus_003.parquet",
    "voraus_004.parquet",
    "voraus_005.parquet",
    "voraus_006.parquet",
    "voraus_007.parquet",
    "voraus_008.parquet",
    "voraus_009.parquet",
    "voraus_010.parquet",
    "voraus_011.parquet",
    "voraus_012.parquet",
    "voraus_013.parquet",
    "voraus_014.parquet",
    "voraus_015.parquet",
    "voraus_016.parquet",
    "voraus_017.parquet",
    "voraus_018.parquet",
    "voraus_019.parquet",
    "voraus_020.parquet",
    "voraus_021.parquet",
    "voraus_022.parquet",
    "voraus_023.parquet",
    "voraus_024.parquet",
    "voraus_025.parquet",
    "voraus_026.parquet",
    "voraus_027.parquet",
    "voraus_028.parquet",
    "voraus_029.parquet",
    "voraus_030.parquet",
    "voraus_031.parquet",
    "voraus_032.parquet",
    "voraus_033.parquet",
    "voraus_034.parquet",
    "voraus_035.parquet",
    "voraus_036.parquet",
    "voraus_037.parquet",
    "voraus_038.parquet",
    "voraus_039.parquet",
    "voraus_040.parquet",
    "voraus_041.parquet",
    "voraus_042.parquet",
    "voraus_043.parquet",
    "voraus_044.parquet",
    "voraus_045.parquet",
    "voraus_046.parquet",
    "voraus_047.parquet",
    "voraus_048.parquet",
    "voraus_049.parquet",
    "voraus_050.parquet",
    "voraus_051.parquet",
    "voraus_052.parquet",
    "voraus_053.parquet",
    "voraus_054.parquet",
    "voraus_055.parquet",
    "voraus_056.parquet",
    "voraus_057.parquet",
    "voraus_058.parquet",
    "voraus_059.parquet",
    "voraus_060.parquet",
)

# Columns present on (most) source files beyond the S-E-F-C channel columns —
# useful for grouping windows into episodes and for labeling discovered clusters.
METADATA_COLUMNS: tuple[str, ...] = (
    "episode_id",
    "time_s",
    "dataset_source",
    "machine_type",
    "ctx_anomaly_label",
    "ctx_anomaly_category",
)

_CHANNEL_PREFIXES = ("setpoint_", "effort_", "feedback_", "auxiliary_")

_NUMERIC_DTYPES = (
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
)


def factorynet_source_url(filename: str) -> str:
    """The public HF `resolve` URL polars can `scan_parquet` directly."""
    return HF_RESOLVE_BASE + filename


class KnownFilesSource:
    """`incremental`'s `Source` protocol over a fixed, known list of file URLs.

    `incremental.py`'s built-in directory source (`_DirSource`) lists S3
    directories via `pyarrow.fs.S3FileSystem`; mixing that with the
    polars/deltalake S3 client used for the write (`sink_delta`) reproducibly
    hung mid-upload in testing — a real incompatibility between the two S3
    clients when both run in one process, not a data or logic bug. This
    sidesteps it entirely: `paths` is a fixed, already-known list (every
    FactoryNet file's HF resolve URL always exists), so there's nothing to list
    or check existence for — only the watermark (which of `paths` this pipeline
    has already processed) determines what's new on each `poll`.
    """

    def __init__(self, paths: Sequence[str]) -> None:
        self._paths = list(paths)

    def poll(self, since: object | None) -> Batch | None:
        processed = set(since) if isinstance(since, list) else set()
        new_files = [p for p in self._paths if p not in processed]
        if not new_files:
            return None
        frame = pl.scan_parquet(new_files, include_file_paths="_source_file")
        return Batch(frame=frame, cursor=sorted(processed | set(new_files)))


def select_channel_columns(
    schema: pl.Schema, prefixes: Sequence[str] = _CHANNEL_PREFIXES
) -> list[str]:
    """Numeric S-E-F-C channel columns this particular schema actually populated
    (excludes the all-Null columns other sources leave behind for schema parity)."""
    prefix_tuple = tuple(prefixes)
    return [
        name
        for name, dtype in schema.items()
        if name.startswith(prefix_tuple) and dtype in _NUMERIC_DTYPES
    ]


def available_metadata_columns(
    schema: pl.Schema, wanted: Sequence[str] = METADATA_COLUMNS
) -> list[str]:
    """The subset of `wanted` metadata columns this schema actually has."""
    return [c for c in wanted if c in schema]


@dataclass
class SchemaGroup:
    """Files that share an exact column-name -> dtype schema.

    Despite sharing a filename prefix or column *count*, FactoryNet's sources
    are not guaranteed to share a schema — `aursad_*`, `simulations_*`, and the
    two standalone `factorywave_*` telemetry files all have 125 columns but are
    four distinct schemas (verified, not assumed). Never `pl.concat` files across
    groups; `build_incremental_pipeline` ingests one group at a time.
    """

    schema: pl.Schema
    files: list[str]


def group_files_by_schema(
    files: Sequence[str] = FACTORYNET_FILES,
    source_url_fn: Callable[[str], str] = factorynet_source_url,
) -> dict[str, SchemaGroup]:
    """Reads every file's schema straight from its source URL (parquet footer
    only, not the data — one lightweight request per file) and groups files that
    match exactly. Keys are `"schema_00"`, `"schema_01"`, ... ordered from most
    to fewest files.
    """
    raw_groups: dict[tuple, list[str]] = {}
    schemas: dict[tuple, pl.Schema] = {}
    for filename in files:
        sch = pl.scan_parquet(source_url_fn(filename)).collect_schema()
        key = tuple(sorted(sch.items()))
        raw_groups.setdefault(key, []).append(filename)
        schemas[key] = sch

    ordered = sorted(raw_groups.items(), key=lambda kv: -len(kv[1]))
    return {
        f"schema_{i:02d}": SchemaGroup(schema=schemas[key], files=sorted(group_files))
        for i, (key, group_files) in enumerate(ordered)
    }


def source_file_basename() -> pl.Expr:
    """The bare filename from the `_source_file` column `incremental`'s
    sources inject automatically (full source URL/path)."""
    return pl.col("_source_file").str.extract(r"([^/]+)$")


def namespaced_episode_id_expr() -> pl.Expr:
    """`episode_id` prefixed with its source filename.

    FactoryNet reuses episode ids *across* files in the same schema group
    (every `voraus_*.parquet` shard has its own `"VORAUS_0".."VORAUS_36"`), so
    without this, ingesting multiple files into one table would silently merge
    unrelated episodes under `FactoryNetWindowDataset`'s per-episode grouping.
    """
    return source_file_basename() + "/" + pl.col("episode_id")


def split_column_expr(train_files: Sequence[str], test_files: Sequence[str]) -> pl.Expr:
    """A `"train"`/`"test"` column derived from `_source_file`, per a file-level
    split decided in advance by `split_files_train_test` — independent of
    ingestion order, so files can be added across multiple `incremental` runs
    without changing any earlier file's split."""
    mapping = {f: "train" for f in train_files} | {f: "test" for f in test_files}
    return source_file_basename().replace_strict(mapping, return_dtype=pl.String)


def split_files_train_test(
    files: Sequence[str], test_size: float = 0.2, seed: int = 0
) -> tuple[list[str], list[str]]:
    """Splits whole files (not rows or episodes) between train/test.

    FactoryNet never splits one episode across files, so a file-level split is
    enough to guarantee no episode leaks between train and test.
    """
    shuffled = list(files)
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_size))
    return shuffled[n_test:], shuffled[:n_test]


def build_incremental_pipeline(
    bucket: str,
    group_id: str,
    group: SchemaGroup,
    target: str | None = None,
    storage_options: Mapping[str, str] | None = None,
    source_url_fn: Callable[[str], str] = factorynet_source_url,
    extra_columns: Sequence[str] = (),
) -> IncrementalPipeline:
    """Builds an `@incremental` + `@schema`-validated pipeline that streams every
    file in `group` straight from its source URL into its own Delta table on
    `bucket` — no intermediate mirror or staging copy.

    Composed exactly per `data_warehousing_with_polars.schema`'s documented
    pattern (`@incremental` outermost, `@schema` innermost): the watermark
    guarantees each file is ingested exactly once, ever (`merge_on=None` is
    correct — FactoryNet's telemetry rows are immutable historical fact data,
    not a dimension needing upsert/SCD semantics); missing documented columns
    are quarantined into their own Delta table instead of crashing the run;
    undocumented extras are dropped; type mismatches are cast. `episode_id` (if
    the group has one) is namespaced by source filename — see
    `namespaced_episode_id_expr`. The output also always gets a `source_file`
    column (the bare filename) — cheap lineage, and what downstream consumers
    should filter on for a train/test split (`split_column_expr` matches
    `_source_file`'s basename the same way) rather than parsing `episode_id`.

    Args:
        bucket:          Target S3 bucket.
        group_id:        Key from `group_files_by_schema`'s result, e.g. `"schema_00"`
                         — used to name the target table when `target` isn't given.
        group:           The `SchemaGroup` to ingest.
        target:          Delta table path. Defaults to
                         `s3://{bucket}/delta/factorynet_{group_id}`.
        storage_options: Forwarded to every Delta read/write this pipeline makes
                         — e.g. `{"timeout": "600s", "connect_timeout": "60s"}` to
                         raise `object_store`'s S3 client timeouts on a
                         slow/degrading connection.
        source_url_fn:   Maps a filename to the URL/path `KnownFilesSource` reads
                         from. Defaults to `factorynet_source_url` (Hugging Face);
                         override for tests or to source from an existing mirror.
        extra_columns:   Columns to document and keep beyond the generic S-E-F-C
                         channels and `METADATA_COLUMNS` — e.g.
                         `("ctx_action", "ctx_setting")` for VORAUS-AD's discrete
                         operating-regime fields, which aren't generic enough to
                         include by default. Anything here not actually present
                         in `group.schema` is silently skipped, same as
                         `available_metadata_columns`. **Decide this up front**:
                         a column left out here is dropped (`on_extra="drop"`) at
                         ingestion and gone from the table for files already
                         watermarked — adding it later only affects new files.
    """
    documented_columns = (
        select_channel_columns(group.schema)
        + available_metadata_columns(group.schema)
        + [c for c in extra_columns if c in group.schema]
    )
    expect = {c: group.schema[c] for c in documented_columns}

    resolved_target = target or f"s3://{bucket}/delta/factorynet_{group_id}"
    quarantine = f"{resolved_target}_quarantine"
    paths = [source_url_fn(f) for f in group.files]

    @incremental(
        source=KnownFilesSource(paths),
        target=resolved_target,
        merge_on=None,
        storage_options=dict(storage_options) if storage_options is not None else None,
    )
    @validate_schema(
        expect=expect,
        on_missing="quarantine",
        on_extra="drop",
        evolution="cast",
        quarantine=quarantine,
    )
    def _ingest(lf: pl.LazyFrame) -> pl.LazyFrame:
        if "episode_id" in expect:
            lf = lf.with_columns(namespaced_episode_id_expr().alias("episode_id"))
        lf = lf.with_columns(source_file_basename().alias("source_file"))
        return lf.select([*expect, "source_file"])

    return _ingest


def stream_factorynet_to_bucket(
    bucket: str,
    groups: Mapping[str, SchemaGroup] | None = None,
    storage_options: Mapping[str, str] | None = None,
) -> dict[str, list[str]]:
    """Streams every FactoryNet file straight from Hugging Face into its own Delta
    table per schema group on `bucket` — one `@incremental` pipeline per group
    (`build_incremental_pipeline`), so a rerun only ever re-checks, never
    re-ingests, files already processed.

    Args:
        bucket:          Target S3 bucket.
        groups:          Defaults to `group_files_by_schema()` (reads every
                         FactoryNet file's schema first — ~120 lightweight
                         requests). Pass a pre-computed or filtered mapping to
                         skip that, or to target only specific groups.
        storage_options: Forwarded to `build_incremental_pipeline` for every group.

    Returns:
        `{group_id: [processed_file_uri, ...]}` — the files newly ingested this
        run, per group (`[]` for a group already fully caught up).
    """
    resolved_groups = groups if groups is not None else group_files_by_schema()
    processed: dict[str, list[str]] = {}
    for group_id, group in resolved_groups.items():
        pipeline = build_incremental_pipeline(
            bucket, group_id, group, storage_options=storage_options
        )
        processed[group_id] = pipeline.run()
    return processed
