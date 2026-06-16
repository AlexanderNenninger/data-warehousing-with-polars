import polars as pl
from data_warehousing_with_polars import from_frame, incremental


def test_pipeline_version_written(tmp_path):
    target = str(tmp_path / "tgt1")
    src = from_frame(lambda: pl.DataFrame({"id": [1]}).lazy())

    @incremental(source=src, target=target, merge_on="id")
    def pipe(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    labels = pipe.run()
    assert isinstance(labels, list)

    stored = pipe._load_pipeline_version()
    assert stored == pipe.pipeline_hash


def test_pipeline_version_changes_on_code_update(tmp_path):
    target = str(tmp_path / "tgt2")
    src = from_frame(lambda: pl.DataFrame({"id": [1]}).lazy())

    @incremental(source=src, target=target, merge_on="id")
    def p1(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.select(pl.col("id"))

    p1.run()
    old = p1._load_pipeline_version()

    @incremental(source=src, target=target, merge_on="id")
    def p2(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.select((pl.col("id") * 2).alias("id"))

    p2.run()
    new = p2._load_pipeline_version()

    assert old != new
    assert new == p2.pipeline_hash


def test_fail_on_version_mismatch_raises(tmp_path):
    target = str(tmp_path / "tgt3")
    src = from_frame(lambda: pl.DataFrame({"id": [1]}).lazy())

    @incremental(source=src, target=target, merge_on="id")
    def base(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf

    base.run()

    @incremental(source=src, target=target, merge_on="id", fail_on_version_mismatch=True)
    def changed(lf: pl.LazyFrame) -> pl.LazyFrame:
        return lf.select((pl.col("id") * 3).alias("id"))

    import pytest

    with pytest.raises(RuntimeError):
        changed.run()
