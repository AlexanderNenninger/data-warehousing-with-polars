from pathlib import Path
from typing import Literal, Optional, Union

import polars as pl
from polars.plugins import register_plugin_function

from polars_list_ext._internal import __version__ as __version__  # ty: ignore[unresolved-import]
from polars_list_ext._internal import (  # noqa: F401  # ty: ignore[unresolved-import]
    fft_freqs,
    fft_freqs_linspace,
)

root_path = Path(__file__).parent


def apply_fft(
    list_column: Union[pl.Expr, str, pl.Series],
    sample_rate: int,
    window: Optional[str] = None,
    bp_min: Optional[float] = None,
    bp_max: Optional[float] = None,
    bp_ord: Optional[int] = None,
    norm: Optional[str] = None,
    skip_fft: bool = False,
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column],
        kwargs={
            "sample_rate": sample_rate,
            "window": window,
            "bp_min": bp_min,
            "bp_max": bp_max,
            "bp_ord": bp_ord,
            "norm": norm,
            "skip_fft": skip_fft,
        },
        plugin_path=root_path,
        function_name="expr_fft",
        is_elementwise=True,
    )


def operate_scalar_on_list(
    list_column: Union[pl.Expr, str, pl.Series],
    scalar_column: Union[pl.Expr, str, pl.Series],
    operation: Literal["add", "sub", "mul", "div"],
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column, scalar_column],
        kwargs={
            "operation": operation,
        },
        plugin_path=root_path,
        function_name="expr_operate_scalar_on_list",
        is_elementwise=True,
    )


def interpolate_columns(
    x_data: Union[pl.Expr, str, pl.Series],
    y_data: Union[pl.Expr, str, pl.Series],
    x_interp: Union[pl.Expr, str, pl.Series],
) -> pl.Expr:
    return register_plugin_function(
        args=[x_data, y_data, x_interp],
        plugin_path=root_path,
        function_name="expr_interpolate_columns",
        is_elementwise=True,
    )


def aggregate_list_col_elementwise(
    list_column: Union[pl.Expr, str, pl.Series],
    list_size: int,
    aggregation: Literal["mean", "sum", "count"] = "mean",
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column],
        kwargs={
            "list_size": list_size,
            "aggregation": aggregation,
        },
        plugin_path=root_path,
        function_name="expr_aggregate_list_col_elementwise",
        is_elementwise=False,
        returns_scalar=True,
    )


def agg_of_range(
    list_column_y: Union[pl.Expr, str, pl.Series],
    list_column_x: Union[pl.Expr, str, pl.Series],
    aggregation: Literal["mean", "median", "sum", "count", "max", "min"],
    x_min: float,
    x_max: float,
    x_range_excluded: Optional[tuple[float, float]] = None,
    x_min_idx_offset: Optional[int] = None,
    x_max_idx_offset: Optional[int] = None,
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column_y, list_column_x],
        kwargs={
            "aggregation": aggregation,
            "x_min": x_min,
            "x_max": x_max,
            "x_range_excluded": x_range_excluded,
            "x_min_idx_offset": x_min_idx_offset,
            "x_max_idx_offset": x_max_idx_offset,
        },
        plugin_path=root_path,
        function_name="expr_agg_of_range",
        is_elementwise=True,
    )


def mean_of_range(
    list_column_y: Union[pl.Expr, str, pl.Series],
    list_column_x: Union[pl.Expr, str, pl.Series],
    x_min: float,
    x_max: float,
    x_range_excluded: Optional[tuple[float, float]] = None,
    x_min_idx_offset: Optional[int] = None,
    x_max_idx_offset: Optional[int] = None,
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column_y, list_column_x],
        kwargs={
            "aggregation": "mean",
            "x_min": x_min,
            "x_max": x_max,
            "x_range_excluded": x_range_excluded,
            "x_min_idx_offset": x_min_idx_offset,
            "x_max_idx_offset": x_max_idx_offset,
        },
        plugin_path=root_path,
        function_name="expr_agg_of_range",
        is_elementwise=True,
    )


@pl.api.register_expr_namespace("list_ext")
class ListExtNamespace:
    """Extension namespace for List column combinators.

    Accessed via `pl.col("col").list_ext.<method>()`.
    """

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def zip(self, other: Union[pl.Expr, str]) -> pl.Expr:
        """Zip two List columns element-wise into a List[Struct{first, second}] column.

        Pairs elements at the same index from the two lists. If the lists have
        different lengths the shorter one determines the output length. Null rows
        in either input produce a null output row.

        Args:
            other: The right-hand List column to zip with.

        Returns:
            A ``List[Struct{first: T, second: U}]`` expression.

        Example:
            ```python
            import polars as pl
            import polars_list_ext  # noqa: F401 — registers the namespace
            df = pl.DataFrame({"a": [[1, 2, 3]], "b": [[4, 5, 6]]})
            df.with_columns(pl.col("a").list_ext.zip(pl.col("b")).alias("zipped"))
            # shape: (1, 3)
            ```
        """
        other_expr = pl.col(other) if isinstance(other, str) else other
        return register_plugin_function(
            args=[self._expr, other_expr],
            plugin_path=root_path,
            function_name="expr_list_zip",
            is_elementwise=True,
        )

    def unzip(self) -> pl.Expr:
        """Unzip a List[Struct] column into a Struct of List columns.

        Mirrors ``Expr.struct.unnest()`` but operates on list elements rather
        than top-level struct columns. Works for any number of struct fields.
        Null rows produce a null output row.

        Returns:
            A ``Struct{f1: List[T1], ..., fn: List[Tn]}`` expression whose field
            names and types mirror those of the inner struct.

        Example:
            ```python
            import polars as pl
            import polars_list_ext  # noqa: F401 — registers the namespace
            df = pl.DataFrame({"pairs": [
                [{"first": 1, "second": 4}, {"first": 2, "second": 5}]
            ]})
            df.with_columns(pl.col("pairs").list_ext.unzip().alias("unzipped"))
            # shape: (1, 2)
            ```
        """
        return register_plugin_function(
            args=[self._expr],
            plugin_path=root_path,
            function_name="expr_list_unzip",
            is_elementwise=True,
        )

    def join(
        self,
        other: Union[pl.Expr, str],
        on: str,
        how: str = "inner",
        suffix: str = "_right",
    ) -> pl.Expr:
        """Join two List[Struct] columns row-wise on a common key field.

        Performs a key-based join on the struct elements within each row.
        The key field is matched by string representation, supporting any
        dtype that serialises unambiguously (integers, strings, booleans, …).

        Args:
            other: The right-hand ``List[Struct]`` column to join with.
            on: Name of the key field present in both inner structs.
            how: Join type — ``"inner"`` (default), ``"left"``, or ``"anti"``.
            suffix: Suffix appended to right-side field names that collide
                with left-side names. Defaults to ``"_right"``.

        Returns:
            - ``inner`` / ``left``: ``List[Struct{left fields, right non-key fields}]``
            - ``anti``: ``List[Struct{left fields}]``

        Example:
            ```python
            import polars as pl
            import polars_list_ext  # noqa: F401 — registers the namespace
            orders = pl.DataFrame({"o": [[{"id": 1, "qty": 10}, {"id": 2, "qty": 5}]]})
            products = pl.DataFrame({"p": [[{"id": 1, "name": "A"}, {"id": 3, "name": "C"}]]})
            orders.with_columns(
                pl.col("o").list_ext.join(pl.col("p"), on="id").alias("joined")
            )
            ```
        """
        other_expr = pl.col(other) if isinstance(other, str) else other
        return register_plugin_function(
            args=[self._expr, other_expr],
            kwargs={"on": on, "how": how, "suffix": suffix},
            plugin_path=root_path,
            function_name="expr_list_join",
            is_elementwise=True,
        )

    def enumerate(self) -> pl.Expr:
        """Zip each element with its 0-based position index.

        Returns:
            ``List[Struct{index: UInt32, value: T}]``

        Example:
            ```python
            pl.col("xs").list_ext.enumerate()
            ```
        """
        return register_plugin_function(
            args=[self._expr],
            plugin_path=root_path,
            function_name="expr_list_enumerate",
            is_elementwise=True,
        )

    def dedup(self) -> pl.Expr:
        """Remove consecutive duplicate elements (like Unix ``uniq``).

        Only adjacent duplicates are removed — use ``list.unique()`` for
        set-based deduplication. Null elements are treated as equal to each
        other.

        Returns:
            ``List[T]`` with consecutive duplicates removed.

        Example:
            ```python
            pl.col("xs").list_ext.dedup()
            ```
        """
        return register_plugin_function(
            args=[self._expr],
            plugin_path=root_path,
            function_name="expr_list_dedup",
            is_elementwise=True,
        )

    def rotate(self, n: int) -> pl.Expr:
        """Rotate list elements by ``n`` positions.

        Positive ``n`` rotates right (towards higher indices); negative
        rotates left. Wraps modulo the list length.

        Args:
            n: Number of positions to rotate.

        Returns:
            ``List[T]``

        Example:
            ```python
            pl.col("xs").list_ext.rotate(2)
            ```
        """
        return register_plugin_function(
            args=[self._expr],
            kwargs={"n": n},
            plugin_path=root_path,
            function_name="expr_list_rotate",
            is_elementwise=True,
        )

    def windows(self, size: int, step: int = 1) -> pl.Expr:
        """Produce a sliding window view as ``List[List[T]]``.

        Each inner list has exactly ``size`` elements. Lists shorter than
        ``size`` produce an empty outer list.

        Args:
            size: Window size (must be > 0).
            step: Step between window starts (default 1, must be > 0).

        Returns:
            ``List[List[T]]``

        Example:
            ```python
            pl.col("signal").list_ext.windows(4, step=2)
            ```
        """
        return register_plugin_function(
            args=[self._expr],
            kwargs={"size": size, "step": step},
            plugin_path=root_path,
            function_name="expr_list_windows",
            is_elementwise=True,
        )

    def chunks(self, size: int) -> pl.Expr:
        """Partition each list into non-overlapping chunks of ``size``.

        The last chunk may be smaller than ``size`` if the list length is not
        a multiple of ``size``.

        Args:
            size: Chunk size (must be > 0).

        Returns:
            ``List[List[T]]``

        Example:
            ```python
            pl.col("signal").list_ext.chunks(8)
            ```
        """
        return register_plugin_function(
            args=[self._expr],
            kwargs={"size": size},
            plugin_path=root_path,
            function_name="expr_list_chunks",
            is_elementwise=True,
        )

    def position(self, op: str, value: float) -> pl.Expr:
        """Return the index of the first element satisfying a condition.

        The list is cast to ``Float64`` for comparison. Returns ``null`` if
        no element matches.

        Args:
            op: Comparison operator — ``"eq"`` | ``"ne"`` | ``"gt"`` |
                ``"ge"`` | ``"lt"`` | ``"le"``.
            value: Comparison value.

        Returns:
            ``UInt32`` — index of first match, or ``null``.

        Example:
            ```python
            pl.col("signal").list_ext.position("gt", 0.5)
            ```
        """
        return register_plugin_function(
            args=[self._expr],
            kwargs={"op": op, "value": float(value)},
            plugin_path=root_path,
            function_name="expr_list_position",
            is_elementwise=True,
        )

    def flat_map(self, op: str, value: float) -> pl.Expr:
        """Apply a scalar arithmetic operation then return the result as a flat list.

        Equivalent to ``list.eval(pl.element() <op> value)`` but in a single
        Rust pass. The list is cast to ``Float64``.

        Args:
            op: Arithmetic operation — ``"add"`` | ``"sub"`` | ``"mul"`` |
                ``"div"``.
            value: Scalar operand.

        Returns:
            ``List[Float64]``

        Example:
            ```python
            pl.col("signal").list_ext.flat_map("mul", 2.0)
            ```
        """
        return register_plugin_function(
            args=[self._expr],
            kwargs={"op": op, "value": float(value)},
            plugin_path=root_path,
            function_name="expr_list_flat_map",
            is_elementwise=True,
        )
