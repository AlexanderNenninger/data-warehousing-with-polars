use crate::util::list_f64_dtype;
use polars::chunked_array::builder::AnonymousListBuilder;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::export::polars_core::utils::align_chunks_binary;
use serde::Deserialize;
use std::collections::HashMap;

// ── Output type functions ────────────────────────────────────────────────────

/// `List[T]` + `List[U]` → `List[Struct{first: T, second: U}]`
fn zip_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let inner_t = match input_fields[0].dtype() {
        DataType::List(inner) => *inner.clone(),
        dt => {
            return Err(PolarsError::ComputeError(
                format!("(list_ext.zip): expected List dtype, got {dt}").into(),
            ))
        }
    };
    let inner_u = match input_fields[1].dtype() {
        DataType::List(inner) => *inner.clone(),
        dt => {
            return Err(PolarsError::ComputeError(
                format!("(list_ext.zip): expected List dtype, got {dt}").into(),
            ))
        }
    };
    let struct_dtype = DataType::Struct(vec![
        Field::new(PlSmallStr::from_static("first"), inner_t),
        Field::new(PlSmallStr::from_static("second"), inner_u),
    ]);
    Ok(Field::new(
        input_fields[0].name.clone(),
        DataType::List(Box::new(struct_dtype)),
    ))
}

/// `List[Struct{f1:T1, ..., fn:Tn}]` → `Struct{f1:List[T1], ..., fn:List[Tn]}`
fn unzip_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let inner_struct_fields = match input_fields[0].dtype() {
        DataType::List(inner) => match inner.as_ref() {
            DataType::Struct(fields) => fields.clone(),
            dt => {
                return Err(PolarsError::ComputeError(
                    format!(
                        "(list_ext.unzip): expected List[Struct], inner type is {dt}"
                    )
                    .into(),
                ))
            }
        },
        dt => {
            return Err(PolarsError::ComputeError(
                format!("(list_ext.unzip): expected List dtype, got {dt}").into(),
            ))
        }
    };
    let out_fields: Vec<Field> = inner_struct_fields
        .iter()
        .map(|f| Field::new(f.name.clone(), DataType::List(Box::new(f.dtype().clone()))))
        .collect();
    Ok(Field::new(
        input_fields[0].name.clone(),
        DataType::Struct(out_fields),
    ))
}

// ── expr_list_zip ────────────────────────────────────────────────────────────

/// Zip two `List` columns element-wise into a `List[Struct{first, second}]` column.
///
/// Each row pairs elements at the same index from the two lists into a struct.
/// If the lists have different lengths the shorter one determines the output
/// length (excess elements from the longer list are dropped). Null rows in
/// either input produce a null output row.
///
/// ## Parameters
/// - `inputs[0]`: `List[T]` -- the left list column
/// - `inputs[1]`: `List[U]` -- the right list column
///
/// ## Return value
/// `List[Struct{first: T, second: U}]`
#[polars_expr(output_type_func = zip_output_type)]
fn expr_list_zip(inputs: &[Series]) -> PolarsResult<Series> {
    let lhs = inputs[0].list()?;
    let rhs = inputs[1].list()?;

    let inner_t = match lhs.dtype() {
        DataType::List(inner) => *inner.clone(),
        _ => unreachable!(),
    };
    let inner_u = match rhs.dtype() {
        DataType::List(inner) => *inner.clone(),
        _ => unreachable!(),
    };
    let struct_dtype = DataType::Struct(vec![
        Field::new(PlSmallStr::from_static("first"), inner_t),
        Field::new(PlSmallStr::from_static("second"), inner_u),
    ]);

    let n_rows = lhs.len();

    // Collect all rows first — AnonymousListBuilder<'a> holds references, so
    // the Series values must outlive the builder.
    let (lhs_aligned, rhs_aligned) = align_chunks_binary(lhs, rhs);
    let rows: PolarsResult<Vec<Option<Series>>> = lhs_aligned
        .amortized_iter()
        .zip(rhs_aligned.amortized_iter())
        .map(|(l_opt, r_opt)| match (l_opt, r_opt) {
            (Some(l), Some(r)) => {
                let l_s = l.as_ref();
                let r_s = r.as_ref();
                let len = l_s.len().min(r_s.len());
                // Rename to the output field names so StructChunked has unique fields.
                let l_sliced = l_s
                    .slice(0, len)
                    .with_name(PlSmallStr::from_static("first"));
                let r_sliced = r_s
                    .slice(0, len)
                    .with_name(PlSmallStr::from_static("second"));
                StructChunked::from_series(
                    PlSmallStr::EMPTY,
                    len,
                    [l_sliced, r_sliced].iter(),
                )
                .map(|sc| Some(sc.into_series()))
            }
            _ => Ok(None),
        })
        .collect();
    let rows = rows?;

    let mut builder =
        AnonymousListBuilder::new(PlSmallStr::EMPTY, n_rows, Some(struct_dtype.clone()));
    for row in &rows {
        match row {
            Some(s) => builder.append_series(s)?,
            None => builder.append_null(),
        }
    }

    builder
        .finish()
        .into_series()
        .cast(&DataType::List(Box::new(struct_dtype)))
}

// ── expr_list_unzip ──────────────────────────────────────────────────────────

/// Unzip a `List[Struct{f1:T1, ..., fn:Tn}]` column into a
/// `Struct{f1:List[T1], ..., fn:List[Tn]}` column.
///
/// Mirrors `Expr::struct_().unnest()` but operates on list elements rather
/// than top-level struct columns. Works for any number of struct fields.
/// Null rows produce a null output row.
///
/// ## Parameters
/// - `inputs[0]`: `List[Struct{f1:T1, ..., fn:Tn}]`
///
/// ## Return value
/// `Struct{f1:List[T1], ..., fn:List[Tn]}`
#[polars_expr(output_type_func = unzip_output_type)]
fn expr_list_unzip(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].list()?;

    let struct_fields: Vec<Field> = match ca.dtype() {
        DataType::List(inner) => match inner.as_ref() {
            DataType::Struct(fields) => fields.clone(),
            dt => {
                return Err(PolarsError::ComputeError(
                    format!(
                        "(list_ext.unzip): expected List[Struct], inner type is {dt}"
                    )
                    .into(),
                ))
            }
        },
        dt => {
            return Err(PolarsError::ComputeError(
                format!("(list_ext.unzip): expected List dtype, got {dt}").into(),
            ))
        }
    };

    let n_rows = ca.len();

    // Collect all rows first — AnonymousListBuilder<'a> holds references.
    // Each entry is either None (null row) or a Vec of per-field Series.
    let all_rows: Vec<Option<Vec<Series>>> = ca
        .into_iter()
        .map(|opt_row| match opt_row {
            None => None,
            Some(row) => row.struct_().ok().map(|sc| sc.fields_as_series()),
        })
        .collect();

    // One AnonymousListBuilder per struct field.
    let mut builders: Vec<AnonymousListBuilder> = struct_fields
        .iter()
        .map(|f| {
            AnonymousListBuilder::new(f.name.clone(), n_rows, Some(f.dtype().clone()))
        })
        .collect();

    for opt_fields in &all_rows {
        match opt_fields {
            None => {
                for b in builders.iter_mut() {
                    b.append_null();
                }
            }
            Some(field_series_vec) => {
                for (b, field_series) in builders.iter_mut().zip(field_series_vec.iter())
                {
                    b.append_series(field_series)?;
                }
            }
        }
    }

    let out_series: Vec<Series> = builders
        .iter_mut()
        .map(|b| b.finish().into_series())
        .collect();

    StructChunked::from_series(inputs[0].name().clone(), n_rows, out_series.iter())
        .map(|sc| sc.into_series())
}
// ── expr_list_join ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ListJoinKwargs {
    on: String,
    how: String,
    suffix: String,
}

fn list_struct_fields(field: &Field) -> PolarsResult<Vec<Field>> {
    match field.dtype() {
        DataType::List(inner) => match inner.as_ref() {
            DataType::Struct(fields) => Ok(fields.clone()),
            dt => Err(PolarsError::ComputeError(
                format!("(list_ext.join): expected List[Struct], got List[{dt}]").into(),
            )),
        },
        dt => Err(PolarsError::ComputeError(
            format!("(list_ext.join): expected List[Struct], got {dt}").into(),
        )),
    }
}

fn join_output_type(
    input_fields: &[Field],
    kwargs: ListJoinKwargs,
) -> PolarsResult<Field> {
    let left_fields = list_struct_fields(&input_fields[0])?;
    let right_fields = list_struct_fields(&input_fields[1])?;

    let left_names: std::collections::HashSet<&str> =
        left_fields.iter().map(|f| f.name().as_str()).collect();

    let mut out_fields = left_fields.clone();
    if kwargs.how != "anti" {
        for rf in &right_fields {
            if rf.name().as_str() == kwargs.on.as_str() {
                continue;
            }
            let out_name = if left_names.contains(rf.name().as_str()) {
                PlSmallStr::from(format!("{}{}", rf.name(), kwargs.suffix))
            } else {
                rf.name().clone()
            };
            out_fields.push(Field::new(out_name, rf.dtype().clone()));
        }
    }

    Ok(Field::new(
        input_fields[0].name.clone(),
        DataType::List(Box::new(DataType::Struct(out_fields))),
    ))
}

/// Join two `List[Struct]` columns row-wise on a common key field.
///
/// Performs a key-based join on the struct elements within each row. The key
/// field is looked up by name and compared via its string representation, so
/// any dtype that serialises unambiguously (integers, strings, booleans, etc.)
/// is supported.
///
/// ## Parameters
/// - `inputs[0]`: `List[Struct{..., key, ...}]` — the left list column
/// - `inputs[1]`: `List[Struct{..., key, ...}]` — the right list column
/// - `on`: name of the key field present in both structs
/// - `how`: join type — `"inner"` | `"left"` | `"anti"`
/// - `suffix`: suffix appended to right-side field names that collide with
///   left-side names (default `"_right"`)
///
/// ## Return value
/// - `inner` / `left`: `List[Struct{left fields..., right non-key fields...}]`
/// - `anti`: `List[Struct{left fields...}]`
#[polars_expr(output_type_func_with_kwargs = join_output_type)]
fn expr_list_join(
    inputs: &[Series],
    kwargs: ListJoinKwargs,
) -> PolarsResult<Series> {
    let lhs_ca = inputs[0].list()?;
    let rhs_ca = inputs[1].list()?;

    // Validate key presence in both inner struct schemas.
    let left_inner_fields: Vec<Field> =
        list_struct_fields(&Field::new(PlSmallStr::EMPTY, inputs[0].dtype().clone()))?;
    let right_inner_fields: Vec<Field> =
        list_struct_fields(&Field::new(PlSmallStr::EMPTY, inputs[1].dtype().clone()))?;

    let left_key_idx = left_inner_fields
        .iter()
        .position(|f| f.name().as_str() == kwargs.on.as_str())
        .ok_or_else(|| {
            PolarsError::ComputeError(
                format!(
                    "(list_ext.join): key '{}' not found in left struct",
                    kwargs.on
                )
                .into(),
            )
        })?;
    let right_key_idx = right_inner_fields
        .iter()
        .position(|f| f.name().as_str() == kwargs.on.as_str())
        .ok_or_else(|| {
            PolarsError::ComputeError(
                format!(
                    "(list_ext.join): key '{}' not found in right struct",
                    kwargs.on
                )
                .into(),
            )
        })?;

    // Map right non-key field indices → output names.
    let left_names: std::collections::HashSet<&str> = left_inner_fields
        .iter()
        .map(|f| f.name().as_str())
        .collect();
    let right_non_key: Vec<(usize, PlSmallStr)> = right_inner_fields
        .iter()
        .enumerate()
        .filter(|(_, f)| f.name().as_str() != kwargs.on.as_str())
        .map(|(i, f)| {
            let out_name = if left_names.contains(f.name().as_str()) {
                PlSmallStr::from(format!("{}{}", f.name(), kwargs.suffix))
            } else {
                f.name().clone()
            };
            (i, out_name)
        })
        .collect();

    // Output struct dtype for the list elements.
    let mut out_struct_fields = left_inner_fields.clone();
    if kwargs.how != "anti" {
        for (i, name) in &right_non_key {
            out_struct_fields.push(Field::new(
                name.clone(),
                right_inner_fields[*i].dtype().clone(),
            ));
        }
    }
    let out_inner_dtype = DataType::Struct(out_struct_fields);
    let out_dtype = DataType::List(Box::new(out_inner_dtype.clone()));

    // Collect row results before building (AnonymousListBuilder borrows).
    let (lhs_aligned, rhs_aligned) = align_chunks_binary(lhs_ca, rhs_ca);
    let rows: PolarsResult<Vec<Option<Series>>> = lhs_aligned
        .amortized_iter()
        .zip(rhs_aligned.amortized_iter())
        .map(|(l_opt, r_opt)| -> PolarsResult<Option<Series>> {
            match (l_opt, r_opt) {
                (Some(l), Some(r)) => Ok(Some(join_row(
                    l.as_ref(),
                    r.as_ref(),
                    left_key_idx,
                    right_key_idx,
                    &right_non_key,
                    &kwargs.how,
                )?)),
                _ => Ok(None),
            }
        })
        .collect();
    let rows = rows?;

    let n_rows = lhs_ca.len();
    let mut builder =
        AnonymousListBuilder::new(PlSmallStr::EMPTY, n_rows, Some(out_inner_dtype));
    for row in &rows {
        match row {
            Some(s) => builder.append_series(s)?,
            None => builder.append_null(),
        }
    }

    builder.finish().into_series().cast(&out_dtype)
}

fn join_row(
    left_row: &Series,
    right_row: &Series,
    left_key_idx: usize,
    right_key_idx: usize,
    right_non_key: &[(usize, PlSmallStr)],
    how: &str,
) -> PolarsResult<Series> {
    let left_sc = left_row.struct_()?;
    let right_sc = right_row.struct_()?;

    let left_fields = left_sc.fields_as_series();
    let right_fields = right_sc.fields_as_series();

    // Build right key → index map via string representation.
    let right_key_str = right_fields[right_key_idx].cast(&DataType::String)?;
    let right_key_ca = right_key_str.str()?;
    let mut right_map: HashMap<&str, u32> = HashMap::with_capacity(right_sc.len());
    for (i, opt_k) in right_key_ca.iter().enumerate() {
        if let Some(k) = opt_k {
            right_map.insert(k, i as u32);
        }
    }

    let left_key_str = left_fields[left_key_idx].cast(&DataType::String)?;
    let left_key_ca = left_key_str.str()?;

    // Collect per-output-row (left_idx, right_idx?) pairs.
    let mut left_idxs: Vec<u32> = Vec::with_capacity(left_sc.len());
    let mut right_idxs: Vec<Option<u32>> = Vec::with_capacity(left_sc.len());

    for (i, opt_k) in left_key_ca.iter().enumerate() {
        let matched = opt_k.and_then(|k| right_map.get(k).copied());
        match (how, matched) {
            ("inner", Some(j)) => {
                left_idxs.push(i as u32);
                right_idxs.push(Some(j));
            }
            ("left", Some(j)) => {
                left_idxs.push(i as u32);
                right_idxs.push(Some(j));
            }
            ("left", None) => {
                left_idxs.push(i as u32);
                right_idxs.push(None);
            }
            ("anti", None) => {
                left_idxs.push(i as u32);
                // right_idxs not used for anti
            }
            _ => {} // inner with no match, or anti with match: skip
        }
    }

    let n_out = left_idxs.len();
    let left_idx_ca = IdxCa::from_vec(PlSmallStr::EMPTY, left_idxs);

    // Gather all left fields.
    let mut out_series: Vec<Series> = left_fields
        .iter()
        .map(|s| s.take(&left_idx_ca))
        .collect::<PolarsResult<Vec<_>>>()?;

    // Gather right non-key fields (with nullable index for left join).
    if how != "anti" {
        let right_idx_ca: IdxCa = right_idxs.into_iter().collect();
        for (field_idx, out_name) in right_non_key {
            let gathered = right_fields[*field_idx].take(&right_idx_ca)?;
            out_series.push(gathered.with_name(out_name.clone()));
        }
    }

    StructChunked::from_series(PlSmallStr::EMPTY, n_out, out_series.iter())
        .map(|sc| sc.into_series())
}
// ── expr_list_enumerate ──────────────────────────────────────────────────────

/// Zip each list element with its 0-based index producing
/// `List[Struct{index: UInt32, value: T}]`.
///
/// ## Parameters
/// - `inputs[0]`: `List[T]`
///
/// ## Return value
/// `List[Struct{index: UInt32, value: T}]`
#[polars_expr(output_type_func = enumerate_output_type)]
fn expr_list_enumerate(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].list()?;
    let inner_t = match ca.dtype() {
        DataType::List(inner) => *inner.clone(),
        _ => unreachable!(),
    };
    let out_dtype = DataType::Struct(vec![
        Field::new(PlSmallStr::from_static("index"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("value"), inner_t),
    ]);
    let n_rows = ca.len();
    let rows: PolarsResult<Vec<Option<Series>>> = ca
        .into_iter()
        .map(|opt| match opt {
            None => Ok(None),
            Some(s) => {
                let len = s.len();
                let idx = Series::new(
                    PlSmallStr::from_static("index"),
                    (0u32..len as u32).collect::<Vec<_>>(),
                );
                let val = s.with_name(PlSmallStr::from_static("value"));
                StructChunked::from_series(PlSmallStr::EMPTY, len, [idx, val].iter())
                    .map(|sc| Some(sc.into_series()))
            }
        })
        .collect();
    let rows = rows?;
    let mut builder =
        AnonymousListBuilder::new(PlSmallStr::EMPTY, n_rows, Some(out_dtype.clone()));
    for row in &rows {
        match row {
            Some(s) => builder.append_series(s)?,
            None => builder.append_null(),
        }
    }
    builder
        .finish()
        .into_series()
        .cast(&DataType::List(Box::new(out_dtype)))
}

fn enumerate_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let inner_t = match input_fields[0].dtype() {
        DataType::List(inner) => *inner.clone(),
        dt => {
            return Err(PolarsError::ComputeError(
                format!("(list_ext.enumerate): expected List, got {dt}").into(),
            ))
        }
    };
    Ok(Field::new(
        input_fields[0].name.clone(),
        DataType::List(Box::new(DataType::Struct(vec![
            Field::new(PlSmallStr::from_static("index"), DataType::UInt32),
            Field::new(PlSmallStr::from_static("value"), inner_t),
        ]))),
    ))
}

// ── expr_list_dedup ──────────────────────────────────────────────────────────

/// Remove consecutive duplicate elements from each list (like Unix `uniq`).
///
/// Only adjacent duplicates are removed. Uses string-cast equality so works
/// for any dtype that serialises unambiguously. Null elements are treated as
/// equal to each other.
///
/// ## Parameters
/// - `inputs[0]`: `List[T]`
///
/// ## Return value
/// `List[T]` with consecutive duplicates removed.
#[polars_expr(output_type_func = same_inner_type)]
fn expr_list_dedup(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].list()?;
    let inner_t = match ca.dtype() {
        DataType::List(inner) => *inner.clone(),
        _ => unreachable!(),
    };
    let n_rows = ca.len();
    let rows: PolarsResult<Vec<Option<Series>>> = ca
        .into_iter()
        .map(|opt| match opt {
            None => Ok(None),
            Some(s) => {
                if s.is_empty() {
                    return Ok(Some(Series::new_empty(PlSmallStr::EMPTY, &inner_t)));
                }
                let str_rep = s.cast(&DataType::String)?;
                let str_ca = str_rep.str()?;
                let mut keep: Vec<u32> = vec![0];
                for i in 1..s.len() {
                    if str_ca.get(i) != str_ca.get(i - 1) {
                        keep.push(i as u32);
                    }
                }
                let idx = IdxCa::from_vec(PlSmallStr::EMPTY, keep);
                Ok(Some(s.take(&idx)?))
            }
        })
        .collect();
    let rows = rows?;
    let mut builder =
        AnonymousListBuilder::new(PlSmallStr::EMPTY, n_rows, Some(inner_t.clone()));
    for row in &rows {
        match row {
            Some(s) => builder.append_series(s)?,
            None => builder.append_null(),
        }
    }
    builder
        .finish()
        .into_series()
        .cast(&DataType::List(Box::new(inner_t)))
}

// ── expr_list_rotate ─────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct RotateKwargs {
    n: i64,
}

/// Rotate list elements by `n` positions (positive = right, negative = left).
///
/// ## Parameters
/// - `inputs[0]`: `List[T]`
/// - `n`: rotation amount; wraps modulo list length.
///
/// ## Return value
/// `List[T]`
#[polars_expr(output_type_func = same_inner_type)]
fn expr_list_rotate(
    inputs: &[Series],
    kwargs: RotateKwargs,
) -> PolarsResult<Series> {
    let ca = inputs[0].list()?;
    let inner_t = match ca.dtype() {
        DataType::List(inner) => *inner.clone(),
        _ => unreachable!(),
    };
    let n_rows = ca.len();
    let rows: PolarsResult<Vec<Option<Series>>> = ca
        .into_iter()
        .map(|opt| match opt {
            None => Ok(None),
            Some(s) => {
                let len = s.len();
                if len == 0 {
                    return Ok(Some(s));
                }
                let shift = kwargs.n.rem_euclid(len as i64) as usize;
                if shift == 0 {
                    return Ok(Some(s));
                }
                // right-rotate by `shift`: last `shift` elements come first.
                // Build rotated index order and gather in one pass.
                let split = len - shift;
                let idx: Vec<u32> =
                    (split..len).chain(0..split).map(|i| i as u32).collect();
                let idx_ca = IdxCa::from_vec(PlSmallStr::EMPTY, idx);
                Ok(Some(s.take(&idx_ca)?))
            }
        })
        .collect();
    let rows = rows?;
    let mut builder =
        AnonymousListBuilder::new(PlSmallStr::EMPTY, n_rows, Some(inner_t.clone()));
    for row in &rows {
        match row {
            Some(s) => builder.append_series(s)?,
            None => builder.append_null(),
        }
    }
    builder
        .finish()
        .into_series()
        .cast(&DataType::List(Box::new(inner_t)))
}

// ── expr_list_windows ────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct WindowsKwargs {
    size: usize,
    step: usize,
}

fn windows_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let inner_t = match input_fields[0].dtype() {
        DataType::List(inner) => *inner.clone(),
        dt => {
            return Err(PolarsError::ComputeError(
                format!("(list_ext.windows): expected List, got {dt}").into(),
            ))
        }
    };
    Ok(Field::new(
        input_fields[0].name.clone(),
        DataType::List(Box::new(DataType::List(Box::new(inner_t)))),
    ))
}

/// Produce a sliding window view of each list as `List[List[T]]`.
///
/// ## Parameters
/// - `inputs[0]`: `List[T]`
/// - `size`: window size (must be > 0)
/// - `step`: step between window starts (default 1, must be > 0)
///
/// ## Return value
/// `List[List[T]]` — each inner list has exactly `size` elements. Lists
/// shorter than `size` produce an empty outer list.
#[polars_expr(output_type_func = windows_output_type)]
fn expr_list_windows(
    inputs: &[Series],
    kwargs: WindowsKwargs,
) -> PolarsResult<Series> {
    if kwargs.size == 0 || kwargs.step == 0 {
        return Err(PolarsError::ComputeError(
            "(list_ext.windows): size and step must be > 0".into(),
        ));
    }
    let ca = inputs[0].list()?;
    let inner_t = match ca.dtype() {
        DataType::List(inner) => *inner.clone(),
        _ => unreachable!(),
    };
    let inner_list_t = DataType::List(Box::new(inner_t.clone()));
    let n_rows = ca.len();

    // Each outer row is itself a List[List[T]].
    let rows: PolarsResult<Vec<Option<Series>>> = ca
        .into_iter()
        .map(|opt| match opt {
            None => Ok(None),
            Some(s) => {
                let len = s.len();
                if len < kwargs.size {
                    // Return empty List[List[T]].
                    let empty_inner = Series::new_empty(PlSmallStr::EMPTY, &inner_t);
                    let mut b = AnonymousListBuilder::new(
                        PlSmallStr::EMPTY,
                        0,
                        Some(inner_t.clone()),
                    );
                    let _ = b.append_series(&empty_inner); // unused; just for type
                    drop(b);
                    let mut outer = AnonymousListBuilder::new(
                        PlSmallStr::EMPTY,
                        0,
                        Some(inner_t.clone()),
                    );
                    return Ok(Some(outer.finish().into_series().cast(&inner_list_t)?));
                }
                let n_windows = (len - kwargs.size) / kwargs.step + 1;
                let windows_vec: Vec<Series> = (0..len)
                    .step_by(kwargs.step)
                    .take_while(|&start| start + kwargs.size <= len)
                    .map(|start| s.slice(start as i64, kwargs.size))
                    .collect();
                let mut outer = AnonymousListBuilder::new(
                    PlSmallStr::EMPTY,
                    n_windows,
                    Some(inner_t.clone()),
                );
                for w in &windows_vec {
                    outer.append_series(w)?;
                }
                Ok(Some(outer.finish().into_series().cast(&inner_list_t)?))
            }
        })
        .collect();
    let rows = rows?;
    let mut builder =
        AnonymousListBuilder::new(PlSmallStr::EMPTY, n_rows, Some(inner_list_t.clone()));
    for row in &rows {
        match row {
            Some(s) => builder.append_series(s)?,
            None => builder.append_null(),
        }
    }
    builder
        .finish()
        .into_series()
        .cast(&DataType::List(Box::new(inner_list_t)))
}

// ── expr_list_chunks ─────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ChunksKwargs {
    size: usize,
}

/// Partition each list into non-overlapping chunks of `size` as `List[List[T]]`.
///
/// The last chunk may be smaller than `size` if the list length is not a
/// multiple of `size`.
///
/// ## Parameters
/// - `inputs[0]`: `List[T]`
/// - `size`: chunk size (must be > 0)
///
/// ## Return value
/// `List[List[T]]`
#[polars_expr(output_type_func = windows_output_type)]
fn expr_list_chunks(
    inputs: &[Series],
    kwargs: ChunksKwargs,
) -> PolarsResult<Series> {
    if kwargs.size == 0 {
        return Err(PolarsError::ComputeError(
            "(list_ext.chunks): size must be > 0".into(),
        ));
    }
    let ca = inputs[0].list()?;
    let inner_t = match ca.dtype() {
        DataType::List(inner) => *inner.clone(),
        _ => unreachable!(),
    };
    let inner_list_t = DataType::List(Box::new(inner_t.clone()));
    let n_rows = ca.len();

    let rows: PolarsResult<Vec<Option<Series>>> = ca
        .into_iter()
        .map(|opt| match opt {
            None => Ok(None),
            Some(s) => {
                let len = s.len();
                let n_chunks = len.div_ceil(kwargs.size).max(1);
                let mut outer = AnonymousListBuilder::new(
                    PlSmallStr::EMPTY,
                    n_chunks,
                    Some(inner_t.clone()),
                );
                if len == 0 {
                    return Ok(Some(outer.finish().into_series().cast(&inner_list_t)?));
                }
                let chunks_vec: Vec<Series> = (0..len)
                    .step_by(kwargs.size)
                    .map(|start| {
                        let end = (start + kwargs.size).min(len);
                        s.slice(start as i64, end - start)
                    })
                    .collect();
                for chunk in &chunks_vec {
                    outer.append_series(chunk)?;
                }
                Ok(Some(outer.finish().into_series().cast(&inner_list_t)?))
            }
        })
        .collect();
    let rows = rows?;
    let mut builder =
        AnonymousListBuilder::new(PlSmallStr::EMPTY, n_rows, Some(inner_list_t.clone()));
    for row in &rows {
        match row {
            Some(s) => builder.append_series(s)?,
            None => builder.append_null(),
        }
    }
    builder
        .finish()
        .into_series()
        .cast(&DataType::List(Box::new(inner_list_t)))
}

// ── expr_list_position ───────────────────────────────────────────────────────

#[derive(Deserialize)]
struct PositionKwargs {
    op: String,
    value: f64,
}

/// Return the 0-based index of the first element matching the given condition,
/// or `null` if no element matches.
///
/// ## Parameters
/// - `inputs[0]`: `List[T]` (cast to `List[Float64]` internally)
/// - `op`: comparison operator — `"eq"` | `"ne"` | `"gt"` | `"ge"` | `"lt"` | `"le"`
/// - `value`: the comparison value
///
/// ## Return value
/// `UInt32` — index of first match, or `null`.
#[polars_expr(output_type = UInt32)]
fn expr_list_position(
    inputs: &[Series],
    kwargs: PositionKwargs,
) -> PolarsResult<Series> {
    let ca = inputs[0]
        .cast(&DataType::List(Box::new(DataType::Float64)))?
        .list()?
        .clone();
    let valid_ops = ["eq", "ne", "gt", "ge", "lt", "le"];
    if !valid_ops.contains(&kwargs.op.as_str()) {
        return Err(PolarsError::ComputeError(
            format!(
                "(list_ext.position): op must be one of [{}], got {:?}",
                valid_ops.join(", "),
                kwargs.op
            )
            .into(),
        ));
    }
    let out: UInt32Chunked = ca
        .into_iter()
        .map(|opt| match opt {
            None => None,
            Some(s) => {
                let f = s.f64().ok()?;
                f.iter().enumerate().find_map(|(i, v)| {
                    let matches = match (kwargs.op.as_str(), v) {
                        (_, None) => false,
                        ("eq", Some(x)) => x == kwargs.value,
                        ("ne", Some(x)) => x != kwargs.value,
                        ("gt", Some(x)) => x > kwargs.value,
                        ("ge", Some(x)) => x >= kwargs.value,
                        ("lt", Some(x)) => x < kwargs.value,
                        ("le", Some(x)) => x <= kwargs.value,
                        _ => false,
                    };
                    if matches {
                        Some(i as u32)
                    } else {
                        None
                    }
                })
            }
        })
        .collect();
    Ok(out.into_series())
}

// ── expr_list_flat_map ───────────────────────────────────────────────────────

#[derive(Deserialize)]
struct FlatMapKwargs {
    op: String,
    value: f64,
}

/// Apply an element-wise map then flatten one level: equivalent to
/// `list.eval(...).explode()` but in a single pass.
///
/// The supported operations are arithmetic transforms on `List[Float64]`.
///
/// ## Parameters
/// - `inputs[0]`: `List[List[T]]` (outer list to flatten after mapping)
///   *or* `List[T]` when used as a map+flatten on a single list
/// - `op`: `"add"` | `"sub"` | `"mul"` | `"div"`
/// - `value`: scalar to apply
///
/// ## Return value
/// `List[Float64]` — mapped and flattened.
#[polars_expr(output_type_func = list_f64_dtype)]
fn expr_list_flat_map(
    inputs: &[Series],
    kwargs: FlatMapKwargs,
) -> PolarsResult<Series> {
    let valid_ops = ["add", "sub", "mul", "div"];
    if !valid_ops.contains(&kwargs.op.as_str()) {
        return Err(PolarsError::ComputeError(
            format!(
                "(list_ext.flat_map): op must be one of [{}], got {:?}",
                valid_ops.join(", "),
                kwargs.op
            )
            .into(),
        ));
    }
    // Cast outer list elements to Float64.
    let ca = inputs[0]
        .cast(&DataType::List(Box::new(DataType::Float64)))?
        .list()?
        .clone();
    let n_rows = ca.len();
    // Collect mapped+flattened values per row.
    let rows: Vec<Option<Series>> = ca
        .into_iter()
        .map(|opt| match opt {
            None => None,
            Some(s) => {
                let f = s.f64().ok()?;
                let mapped: Vec<Option<f64>> = f
                    .iter()
                    .map(|v| {
                        v.map(|x| match kwargs.op.as_str() {
                            "add" => x + kwargs.value,
                            "sub" => x - kwargs.value,
                            "mul" => x * kwargs.value,
                            "div" => x / kwargs.value,
                            _ => unreachable!(),
                        })
                    })
                    .collect();
                Some(Series::new(PlSmallStr::EMPTY, mapped))
            }
        })
        .collect();
    let mut builder =
        AnonymousListBuilder::new(PlSmallStr::EMPTY, n_rows, Some(DataType::Float64));
    for row in &rows {
        match row {
            Some(s) => builder.append_series(s)?,
            None => builder.append_null(),
        }
    }
    builder
        .finish()
        .into_series()
        .cast(&DataType::List(Box::new(DataType::Float64)))
}

// ── shared output type helper ─────────────────────────────────────────────────

fn same_inner_type(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        input_fields[0].name.clone(),
        input_fields[0].dtype().clone(),
    ))
}
