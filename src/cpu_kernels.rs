use pyo3::prelude::*;
use std::sync::Arc;

use crate::tensor::TensorView;

/// Copies from a potentially strided view into a contiguous output buffer.
///
/// `out` must have length `view.numel`.
pub fn cpu_copy_strided_to_contiguous(out: &mut [f32], buf: &Arc<Vec<f32>>, view: &TensorView) -> PyResult<()> {
    if out.len() != view.numel {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "output length does not match view.numel",
        ));
    }
    let mut out_i = 0usize;
    for_each_offset(view, |src_off| {
        out[out_i] = buf[src_off];
        out_i += 1;
    })?;
    Ok(())
}

/// Iterate all element offsets (in elements) for a view, in row-major order.
pub fn for_each_offset<F: FnMut(usize)>(view: &TensorView, mut f: F) -> PyResult<()> {
    fn rec<F: FnMut(usize)>(
        view: &TensorView,
        dim: usize,
        base_off: isize,
        f: &mut F,
    ) -> PyResult<()> {
        if dim == view.shape.len() {
            if base_off < 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "negative offset in view iteration (unsupported)",
                ));
            }
            f(base_off as usize);
            return Ok(());
        }
        let len = view.shape[dim];
        let stride = view.strides[dim];
        for i in 0..len {
            rec(view, dim + 1, base_off + (i as isize) * stride, f)?;
        }
        Ok(())
    }

    rec(view, 0, view.offset as isize, &mut f)
}

/// Elementwise binary op on two same-shaped tensors (no broadcasting), producing a contiguous output.
pub fn cpu_ew_binary(
    out: &mut [f32],
    a_buf: &Arc<Vec<f32>>,
    a_view: &TensorView,
    b_buf: &Arc<Vec<f32>>,
    b_view: &TensorView,
    op: impl Fn(f32, f32) -> f32,
) -> PyResult<()> {
    if a_view.shape != b_view.shape {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "shape mismatch for elementwise op (no broadcasting supported)",
        ));
    }
    if out.len() != a_view.numel || out.len() != b_view.numel {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "output length does not match numel",
        ));
    }

    let mut out_i = 0usize;
    for_each_offsets_pair(a_view, b_view, |a_off, b_off| {
        out[out_i] = op(a_buf[a_off], b_buf[b_off]);
        out_i += 1;
    })?;
    Ok(())
}

/// Elementwise unary op on a tensor, producing a contiguous output.
pub fn cpu_ew_unary(
    out: &mut [f32],
    x_buf: &Arc<Vec<f32>>,
    x_view: &TensorView,
    op: impl Fn(f32) -> f32,
) -> PyResult<()> {
    if out.len() != x_view.numel {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "output length does not match numel",
        ));
    }
    let mut out_i = 0usize;
    for_each_offset(x_view, |x_off| {
        out[out_i] = op(x_buf[x_off]);
        out_i += 1;
    })?;
    Ok(())
}

/// Sum all elements of a (possibly strided) tensor.
pub fn cpu_sum_all(buf: &Arc<Vec<f32>>, view: &TensorView) -> PyResult<f32> {
    let mut acc: f32 = 0.0;
    for_each_offset(view, |off| {
        acc += buf[off];
    })?;
    Ok(acc)
}

fn for_each_offsets_pair<F: FnMut(usize, usize)>(
    a_view: &TensorView,
    b_view: &TensorView,
    mut f: F,
) -> PyResult<()> {
    fn rec<F: FnMut(usize, usize)>(
        a_view: &TensorView,
        b_view: &TensorView,
        dim: usize,
        a_off: isize,
        b_off: isize,
        f: &mut F,
    ) -> PyResult<()> {
        if dim == a_view.shape.len() {
            if a_off < 0 || b_off < 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "negative offset in view iteration (unsupported)",
                ));
            }
            f(a_off as usize, b_off as usize);
            return Ok(());
        }
        let len = a_view.shape[dim];
        let a_stride = a_view.strides[dim];
        let b_stride = b_view.strides[dim];
        for i in 0..len {
            rec(
                a_view,
                b_view,
                dim + 1,
                a_off + (i as isize) * a_stride,
                b_off + (i as isize) * b_stride,
                f,
            )?;
        }
        Ok(())
    }
    rec(
        a_view,
        b_view,
        0,
        a_view.offset as isize,
        b_view.offset as isize,
        &mut f,
    )
}


