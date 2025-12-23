use pyo3::prelude::*;
use pyo3::types::{PyAny, PyList};
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::Arc;

use crate::device::{Device, DeviceType};

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

#[cfg(feature = "cuda")]
fn map_cuda_err(e: cudarc::driver::DriverError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA error: {e:?}"))
}

/// A tensor view into a flat storage buffer.
///
/// - `shape`: lengths per dimension
/// - `strides`: stride (in elements) per dimension
/// - `offset`: starting offset (in elements) into the underlying storage
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorView {
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset: usize,
    pub numel: usize,
}

impl TensorView {
    pub fn new_contiguous(shape: Vec<usize>) -> PyResult<Self> {
        let numel = numel_from_shape(&shape)?;
        let strides = contiguous_strides(&shape);
        Ok(TensorView {
            shape,
            strides,
            offset: 0,
            numel,
        })
    }

    pub fn is_contiguous(&self) -> bool {
        if self.offset != 0 {
            return false;
        }
        self.strides == contiguous_strides(&self.shape)
    }

    pub fn reshape_view(&self, new_shape: Vec<usize>) -> PyResult<Self> {
        let new_numel = numel_from_shape(&new_shape)?;
        if new_numel != self.numel {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "cannot reshape tensor of numel={} into shape with numel={}",
                self.numel, new_numel
            )));
        }
        if !self.is_contiguous() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "reshape requires a contiguous tensor (call .contiguous() first)",
            ));
        }
        TensorView::new_contiguous(new_shape)
    }

    pub fn transpose_view(&self, dim0: usize, dim1: usize) -> PyResult<Self> {
        let nd = self.shape.len();
        if dim0 >= nd || dim1 >= nd {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "transpose dim out of range",
            ));
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(dim0, dim1);
        strides.swap(dim0, dim1);
        Ok(TensorView {
            shape,
            strides,
            offset: self.offset,
            numel: self.numel,
        })
    }

    pub fn slice_view(&self, dim: usize, start: isize, end: isize, step: isize) -> PyResult<Self> {
        let nd = self.shape.len();
        if dim >= nd {
            return Err(pyo3::exceptions::PyIndexError::new_err("slice dim out of range"));
        }
        if step <= 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "slice step must be > 0",
            ));
        }

        let dim_len = self.shape[dim] as isize;
        let start = clamp_index(start, dim_len);
        let end = clamp_index(end, dim_len);
        if end < start {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "slice end must be >= start",
            ));
        }

        let len = ((end - start) + (step - 1)) / step; // ceil div
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        let base_stride = strides[dim];
        strides[dim] = base_stride * step;
        shape[dim] = len as usize;
        let offset = (self.offset as isize + start * base_stride) as usize;
        let numel = numel_from_shape(&shape)?;
        Ok(TensorView {
            shape,
            strides,
            offset,
            numel,
        })
    }
}

fn clamp_index(idx: isize, len: isize) -> isize {
    // minimal “python-like” support: allow negative indices, clamp to [0, len]
    let mut i = idx;
    if i < 0 {
        i += len;
    }
    if i < 0 {
        i = 0;
    }
    if i > len {
        i = len;
    }
    i
}

fn numel_from_shape(shape: &[usize]) -> PyResult<usize> {
    if shape.is_empty() {
        return Ok(1);
    }
    let mut n: usize = 1;
    for &d in shape {
        if d == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "shape dimensions must be > 0",
            ));
        }
        n = n
            .checked_mul(d)
            .ok_or_else(|| pyo3::exceptions::PyOverflowError::new_err("numel overflow"))?;
    }
    Ok(n)
}

fn contiguous_strides(shape: &[usize]) -> Vec<isize> {
    if shape.is_empty() {
        return vec![];
    }
    let mut strides = vec![0isize; shape.len()];
    let mut acc: isize = 1;
    for (i, &d) in shape.iter().enumerate().rev() {
        strides[i] = acc;
        acc *= d as isize;
    }
    strides
}

#[derive(Clone, Debug)]
pub enum TensorData {
    Cpu(Arc<Vec<f32>>),
    #[cfg(feature = "cuda")]
    Cuda(Arc<CudaSlice<f32>>),
}

#[derive(Debug)]
pub enum TensorGrad {
    Cpu(Vec<f32>),
    #[cfg(feature = "cuda")]
    Cuda(CudaSlice<f32>),
}

#[derive(Clone, Debug)]
pub enum TensorOp {
    None,
    Add(Rc<RefCell<TensorInner>>, Rc<RefCell<TensorInner>>),
    Mul(Rc<RefCell<TensorInner>>, Rc<RefCell<TensorInner>>),
    Pow(Rc<RefCell<TensorInner>>, f32),
    ReLU(Rc<RefCell<TensorInner>>),
    Sum(Rc<RefCell<TensorInner>>),
}

#[derive(Debug)]
pub struct TensorInner {
    pub data: TensorData,
    pub view: TensorView,
    pub grad: Option<TensorGrad>,
    pub op: TensorOp,
    pub device: DeviceType,
}

impl TensorInner {
    pub fn backward_step_cpu(&self, dout: &[f32]) -> PyResult<()> {
        match &self.op {
            TensorOp::None => Ok(()),
            TensorOp::Add(a, b) => {
                add_grad_cpu(a, dout)?;
                add_grad_cpu(b, dout)?;
                Ok(())
            }
            TensorOp::Mul(a, b) => {
                // da += b * dout; db += a * dout
                let a_t = Tensor::new_with_inner(a.clone());
                let b_t = Tensor::new_with_inner(b.clone());
                let (a_buf, a_view) = a_t.to_cpu_contiguous_data()?;
                let (b_buf, b_view) = b_t.to_cpu_contiguous_data()?;
                if a_view.numel != dout.len() || b_view.numel != dout.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "grad shape mismatch in mul backward",
                    ));
                }
                let mut da = vec![0f32; dout.len()];
                let mut db = vec![0f32; dout.len()];
                for i in 0..dout.len() {
                    da[i] = b_buf[i] * dout[i];
                    db[i] = a_buf[i] * dout[i];
                }
                add_grad_cpu(a, &da)?;
                add_grad_cpu(b, &db)?;
                Ok(())
            }
            TensorOp::Pow(a, exp) => {
                let a_t = Tensor::new_with_inner(a.clone());
                let (a_buf, a_view) = a_t.to_cpu_contiguous_data()?;
                if a_view.numel != dout.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "grad shape mismatch in pow backward",
                    ));
                }
                let mut da = vec![0f32; dout.len()];
                for i in 0..dout.len() {
                    da[i] = dout[i] * (exp * a_buf[i].powf(exp - 1.0));
                }
                add_grad_cpu(a, &da)?;
                Ok(())
            }
            TensorOp::ReLU(a) => {
                let a_t = Tensor::new_with_inner(a.clone());
                let (a_buf, a_view) = a_t.to_cpu_contiguous_data()?;
                if a_view.numel != dout.len() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "grad shape mismatch in relu backward",
                    ));
                }
                let mut da = vec![0f32; dout.len()];
                for i in 0..dout.len() {
                    da[i] = if a_buf[i] > 0.0 { dout[i] } else { 0.0 };
                }
                add_grad_cpu(a, &da)?;
                Ok(())
            }
            TensorOp::Sum(a) => {
                // dout is scalar (len=1). da += fill(dout[0])
                if dout.len() != 1 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "sum backward requires scalar upstream grad",
                    ));
                }
                let a_numel = a.borrow().view.numel;
                let fill = vec![dout[0]; a_numel];
                add_grad_cpu(a, &fill)?;
                Ok(())
            }
        }
    }

    pub fn children(&self) -> Vec<Rc<RefCell<TensorInner>>> {
        match &self.op {
            TensorOp::None => vec![],
            TensorOp::Add(a, b) | TensorOp::Mul(a, b) => vec![a.clone(), b.clone()],
            TensorOp::Pow(a, _) | TensorOp::ReLU(a) | TensorOp::Sum(a) => vec![a.clone()],
        }
    }
}

fn add_grad_cpu(target: &Rc<RefCell<TensorInner>>, contrib: &[f32]) -> PyResult<()> {
    let mut t = target.borrow_mut();
    let n = t.view.numel;
    if contrib.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grad contribution length mismatch",
        ));
    }
    if t.grad.is_none() {
        t.grad = Some(TensorGrad::Cpu(vec![0.0; n]));
    }
    match t.grad.as_mut().unwrap() {
        TensorGrad::Cpu(g) => {
            for i in 0..n {
                g[i] += contrib[i];
            }
            Ok(())
        }
        #[cfg(feature = "cuda")]
        TensorGrad::Cuda(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "mixed-device grad accumulation (expected CPU grad)",
        )),
    }
}

#[cfg(feature = "cuda")]
fn add_grad_cuda(target: &Rc<RefCell<TensorInner>>, contrib: CudaSlice<f32>) -> PyResult<()> {
    let mut t = target.borrow_mut();
    let n = t.view.numel;
    if contrib.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "grad contribution length mismatch",
        ));
    }
    let new_grad = match t.grad.take() {
        None => contrib,
        Some(TensorGrad::Cuda(prev)) => {
            crate::cuda_kernels::cuda_add(&prev, &contrib, n).map_err(map_cuda_err)?
        }
        Some(TensorGrad::Cpu(_)) => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "mixed-device grad accumulation (expected CUDA grad)",
            ))
        }
    };
    t.grad = Some(TensorGrad::Cuda(new_grad));
    Ok(())
}

/// Python-exposed Tensor class (N-D, with strided views).
#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Tensor {
    pub inner: Rc<RefCell<TensorInner>>,
}

impl Tensor {
    fn new_with_inner(inner: Rc<RefCell<TensorInner>>) -> Self {
        Tensor { inner }
    }

    pub fn device_type(&self) -> DeviceType {
        self.inner.borrow().device
    }

    fn build_topo(&self) -> Vec<Rc<RefCell<TensorInner>>> {
        let mut topo = Vec::new();
        let mut visited: HashSet<*const RefCell<TensorInner>> = HashSet::new();

        fn rec(
            v: &Rc<RefCell<TensorInner>>,
            visited: &mut HashSet<*const RefCell<TensorInner>>,
            topo: &mut Vec<Rc<RefCell<TensorInner>>>,
        ) {
            let ptr = Rc::as_ptr(v);
            if visited.insert(ptr) {
                let children = v.borrow().children();
                for c in children {
                    rec(&c, visited, topo);
                }
                topo.push(v.clone());
            }
        }

        rec(&self.inner, &mut visited, &mut topo);
        topo
    }

    fn backward_tensor(&self) -> PyResult<()> {
        let topo = self.build_topo();
        let device = self.device_type();
        let out_numel = self.inner.borrow().view.numel;
        if out_numel != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Tensor.backward() currently requires a scalar tensor (call .sum() first)",
            ));
        }

        match device {
            DeviceType::Cpu => self.backward_cpu(&topo),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => self.backward_cuda(&topo),
        }
    }

    fn backward_cpu(&self, topo: &[Rc<RefCell<TensorInner>>]) -> PyResult<()> {
        {
            let mut out = self.inner.borrow_mut();
            out.grad = Some(TensorGrad::Cpu(vec![1.0; out.view.numel]));
        }

        for v in topo.iter().rev() {
            let dout = {
                let node = v.borrow();
                let dout = match &node.grad {
                    Some(TensorGrad::Cpu(g)) => g.clone(),
                    _ => vec![0.0; node.view.numel],
                };
                dout
            };
            // Use node methods on a borrowed snapshot.
            let node_snapshot = Tensor::new_with_inner(v.clone());
            // `backward_step_cpu` uses `self.op` from the inner; so we need to call on the inner.
            // We'll just borrow and call directly.
            let node_borrow = node_snapshot.inner.borrow();
            node_borrow.backward_step_cpu(&dout)?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn backward_cuda(&self, topo: &[Rc<RefCell<TensorInner>>]) -> PyResult<()> {
        {
            let mut out = self.inner.borrow_mut();
            let ones = crate::cuda_kernels::cuda_fill(1.0, out.view.numel).map_err(map_cuda_err)?;
            out.grad = Some(TensorGrad::Cuda(ones));
        }

        for v in topo.iter().rev() {
            let op = { v.borrow().op.clone() };
            let dout = {
                let node = v.borrow();
                match &node.grad {
                    Some(TensorGrad::Cuda(g)) => g.try_clone().map_err(map_cuda_err)?,
                    _ => crate::cuda_kernels::cuda_fill(0.0, node.view.numel).map_err(map_cuda_err)?,
                }
            };

            match op {
                TensorOp::None => {}
                TensorOp::Add(a, b) => {
                    add_grad_cuda(&a, dout.try_clone().map_err(map_cuda_err)?)?;
                    add_grad_cuda(&b, dout)?;
                }
                TensorOp::Mul(a, b) => {
                    let a_t = Tensor::new_with_inner(a.clone());
                    let b_t = Tensor::new_with_inner(b.clone());
                    let (a_dev, a_view) = a_t.to_cuda_contiguous_slice()?;
                    let (b_dev, b_view) = b_t.to_cuda_contiguous_slice()?;
                    if a_view.numel != b_view.numel || a_view.numel != dout.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "grad shape mismatch in mul backward",
                        ));
                    }
                    let da = crate::cuda_kernels::cuda_mul(b_dev.as_ref(), &dout, a_view.numel).map_err(map_cuda_err)?;
                    let db = crate::cuda_kernels::cuda_mul(a_dev.as_ref(), &dout, a_view.numel).map_err(map_cuda_err)?;
                    add_grad_cuda(&a, da)?;
                    add_grad_cuda(&b, db)?;
                }
                TensorOp::Pow(a, exp) => {
                    let a_t = Tensor::new_with_inner(a.clone());
                    let (a_dev, a_view) = a_t.to_cuda_contiguous_slice()?;
                    if a_view.numel != dout.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "grad shape mismatch in pow backward",
                        ));
                    }
                    let da = crate::cuda_kernels::cuda_pow_backward(a_dev.as_ref(), &dout, exp, a_view.numel)
                        .map_err(map_cuda_err)?;
                    add_grad_cuda(&a, da)?;
                }
                TensorOp::ReLU(a) => {
                    let a_t = Tensor::new_with_inner(a.clone());
                    let (a_dev, a_view) = a_t.to_cuda_contiguous_slice()?;
                    if a_view.numel != dout.len() {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "grad shape mismatch in relu backward",
                        ));
                    }
                    let da = crate::cuda_kernels::cuda_relu_backward(a_dev.as_ref(), &dout, a_view.numel)
                        .map_err(map_cuda_err)?;
                    add_grad_cuda(&a, da)?;
                }
                TensorOp::Sum(a) => {
                    // dout is scalar; fill to input size.
                    if dout.len() != 1 {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "sum backward requires scalar upstream grad",
                        ));
                    }
                    let scalar = crate::device::cuda_ops::with_cuda_stream(|stream| stream.clone_dtoh(&dout))
                        .map_err(map_cuda_err)?[0];
                    let n = a.borrow().view.numel;
                    let fill = crate::cuda_kernels::cuda_fill(scalar, n).map_err(map_cuda_err)?;
                    add_grad_cuda(&a, fill)?;
                }
            }
        }
        Ok(())
    }

    fn ensure_same_device(&self, other: &Tensor) -> PyResult<DeviceType> {
        let a = self.device_type();
        let b = other.device_type();
        if a != b {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tensors must be on the same device (no implicit device transfers)",
            ));
        }
        Ok(a)
    }

    fn ensure_same_shape(&self, other: &Tensor) -> PyResult<Vec<usize>> {
        let a_shape = self.inner.borrow().view.shape.clone();
        let b_shape = other.inner.borrow().view.shape.clone();
        if a_shape != b_shape {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "shape mismatch (no broadcasting supported)",
            ));
        }
        Ok(a_shape)
    }

    fn to_cpu_contiguous_data(&self) -> PyResult<(Arc<Vec<f32>>, TensorView)> {
        // Returns CPU contiguous buffer + view (contiguous).
        let contig = if self.is_contiguous() { self.clone() } else { self.contiguous()? };
        let contig = match contig.device_type() {
            DeviceType::Cpu => contig,
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => contig.cpu()?,
        };
        let inner = contig.inner.borrow();
        let buf = match &inner.data {
            TensorData::Cpu(buf) => buf,
            #[cfg(feature = "cuda")]
            TensorData::Cuda(_) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err("expected CPU tensor"))
            }
        };
        Ok((buf.clone(), inner.view.clone()))
    }

    #[cfg(feature = "cuda")]
    fn to_cuda_contiguous_slice(&self) -> PyResult<(Arc<CudaSlice<f32>>, TensorView)> {
        let contig = if self.is_contiguous() {
            self.clone()
        } else {
            self.contiguous()?
        };
        let contig = match contig.device_type() {
            DeviceType::Cuda => contig,
            DeviceType::Cpu => contig.cuda()?,
        };
        let inner = contig.inner.borrow();
        let TensorData::Cuda(buf) = &inner.data else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("expected CUDA tensor"));
        };
        Ok((buf.clone(), inner.view.clone()))
    }
}

#[pymethods]
impl Tensor {
    #[new]
    #[pyo3(signature = (data, shape=None, device=None))]
    fn py_new(py: Python<'_>, data: &Bound<'_, PyAny>, shape: Option<Vec<usize>>, device: Option<Device>) -> PyResult<Self> {
        let device_type = device.map(|d| d.device_type).unwrap_or(DeviceType::Cpu);
        let (flat, inferred_shape) = flatten_py_any_f32(py, data)?;
        let shape = match shape {
            Some(s) => s,
            None => inferred_shape,
        };
        let view = TensorView::new_contiguous(shape)?;
        if view.numel != flat.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "data length {} does not match numel {}",
                flat.len(),
                view.numel
            )));
        }

        let data = match device_type {
            DeviceType::Cpu => TensorData::Cpu(Arc::new(flat)),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => {
                if !crate::device::is_cuda_available() {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err("CUDA is not available"));
                }
                let slice = crate::device::cuda_ops::with_cuda_stream(|stream| stream.clone_htod(&flat))
                    .map_err(map_cuda_err)?;
                TensorData::Cuda(Arc::new(slice))
            }
        };

        Ok(Tensor::new_with_inner(Rc::new(RefCell::new(TensorInner {
            data,
            view,
            grad: None,
            op: TensorOp::None,
            device: device_type,
        }))))
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.borrow().view.shape.clone()
    }

    #[getter]
    fn strides(&self) -> Vec<isize> {
        self.inner.borrow().view.strides.clone()
    }

    #[getter]
    fn device(&self) -> Device {
        Device {
            device_type: self.inner.borrow().device,
        }
    }

    fn is_contiguous(&self) -> bool {
        self.inner.borrow().view.is_contiguous()
    }

    fn reshape(&self, shape: Vec<usize>) -> PyResult<Tensor> {
        let inner = self.inner.borrow();
        match inner.view.reshape_view(shape.clone()) {
            Ok(new_view) => Ok(Tensor::new_with_inner(Rc::new(RefCell::new(TensorInner {
                data: inner.data.clone(),
                view: new_view,
                grad: None,
                op: TensorOp::None,
                device: inner.device,
            })))),
            Err(_) => {
                drop(inner);
                self.contiguous()?.reshape(shape)
            }
        }
    }

    fn transpose(&self, dim0: usize, dim1: usize) -> PyResult<Tensor> {
        let inner = self.inner.borrow();
        let new_view = inner.view.transpose_view(dim0, dim1)?;
        Ok(Tensor::new_with_inner(Rc::new(RefCell::new(TensorInner {
            data: inner.data.clone(),
            view: new_view,
            grad: None,
            op: TensorOp::None,
            device: inner.device,
        }))))
    }

    #[pyo3(signature = (dim, start, end, step=1))]
    fn slice(&self, dim: usize, start: isize, end: isize, step: isize) -> PyResult<Tensor> {
        let inner = self.inner.borrow();
        let new_view = inner.view.slice_view(dim, start, end, step)?;
        Ok(Tensor::new_with_inner(Rc::new(RefCell::new(TensorInner {
            data: inner.data.clone(),
            view: new_view,
            grad: None,
            op: TensorOp::None,
            device: inner.device,
        }))))
    }

    fn contiguous(&self) -> PyResult<Tensor> {
        if self.is_contiguous() {
            return Ok(self.clone());
        }
        let inner = self.inner.borrow();
        let view = TensorView::new_contiguous(inner.view.shape.clone())?;
        let out = match &inner.data {
            TensorData::Cpu(buf) => {
                let mut out_buf = vec![0f32; inner.view.numel];
                crate::cpu_kernels::cpu_copy_strided_to_contiguous(&mut out_buf, buf, &inner.view)?;
                TensorData::Cpu(Arc::new(out_buf))
            }
            #[cfg(feature = "cuda")]
            TensorData::Cuda(dev_buf) => {
                // MVP: materialize via host (correctness > performance for non-contiguous on GPU).
                let host: Vec<f32> = crate::device::cuda_ops::with_cuda_stream(|stream| stream.clone_dtoh(dev_buf.as_ref()))
                    .map_err(map_cuda_err)?;
                let mut out_buf = vec![0f32; inner.view.numel];
                crate::cpu_kernels::cpu_copy_strided_to_contiguous(&mut out_buf, &Arc::new(host), &inner.view)?;
                let dev_out = crate::device::cuda_ops::with_cuda_stream(|stream| stream.clone_htod(&out_buf))
                    .map_err(map_cuda_err)?;
                TensorData::Cuda(Arc::new(dev_out))
            }
        };
        Ok(Tensor::new_with_inner(Rc::new(RefCell::new(TensorInner {
            data: out,
            view,
            grad: None,
            op: TensorOp::None,
            device: inner.device,
        }))))
    }

    fn to(&self, device: Device) -> PyResult<Tensor> {
        let target = device.device_type;
        let src_device = self.device_type();
        if src_device == target {
            return Ok(self.clone());
        }
        let inner = self.inner.borrow();
        let contig = if inner.view.is_contiguous() {
            self.clone()
        } else {
            drop(inner);
            self.contiguous()?
        };
        let contig_inner = contig.inner.borrow();
        let view = contig_inner.view.clone(); // contiguous view

        let data = match target {
            DeviceType::Cpu => match &contig_inner.data {
                TensorData::Cpu(buf) => TensorData::Cpu(buf.clone()),
                #[cfg(feature = "cuda")]
                TensorData::Cuda(dev_buf) => {
                    let host = crate::device::cuda_ops::with_cuda_stream(|stream| stream.clone_dtoh(dev_buf.as_ref()))
                        .map_err(map_cuda_err)?;
                    TensorData::Cpu(Arc::new(host))
                }
            },
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => match &contig_inner.data {
                TensorData::Cpu(buf) => {
                    if !crate::device::is_cuda_available() {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err("CUDA is not available"));
                    }
                    let dev_buf =
                        crate::device::cuda_ops::with_cuda_stream(|stream| stream.clone_htod(buf.as_slice()))
                            .map_err(map_cuda_err)?;
                    TensorData::Cuda(Arc::new(dev_buf))
                }
                TensorData::Cuda(buf) => TensorData::Cuda(buf.clone()),
            },
        };

        Ok(Tensor::new_with_inner(Rc::new(RefCell::new(TensorInner {
            data,
            view,
            grad: None,
            op: TensorOp::None,
            device: target,
        }))))
    }

    fn cpu(&self) -> PyResult<Tensor> {
        self.to(Device {
            device_type: DeviceType::Cpu,
        })
    }

    #[cfg(feature = "cuda")]
    fn cuda(&self) -> PyResult<Tensor> {
        if crate::device::is_cuda_available() {
            self.to(Device {
                device_type: DeviceType::Cuda,
            })
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err("CUDA is not available"))
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn cuda(&self) -> PyResult<Tensor> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA support is not enabled. Please rebuild with the 'cuda' feature enabled.",
        ))
    }

    fn tolist(&self, py: Python<'_>) -> PyResult<PyObject> {
        // ensure we can read a CPU buffer
        let cpu_tensor = match self.device_type() {
            DeviceType::Cpu => self.clone(),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => self.cpu()?,
        };
        let inner = cpu_tensor.inner.borrow();
        let buf = match &inner.data {
            TensorData::Cpu(buf) => buf,
            #[cfg(feature = "cuda")]
            TensorData::Cuda(_) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err("expected CPU tensor"))
            }
        };
        build_py_nested_list(py, buf, &inner.view)
    }

    /// Backprop through the Tensor graph (requires scalar output: call `.sum()` first).
    fn backward(&self) -> PyResult<()> {
        self.backward_tensor()
    }

    /// Gradient tensor (same shape as `self`), or None if gradients have not been computed.
    #[getter]
    fn grad(&self) -> PyResult<Option<Tensor>> {
        let inner = self.inner.borrow();
        let Some(g) = &inner.grad else { return Ok(None) };
        let view = TensorView::new_contiguous(inner.view.shape.clone())?;
        let data = match g {
            TensorGrad::Cpu(v) => TensorData::Cpu(Arc::new(v.clone())),
            #[cfg(feature = "cuda")]
            TensorGrad::Cuda(dev) => {
                let cloned = dev.try_clone().map_err(map_cuda_err)?;
                TensorData::Cuda(Arc::new(cloned))
            }
        };
        Ok(Some(Tensor::new_with_inner(Rc::new(RefCell::new(TensorInner {
            data,
            view,
            grad: None,
            op: TensorOp::None,
            device: inner.device,
        })))))
    }

    fn __repr__(&self) -> String {
        let inner = self.inner.borrow();
        format!(
            "Tensor(shape={:?}, strides={:?}, device={})",
            inner.view.shape, inner.view.strides, inner.device
        )
    }

    fn __add__(&self, other: Tensor) -> PyResult<Tensor> {
        self.add(&other)
    }

    fn __mul__(&self, other: Tensor) -> PyResult<Tensor> {
        self.mul(&other)
    }

    fn __pow__(&self, exp: f32, _modulo: Option<f32>) -> PyResult<Tensor> {
        self.pow_scalar(exp)
    }

    fn relu(&self) -> PyResult<Tensor> {
        self.relu_tensor()
    }

    fn sum(&self) -> PyResult<Tensor> {
        self.sum_all()
    }
}

impl Tensor {
    pub fn add(&self, other: &Tensor) -> PyResult<Tensor> {
        let device = self.ensure_same_device(other)?;
        let shape = self.ensure_same_shape(other)?;

        let out_view = TensorView::new_contiguous(shape)?;
        let out_data = match device {
            DeviceType::Cpu => {
                let (a_buf, a_view) = self.to_cpu_contiguous_data()?;
                let (b_buf, b_view) = other.to_cpu_contiguous_data()?;
                let mut out = vec![0f32; a_view.numel];
                crate::cpu_kernels::cpu_ew_binary(&mut out, &a_buf, &a_view, &b_buf, &b_view, |x, y| x + y)?;
                TensorData::Cpu(Arc::new(out))
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => {
                let (a_dev, a_view) = self.to_cuda_contiguous_slice()?;
                let (b_dev, b_view) = other.to_cuda_contiguous_slice()?;
                if a_view.numel != b_view.numel {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "shape mismatch (no broadcasting supported)",
                    ));
                }
                let out = crate::cuda_kernels::cuda_add(a_dev.as_ref(), b_dev.as_ref(), a_view.numel)
                    .map_err(map_cuda_err)?;
                TensorData::Cuda(Arc::new(out))
            }
        };

        let op = TensorOp::Add(self.inner.clone(), other.inner.clone());
        Ok(Tensor::new_with_inner(Rc::new(RefCell::new(TensorInner {
            data: out_data,
            view: out_view,
            grad: None,
            op,
            device,
        }))))
    }

    pub fn mul(&self, other: &Tensor) -> PyResult<Tensor> {
        let device = self.ensure_same_device(other)?;
        let shape = self.ensure_same_shape(other)?;
        let out_view = TensorView::new_contiguous(shape)?;
        let out_data = match device {
            DeviceType::Cpu => {
                let (a_buf, a_view) = self.to_cpu_contiguous_data()?;
                let (b_buf, b_view) = other.to_cpu_contiguous_data()?;
                let mut out = vec![0f32; a_view.numel];
                crate::cpu_kernels::cpu_ew_binary(&mut out, &a_buf, &a_view, &b_buf, &b_view, |x, y| x * y)?;
                TensorData::Cpu(Arc::new(out))
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => {
                let (a_dev, a_view) = self.to_cuda_contiguous_slice()?;
                let (b_dev, b_view) = other.to_cuda_contiguous_slice()?;
                if a_view.numel != b_view.numel {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "shape mismatch (no broadcasting supported)",
                    ));
                }
                let out = crate::cuda_kernels::cuda_mul(a_dev.as_ref(), b_dev.as_ref(), a_view.numel)
                    .map_err(map_cuda_err)?;
                TensorData::Cuda(Arc::new(out))
            }
        };

        let op = TensorOp::Mul(self.inner.clone(), other.inner.clone());
        Ok(Tensor::new_with_inner(Rc::new(RefCell::new(TensorInner {
            data: out_data,
            view: out_view,
            grad: None,
            op,
            device,
        }))))
    }

    pub fn pow_scalar(&self, exp: f32) -> PyResult<Tensor> {
        let device = self.device_type();
        let shape = self.inner.borrow().view.shape.clone();
        let out_view = TensorView::new_contiguous(shape)?;
        let out_data = match device {
            DeviceType::Cpu => {
                let (x_buf, x_view) = self.to_cpu_contiguous_data()?;
                let mut out = vec![0f32; x_view.numel];
                crate::cpu_kernels::cpu_ew_unary(&mut out, &x_buf, &x_view, |x| x.powf(exp))?;
                TensorData::Cpu(Arc::new(out))
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => {
                let (x_dev, x_view) = self.to_cuda_contiguous_slice()?;
                let out = crate::cuda_kernels::cuda_pow_scalar(x_dev.as_ref(), exp, x_view.numel)
                    .map_err(map_cuda_err)?;
                TensorData::Cuda(Arc::new(out))
            }
        };

        let op = TensorOp::Pow(self.inner.clone(), exp);
        Ok(Tensor::new_with_inner(Rc::new(RefCell::new(TensorInner {
            data: out_data,
            view: out_view,
            grad: None,
            op,
            device,
        }))))
    }

    pub fn relu_tensor(&self) -> PyResult<Tensor> {
        let device = self.device_type();
        let shape = self.inner.borrow().view.shape.clone();
        let out_view = TensorView::new_contiguous(shape)?;
        let out_data = match device {
            DeviceType::Cpu => {
                let (x_buf, x_view) = self.to_cpu_contiguous_data()?;
                let mut out = vec![0f32; x_view.numel];
                crate::cpu_kernels::cpu_ew_unary(&mut out, &x_buf, &x_view, |x| if x > 0.0 { x } else { 0.0 })?;
                TensorData::Cpu(Arc::new(out))
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => {
                let (x_dev, x_view) = self.to_cuda_contiguous_slice()?;
                let out = crate::cuda_kernels::cuda_relu(x_dev.as_ref(), x_view.numel)
                    .map_err(map_cuda_err)?;
                TensorData::Cuda(Arc::new(out))
            }
        };

        let op = TensorOp::ReLU(self.inner.clone());
        Ok(Tensor::new_with_inner(Rc::new(RefCell::new(TensorInner {
            data: out_data,
            view: out_view,
            grad: None,
            op,
            device,
        }))))
    }

    pub fn sum_all(&self) -> PyResult<Tensor> {
        let device = self.device_type();
        let out_view = TensorView::new_contiguous(vec![])?;
        let out_data = match device {
            DeviceType::Cpu => {
                let inner = self.inner.borrow();
                let buf = match &inner.data {
                    TensorData::Cpu(buf) => buf,
                    #[cfg(feature = "cuda")]
                    TensorData::Cuda(_) => {
                        return Err(pyo3::exceptions::PyRuntimeError::new_err("expected CPU tensor"))
                    }
                };
                let s = crate::cpu_kernels::cpu_sum_all(buf, &inner.view)?;
                TensorData::Cpu(Arc::new(vec![s]))
            }
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => {
                let (x_dev, x_view) = self.to_cuda_contiguous_slice()?;
                let out = crate::cuda_kernels::cuda_sum_all(x_dev.as_ref(), x_view.numel)
                    .map_err(map_cuda_err)?;
                TensorData::Cuda(Arc::new(out))
            }
        };

        let op = TensorOp::Sum(self.inner.clone());
        Ok(Tensor::new_with_inner(Rc::new(RefCell::new(TensorInner {
            data: out_data,
            view: out_view,
            grad: None,
            op,
            device,
        }))))
    }
}

fn flatten_py_any_f32(py: Python<'_>, any: &Bound<'_, PyAny>) -> PyResult<(Vec<f32>, Vec<usize>)> {
    // Two cases:
    // - scalar -> shape=[]
    // - nested lists -> infer shape, flatten row-major
    if let Ok(v) = any.extract::<f32>() {
        return Ok((vec![v], vec![]));
    }
    if let Ok(list) = any.downcast::<PyList>() {
        let mut flat = Vec::new();
        let shape = infer_shape_from_list(py, list)?;
        flatten_list_into(py, list, &mut flat)?;
        Ok((flat, shape))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Tensor data must be a number or a (nested) Python list of numbers",
        ))
    }
}

fn infer_shape_from_list(_py: Python<'_>, list: &Bound<'_, PyList>) -> PyResult<Vec<usize>> {
    let mut shape = Vec::new();
    let mut cur = list.clone();
    loop {
        let len = cur.len();
        shape.push(len);
        if len == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "empty lists are not supported in Tensor construction",
            ));
        }
        let first = cur.get_item(0)?;
        if first.extract::<f32>().is_ok() {
            // leaf
            // validate all are scalars
            for i in 0..len {
                let item = cur.get_item(i)?;
                item.extract::<f32>().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "ragged nested lists are not supported (mixed scalars/lists)",
                    )
                })?;
            }
            return Ok(shape);
        } else if let Ok(next) = first.downcast::<PyList>() {
            // validate all are lists of same length/shape recursively
            for i in 0..len {
                let item = cur.get_item(i)?;
                item.downcast::<PyList>().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err(
                        "ragged nested lists are not supported (mixed scalars/lists)",
                    )
                })?;
            }
            cur = next.clone();
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Tensor data must be numbers or nested lists of numbers",
            ));
        }
    }
}

fn flatten_list_into(_py: Python<'_>, list: &Bound<'_, PyList>, out: &mut Vec<f32>) -> PyResult<()> {
    let len = list.len();
    if len == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "empty lists are not supported in Tensor construction",
        ));
    }
    let first = list.get_item(0)?;
    if first.extract::<f32>().is_ok() {
        for i in 0..len {
            let item = list.get_item(i)?;
            out.push(item.extract::<f32>()?);
        }
        Ok(())
    } else if let Ok(_) = first.downcast::<PyList>() {
        for i in 0..len {
            let item = list.get_item(i)?;
            let sub = item.downcast::<PyList>()?;
            flatten_list_into(_py, &sub, out)?;
        }
        Ok(())
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Tensor data must be numbers or nested lists of numbers",
        ))
    }
}

fn build_py_nested_list(py: Python<'_>, buf: &Arc<Vec<f32>>, view: &TensorView) -> PyResult<PyObject> {
    fn rec(
        py: Python<'_>,
        buf: &Arc<Vec<f32>>,
        view: &TensorView,
        dim: usize,
        base_offset: isize,
    ) -> PyResult<PyObject> {
        if dim == view.shape.len() {
            let idx = base_offset as usize;
            return Ok(buf[idx].into_py(py));
        }
        let len = view.shape[dim];
        let stride = view.strides[dim];
        let py_list = PyList::empty_bound(py);
        for i in 0..len {
            let off = base_offset + (i as isize) * stride;
            py_list.append(rec(py, buf, view, dim + 1, off)?)?;
        }
        Ok(py_list.into_py(py))
    }

    // For scalar tensor (shape=[]), return a python float
    if view.shape.is_empty() {
        return Ok(buf[view.offset].into_py(py));
    }

    rec(py, buf, view, 0, view.offset as isize)
}


