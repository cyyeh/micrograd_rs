use pyo3::prelude::*;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use crate::device::{Device, DeviceType};

/// The operation that produced a Value node
#[derive(Clone, Debug)]
pub enum Op {
    None,
    Add(Rc<RefCell<ValueInner>>, Rc<RefCell<ValueInner>>),
    Mul(Rc<RefCell<ValueInner>>, Rc<RefCell<ValueInner>>),
    Pow(Rc<RefCell<ValueInner>>, f64),
    ReLU(Rc<RefCell<ValueInner>>),
}

/// Inner value storing the actual data and gradient
#[derive(Clone, Debug)]
pub struct ValueInner {
    pub data: f64,
    pub grad: f64,
    pub op: Op,
    pub device: DeviceType,
}

impl ValueInner {
    pub fn new(data: f64) -> Self {
        ValueInner {
            data,
            grad: 0.0,
            op: Op::None,
            device: DeviceType::Cpu,
        }
    }

    pub fn new_with_device(data: f64, device: DeviceType) -> Self {
        ValueInner {
            data,
            grad: 0.0,
            op: Op::None,
            device,
        }
    }

    pub fn with_op(data: f64, op: Op) -> Self {
        ValueInner {
            data,
            grad: 0.0,
            op,
            device: DeviceType::Cpu,
        }
    }

    pub fn with_op_and_device(data: f64, op: Op, device: DeviceType) -> Self {
        ValueInner {
            data,
            grad: 0.0,
            op,
            device,
        }
    }

    /// Compute backward pass for this node
    pub fn backward_step(&self) {
        match &self.op {
            Op::None => {}
            Op::Add(left, right) => {
                left.borrow_mut().grad += self.grad;
                right.borrow_mut().grad += self.grad;
            }
            Op::Mul(left, right) => {
                let left_data = left.borrow().data;
                let right_data = right.borrow().data;
                left.borrow_mut().grad += right_data * self.grad;
                right.borrow_mut().grad += left_data * self.grad;
            }
            Op::Pow(base, exp) => {
                let base_data = base.borrow().data;
                base.borrow_mut().grad += (exp * base_data.powf(exp - 1.0)) * self.grad;
            }
            Op::ReLU(input) => {
                if self.data > 0.0 {
                    input.borrow_mut().grad += self.grad;
                }
            }
        }
    }

    /// Get children of this node for topological sort
    pub fn children(&self) -> Vec<Rc<RefCell<ValueInner>>> {
        match &self.op {
            Op::None => vec![],
            Op::Add(left, right) | Op::Mul(left, right) => vec![left.clone(), right.clone()],
            Op::Pow(base, _) => vec![base.clone()],
            Op::ReLU(input) => vec![input.clone()],
        }
    }
}

/// Python-exposed Value class wrapping the inner value
#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Value {
    pub inner: Rc<RefCell<ValueInner>>,
}

impl Value {
    pub fn new_with_inner(inner: Rc<RefCell<ValueInner>>) -> Self {
        Value { inner }
    }

    pub fn from_f64(data: f64) -> Self {
        Value {
            inner: Rc::new(RefCell::new(ValueInner::new(data))),
        }
    }

    pub fn from_f64_with_device(data: f64, device: DeviceType) -> Self {
        Value {
            inner: Rc::new(RefCell::new(ValueInner::new_with_device(data, device))),
        }
    }

    /// Get the device type for this value
    pub fn get_device(&self) -> DeviceType {
        self.inner.borrow().device
    }

    /// Compute the result device when combining two values
    /// Both values must be on the same device
    fn result_device(&self, other: &Value) -> DeviceType {
        let self_device = self.inner.borrow().device;
        let other_device = other.inner.borrow().device;
        // For now, we require both operands to be on the same device
        // The result will be on that device
        if self_device == other_device {
            self_device
        } else {
            // Default to CPU if devices don't match (caller should ensure they match)
            DeviceType::Cpu
        }
    }

    pub fn add(&self, other: &Value) -> Value {
        let device = self.result_device(other);
        let data = compute_add(self.inner.borrow().data, other.inner.borrow().data, device);
        let op = Op::Add(self.inner.clone(), other.inner.clone());
        Value::new_with_inner(Rc::new(RefCell::new(ValueInner::with_op_and_device(data, op, device))))
    }

    pub fn mul(&self, other: &Value) -> Value {
        let device = self.result_device(other);
        let data = compute_mul(self.inner.borrow().data, other.inner.borrow().data, device);
        let op = Op::Mul(self.inner.clone(), other.inner.clone());
        Value::new_with_inner(Rc::new(RefCell::new(ValueInner::with_op_and_device(data, op, device))))
    }

    pub fn pow_f64(&self, exp: f64) -> Value {
        let device = self.inner.borrow().device;
        let data = compute_pow(self.inner.borrow().data, exp, device);
        let op = Op::Pow(self.inner.clone(), exp);
        Value::new_with_inner(Rc::new(RefCell::new(ValueInner::with_op_and_device(data, op, device))))
    }

    pub fn relu_value(&self) -> Value {
        let device = self.inner.borrow().device;
        let input_data = self.inner.borrow().data;
        let data = compute_relu(input_data, device);
        let op = Op::ReLU(self.inner.clone());
        Value::new_with_inner(Rc::new(RefCell::new(ValueInner::with_op_and_device(data, op, device))))
    }

    /// Build topological order of all nodes in the graph
    fn build_topo(&self) -> Vec<Rc<RefCell<ValueInner>>> {
        let mut topo = Vec::new();
        let mut visited: HashSet<*const RefCell<ValueInner>> = HashSet::new();

        fn build_topo_recursive(
            v: &Rc<RefCell<ValueInner>>,
            visited: &mut HashSet<*const RefCell<ValueInner>>,
            topo: &mut Vec<Rc<RefCell<ValueInner>>>,
        ) {
            let ptr = Rc::as_ptr(v);
            if !visited.contains(&ptr) {
                visited.insert(ptr);
                for child in v.borrow().children() {
                    build_topo_recursive(&child, visited, topo);
                }
                topo.push(v.clone());
            }
        }

        build_topo_recursive(&self.inner, &mut visited, &mut topo);
        topo
    }

    pub fn backward_value(&self) {
        let topo = self.build_topo();
        
        // Set gradient of output to 1
        self.inner.borrow_mut().grad = 1.0;
        
        // Backpropagate in reverse topological order
        for v in topo.iter().rev() {
            let inner = v.borrow();
            inner.backward_step();
        }
    }
}

/// Compute add operation, dispatching to CPU or GPU based on device
fn compute_add(a: f64, b: f64, device: DeviceType) -> f64 {
    match device {
        DeviceType::Cpu => a + b,
        #[cfg(feature = "cuda")]
        DeviceType::Cuda => {
            crate::device::cuda_ops::with_cuda_context(|ctx| ctx.add(a, b))
                .unwrap_or(a + b)
        }
    }
}

/// Compute mul operation, dispatching to CPU or GPU based on device
fn compute_mul(a: f64, b: f64, device: DeviceType) -> f64 {
    match device {
        DeviceType::Cpu => a * b,
        #[cfg(feature = "cuda")]
        DeviceType::Cuda => {
            crate::device::cuda_ops::with_cuda_context(|ctx| ctx.mul(a, b))
                .unwrap_or(a * b)
        }
    }
}

/// Compute pow operation, dispatching to CPU or GPU based on device
fn compute_pow(base: f64, exp: f64, device: DeviceType) -> f64 {
    match device {
        DeviceType::Cpu => base.powf(exp),
        #[cfg(feature = "cuda")]
        DeviceType::Cuda => {
            crate::device::cuda_ops::with_cuda_context(|ctx| ctx.pow(base, exp))
                .unwrap_or(base.powf(exp))
        }
    }
}

/// Compute relu operation, dispatching to CPU or GPU based on device
fn compute_relu(x: f64, device: DeviceType) -> f64 {
    match device {
        DeviceType::Cpu => if x > 0.0 { x } else { 0.0 },
        #[cfg(feature = "cuda")]
        DeviceType::Cuda => {
            crate::device::cuda_ops::with_cuda_context(|ctx| ctx.relu(x))
                .unwrap_or(if x > 0.0 { x } else { 0.0 })
        }
    }
}

#[pymethods]
impl Value {
    #[new]
    #[pyo3(signature = (data, device=None))]
    fn py_new(data: f64, device: Option<Device>) -> Self {
        match device {
            Some(d) => Value::from_f64_with_device(data, d.device_type),
            None => Value::from_f64(data),
        }
    }

    #[getter]
    fn data(&self) -> f64 {
        self.inner.borrow().data
    }

    #[setter]
    fn set_data(&self, value: f64) {
        self.inner.borrow_mut().data = value;
    }

    #[getter]
    fn grad(&self) -> f64 {
        self.inner.borrow().grad
    }

    #[setter]
    fn set_grad(&self, value: f64) {
        self.inner.borrow_mut().grad = value;
    }

    /// Get the device this value is on
    #[getter]
    fn device(&self) -> Device {
        Device {
            device_type: self.inner.borrow().device,
        }
    }

    /// Get the previous nodes (children) in the computation graph
    #[getter]
    fn _prev(&self) -> Vec<Value> {
        self.inner
            .borrow()
            .children()
            .into_iter()
            .map(Value::new_with_inner)
            .collect()
    }

    /// Get the operation that produced this node
    #[getter]
    fn _op(&self) -> String {
        match &self.inner.borrow().op {
            Op::None => String::new(),
            Op::Add(_, _) => "+".to_string(),
            Op::Mul(_, _) => "*".to_string(),
            Op::Pow(_, exp) => format!("**{}", exp),
            Op::ReLU(_) => "ReLU".to_string(),
        }
    }

    fn __repr__(&self) -> String {
        let inner = self.inner.borrow();
        format!("Value(data={}, grad={}, device={})", inner.data, inner.grad, inner.device)
    }

    /// Move the value to a different device
    fn to(&self, device: Device) -> Value {
        let data = self.inner.borrow().data;
        Value::from_f64_with_device(data, device.device_type)
    }

    /// Move the value to CPU
    fn cpu(&self) -> Value {
        let data = self.inner.borrow().data;
        Value::from_f64_with_device(data, DeviceType::Cpu)
    }

    /// Move the value to CUDA (requires cuda feature)
    #[cfg(feature = "cuda")]
    fn cuda(&self) -> PyResult<Value> {
        if crate::device::is_cuda_available() {
            let data = self.inner.borrow().data;
            Ok(Value::from_f64_with_device(data, DeviceType::Cuda))
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "CUDA is not available",
            ))
        }
    }

    /// Move the value to CUDA - returns error when cuda feature is not enabled
    #[cfg(not(feature = "cuda"))]
    fn cuda(&self) -> PyResult<Value> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA support is not enabled. Please rebuild with the 'cuda' feature enabled.",
        ))
    }

    fn __add__(&self, other: ValueOrFloat) -> Value {
        let device = self.get_device();
        match other {
            ValueOrFloat::Value(v) => self.add(&v),
            ValueOrFloat::Float(f) => self.add(&Value::from_f64_with_device(f, device)),
        }
    }

    fn __radd__(&self, other: ValueOrFloat) -> Value {
        self.__add__(other)
    }

    fn __mul__(&self, other: ValueOrFloat) -> Value {
        let device = self.get_device();
        match other {
            ValueOrFloat::Value(v) => self.mul(&v),
            ValueOrFloat::Float(f) => self.mul(&Value::from_f64_with_device(f, device)),
        }
    }

    fn __rmul__(&self, other: ValueOrFloat) -> Value {
        self.__mul__(other)
    }

    fn __pow__(&self, other: f64, _modulo: Option<f64>) -> Value {
        self.pow_f64(other)
    }

    fn __neg__(&self) -> Value {
        let device = self.get_device();
        self.mul(&Value::from_f64_with_device(-1.0, device))
    }

    fn __sub__(&self, other: ValueOrFloat) -> Value {
        let device = self.get_device();
        let neg_other = match other {
            ValueOrFloat::Value(v) => v.__neg__(),
            ValueOrFloat::Float(f) => Value::from_f64_with_device(-f, device),
        };
        self.add(&neg_other)
    }

    fn __rsub__(&self, other: ValueOrFloat) -> Value {
        let neg_self = self.__neg__();
        let device = self.get_device();
        match other {
            ValueOrFloat::Value(v) => v.add(&neg_self),
            ValueOrFloat::Float(f) => Value::from_f64_with_device(f, device).add(&neg_self),
        }
    }

    fn __truediv__(&self, other: ValueOrFloat) -> Value {
        let device = self.get_device();
        match other {
            ValueOrFloat::Value(v) => self.mul(&v.pow_f64(-1.0)),
            ValueOrFloat::Float(f) => self.mul(&Value::from_f64_with_device(f, device).pow_f64(-1.0)),
        }
    }

    fn __rtruediv__(&self, other: ValueOrFloat) -> Value {
        let inv_self = self.pow_f64(-1.0);
        let device = self.get_device();
        match other {
            ValueOrFloat::Value(v) => v.mul(&inv_self),
            ValueOrFloat::Float(f) => Value::from_f64_with_device(f, device).mul(&inv_self),
        }
    }

    fn relu(&self) -> Value {
        self.relu_value()
    }

    fn backward(&self) {
        self.backward_value();
    }
}

/// Enum to handle both Value and float inputs in Python methods
#[derive(FromPyObject)]
pub enum ValueOrFloat {
    Value(Value),
    Float(f64),
}
