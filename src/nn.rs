use pyo3::prelude::*;
use rand::Rng;

use crate::value::Value;
use crate::device::Device;

/// A single neuron with weights and bias
#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    nonlin: bool,
    device: Device,
}

#[pymethods]
impl Neuron {
    #[new]
    #[pyo3(signature = (nin, nonlin=true, device=None))]
    fn new(nin: usize, nonlin: bool, device: Option<Device>) -> Self {
        let device = device.unwrap_or(Device::Cpu);
        let mut rng = rand::thread_rng();
        let weights: Vec<Value> = (0..nin)
            .map(|_| Value::from_f64_with_device(rng.gen_range(-1.0..1.0), device))
            .collect();
        let bias = Value::from_f64_with_device(0.0, device);
        
        Neuron {
            weights,
            bias,
            nonlin,
            device,
        }
    }

    fn __call__(&self, x: Vec<Value>) -> Value {
        // Compute weighted sum: sum(wi * xi) + b
        let mut act = self.bias.clone();
        for (wi, xi) in self.weights.iter().zip(x.iter()) {
            let prod = wi.mul(xi);
            act = act.add(&prod);
        }
        
        if self.nonlin {
            act.relu_value()
        } else {
            act
        }
    }

    fn parameters(&self) -> Vec<Value> {
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    }

    fn zero_grad(&self) {
        for p in self.parameters() {
            p.inner.borrow_mut().grad = 0.0;
        }
    }

    fn __repr__(&self) -> String {
        let name = if self.nonlin { "ReLU" } else { "Linear" };
        format!("{}Neuron({})", name, self.weights.len())
    }
}

/// A layer of neurons
#[pyclass(unsendable)]
#[derive(Clone)]
pub struct Layer {
    neurons: Vec<Neuron>,
    device: Device,
}

#[pymethods]
impl Layer {
    #[new]
    #[pyo3(signature = (nin, nout, nonlin=true, device=None))]
    fn new(nin: usize, nout: usize, nonlin: bool, device: Option<Device>) -> Self {
        let device = device.unwrap_or(Device::Cpu);
        let neurons: Vec<Neuron> = (0..nout)
            .map(|_| Neuron::new(nin, nonlin, Some(device)))
            .collect();
        
        Layer { neurons, device }
    }

    fn __call__(&self, x: Vec<Value>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let out: Vec<Value> = self.neurons
                .iter()
                .map(|n| n.__call__(x.clone()))
                .collect();
            
            if out.len() == 1 {
                Ok(out[0].clone().into_py(py))
            } else {
                Ok(out.into_py(py))
            }
        })
    }

    fn parameters(&self) -> Vec<Value> {
        self.neurons
            .iter()
            .flat_map(|n| n.parameters())
            .collect()
    }

    fn zero_grad(&self) {
        for p in self.parameters() {
            p.inner.borrow_mut().grad = 0.0;
        }
    }

    fn __repr__(&self) -> String {
        let neuron_strs: Vec<String> = self.neurons.iter().map(|n| n.__repr__()).collect();
        format!("Layer of [{}]", neuron_strs.join(", "))
    }
}

/// Multi-layer perceptron
#[pyclass(unsendable)]
#[derive(Clone)]
pub struct MLP {
    layers: Vec<Layer>,
    device: Device,
}

#[pymethods]
impl MLP {
    #[new]
    #[pyo3(signature = (nin, nouts, device=None))]
    fn new(nin: usize, nouts: Vec<usize>, device: Option<Device>) -> Self {
        let device = device.unwrap_or(Device::Cpu);
        let mut sz = vec![nin];
        sz.extend(nouts.clone());
        
        let layers: Vec<Layer> = (0..nouts.len())
            .map(|i| {
                // Last layer is linear (no nonlinearity)
                let nonlin = i != nouts.len() - 1;
                Layer::new(sz[i], sz[i + 1], nonlin, Some(device))
            })
            .collect();
        
        MLP { layers, device }
    }

    fn __call__(&self, x: Vec<Value>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut current: PyObject = x.into_py(py);
            
            for layer in &self.layers {
                // Convert PyObject back to Vec<Value> for the layer
                let x_vec: Vec<Value> = current.extract(py)?;
                current = layer.__call__(x_vec)?;
            }
            
            Ok(current)
        })
    }

    fn parameters(&self) -> Vec<Value> {
        self.layers
            .iter()
            .flat_map(|l| l.parameters())
            .collect()
    }

    fn zero_grad(&self) {
        for p in self.parameters() {
            p.inner.borrow_mut().grad = 0.0;
        }
    }

    fn __repr__(&self) -> String {
        let layer_strs: Vec<String> = self.layers.iter().map(|l| l.__repr__()).collect();
        format!("MLP of [{}]", layer_strs.join(", "))
    }
}
