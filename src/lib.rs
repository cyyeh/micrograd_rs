mod device;
mod value;
mod nn;
mod tensor;
mod cpu_kernels;
#[cfg(feature = "cuda")]
mod cuda_kernels;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn _micrograd_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<device::Device>()?;
    m.add_class::<value::Value>()?;
    m.add_class::<tensor::Tensor>()?;
    m.add_class::<nn::Neuron>()?;
    m.add_class::<nn::Layer>()?;
    m.add_class::<nn::MLP>()?;
    Ok(())
}

