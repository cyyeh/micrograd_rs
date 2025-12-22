mod value;
mod nn;

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn _micrograd_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<value::Value>()?;
    m.add_class::<nn::Neuron>()?;
    m.add_class::<nn::Layer>()?;
    m.add_class::<nn::MLP>()?;
    Ok(())
}

