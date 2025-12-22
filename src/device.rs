/// Device abstraction for CPU and GPU execution
use pyo3::prelude::*;

#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    #[pyo3(name = "CPU")]
    Cpu,
    #[pyo3(name = "GPU")]
    Gpu,
}

#[pymethods]
impl Device {
    #[new]
    fn new() -> Self {
        Device::Cpu
    }

    fn __repr__(&self) -> String {
        match self {
            Device::Cpu => "Device.CPU".to_string(),
            Device::Gpu => "Device.GPU".to_string(),
        }
    }

    fn __str__(&self) -> String {
        match self {
            Device::Cpu => "CPU".to_string(),
            Device::Gpu => "GPU".to_string(),
        }
    }

    /// Check if GPU is available on this system
    #[staticmethod]
    fn is_gpu_available() -> bool {
        #[cfg(feature = "gpu")]
        {
            use crate::gpu::GpuContext;
            GpuContext::is_available()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Create a CPU device
    #[staticmethod]
    fn cpu() -> Self {
        Device::Cpu
    }

    /// Create a GPU device (if available)
    #[staticmethod]
    fn gpu() -> PyResult<Self> {
        #[cfg(feature = "gpu")]
        {
            use crate::gpu::GpuContext;
            if GpuContext::is_available() {
                Ok(Device::Gpu)
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "GPU is not available on this system",
                ))
            }
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "GPU support was not compiled into this build. Rebuild with --features gpu",
            ))
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}
