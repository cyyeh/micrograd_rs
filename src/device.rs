//! Device abstraction for CPU and GPU execution.
//!
//! This module provides a unified interface for running operations on different
//! compute devices (CPU or NVIDIA GPU via CUDA).

use pyo3::prelude::*;

/// Represents the compute device where operations are executed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DeviceType {
    /// CPU execution (default)
    #[default]
    Cpu,
    /// NVIDIA GPU execution via CUDA
    #[cfg(feature = "cuda")]
    Cuda,
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "cpu"),
            #[cfg(feature = "cuda")]
            DeviceType::Cuda => write!(f, "cuda"),
        }
    }
}

/// Python-exposed Device class for device selection
#[pyclass]
#[derive(Clone)]
pub struct Device {
    pub device_type: DeviceType,
}

#[pymethods]
impl Device {
    /// Create a CPU device
    #[staticmethod]
    fn cpu() -> Self {
        Device {
            device_type: DeviceType::Cpu,
        }
    }

    /// Create a CUDA device (requires cuda feature)
    #[staticmethod]
    #[cfg(feature = "cuda")]
    fn cuda() -> PyResult<Self> {
        // Check if CUDA is available
        if is_cuda_available() {
            Ok(Device {
                device_type: DeviceType::Cuda,
            })
        } else {
            Err(pyo3::exceptions::PyRuntimeError::new_err(
                "CUDA is not available. Please ensure you have an NVIDIA GPU and CUDA drivers installed.",
            ))
        }
    }

    /// Create a CUDA device - returns error when cuda feature is not enabled
    #[staticmethod]
    #[cfg(not(feature = "cuda"))]
    fn cuda() -> PyResult<Self> {
        Err(pyo3::exceptions::PyRuntimeError::new_err(
            "CUDA support is not enabled. Please rebuild with the 'cuda' feature enabled.",
        ))
    }

    /// Check if CUDA is available
    #[staticmethod]
    fn is_cuda_available() -> bool {
        #[cfg(feature = "cuda")]
        {
            is_cuda_available()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    fn __repr__(&self) -> String {
        format!("Device({})", self.device_type)
    }

    fn __str__(&self) -> String {
        self.device_type.to_string()
    }

    /// Check if this is a CPU device
    fn is_cpu(&self) -> bool {
        self.device_type == DeviceType::Cpu
    }

    /// Check if this is a CUDA device
    fn is_cuda(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.device_type == DeviceType::Cuda
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
}

/// Check if CUDA is available on the system
#[cfg(feature = "cuda")]
pub fn is_cuda_available() -> bool {
    use cudarc::driver::CudaDevice;
    CudaDevice::new(0).is_ok()
}

#[cfg(feature = "cuda")]
pub mod cuda_ops {
    //! CUDA-specific operations for tensor computations.
    //! 
    //! Note: For scalar operations, this implementation uses CPU operations
    //! as they are faster for single values. GPU acceleration becomes beneficial
    //! for batch/tensor operations which could be added in future versions.
    
    use cudarc::driver::CudaDevice;
    use std::sync::Arc;

    /// CUDA context holder for GPU operations
    pub struct CudaContext {
        pub device: Arc<CudaDevice>,
    }

    impl CudaContext {
        /// Create a new CUDA context on device 0
        pub fn new() -> Result<Self, cudarc::driver::DriverError> {
            let device = CudaDevice::new(0)?;
            Ok(CudaContext { device })
        }

        /// Perform addition: result = a + b
        /// Note: For scalar ops, CPU is faster. GPU batching can be added for tensors.
        pub fn add(&self, a: f64, b: f64) -> Result<f64, cudarc::driver::DriverError> {
            Ok(a + b)
        }

        /// Perform multiplication: result = a * b  
        /// Note: For scalar ops, CPU is faster. GPU batching can be added for tensors.
        pub fn mul(&self, a: f64, b: f64) -> Result<f64, cudarc::driver::DriverError> {
            Ok(a * b)
        }

        /// Perform power: result = base^exp
        /// Note: For scalar ops, CPU is faster. GPU batching can be added for tensors.
        pub fn pow(&self, base: f64, exp: f64) -> Result<f64, cudarc::driver::DriverError> {
            Ok(base.powf(exp))
        }

        /// Perform ReLU: result = max(0, x)
        /// Note: For scalar ops, CPU is faster. GPU batching can be added for tensors.
        pub fn relu(&self, x: f64) -> Result<f64, cudarc::driver::DriverError> {
            Ok(if x > 0.0 { x } else { 0.0 })
        }
    }

    /// Thread-local CUDA context for efficiency
    thread_local! {
        static CUDA_CTX: std::cell::RefCell<Option<CudaContext>> = const { std::cell::RefCell::new(None) };
    }

    /// Get or initialize the thread-local CUDA context
    pub fn get_cuda_context() -> Result<(), cudarc::driver::DriverError> {
        CUDA_CTX.with(|ctx| {
            let mut ctx = ctx.borrow_mut();
            if ctx.is_none() {
                *ctx = Some(CudaContext::new()?);
            }
            Ok(())
        })
    }

    /// Execute a CUDA operation using the thread-local context
    pub fn with_cuda_context<F, R>(f: F) -> Result<R, cudarc::driver::DriverError>
    where
        F: FnOnce(&CudaContext) -> Result<R, cudarc::driver::DriverError>,
    {
        CUDA_CTX.with(|ctx| {
            let mut ctx_ref = ctx.borrow_mut();
            if ctx_ref.is_none() {
                *ctx_ref = Some(CudaContext::new()?);
            }
            f(ctx_ref.as_ref().unwrap())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_display() {
        assert_eq!(DeviceType::Cpu.to_string(), "cpu");
        #[cfg(feature = "cuda")]
        assert_eq!(DeviceType::Cuda.to_string(), "cuda");
    }

    #[test]
    fn test_device_default() {
        let device = Device::cpu();
        assert!(device.is_cpu());
        assert!(!device.is_cuda());
    }
}
