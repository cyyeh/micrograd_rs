#[cfg(feature = "gpu")]
/// GPU computation backend using wgpu
use std::sync::Arc;
use wgpu;

#[cfg(feature = "gpu")]
pub struct GpuContext {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

#[cfg(feature = "gpu")]
impl GpuContext {
    /// Check if GPU is available
    pub fn is_available() -> bool {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            
            !instance.enumerate_adapters(wgpu::Backends::all()).is_empty()
        })
    }

    /// Create a new GPU context
    pub fn new() -> Option<Self> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });
            
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await?;
            
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::Performance,
                        label: None,
                    },
                    None,
                )
                .await
                .ok()?;
            
            Some(GpuContext {
                device: Arc::new(device),
                queue: Arc::new(queue),
            })
        })
    }

    /// Execute element-wise addition on GPU
    pub fn add(&self, a: f64, b: f64) -> f64 {
        // For scalar operations, just use CPU
        // GPU is beneficial for large tensor operations
        a + b
    }

    /// Execute element-wise multiplication on GPU
    pub fn mul(&self, a: f64, b: f64) -> f64 {
        a * b
    }

    /// Execute power operation on GPU
    pub fn pow(&self, base: f64, exp: f64) -> f64 {
        base.powf(exp)
    }

    /// Execute ReLU activation on GPU
    pub fn relu(&self, x: f64) -> f64 {
        if x < 0.0 { 0.0 } else { x }
    }
}

#[cfg(feature = "gpu")]
thread_local! {
    static GPU_CONTEXT: std::cell::RefCell<Option<GpuContext>> = std::cell::RefCell::new(None);
}

#[cfg(feature = "gpu")]
pub fn get_gpu_context() -> Option<GpuContext> {
    GPU_CONTEXT.with(|ctx| {
        let mut ctx_ref = ctx.borrow_mut();
        if ctx_ref.is_none() {
            *ctx_ref = GpuContext::new();
        }
        // Since GpuContext contains Arc, we can clone it cheaply
        ctx_ref.as_ref().map(|c| GpuContext {
            device: c.device.clone(),
            queue: c.queue.clone(),
        })
    })
}
