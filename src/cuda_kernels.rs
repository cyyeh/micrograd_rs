//! CUDA kernel compilation/loading and launch helpers for Tensor ops.
//!
//! This module is only compiled when the `cuda` feature is enabled.

#![cfg(feature = "cuda")]

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;
use std::cell::RefCell;
use std::sync::Arc;

const KERNEL_SRC: &str = r#"
extern "C" __global__ void ew_add_f32(float* out, const float* a, const float* b, unsigned int n) {
    unsigned int idx = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int stride = (unsigned int)(blockDim.x * gridDim.x);
    for (unsigned int i = idx; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}

extern "C" __global__ void ew_mul_f32(float* out, const float* a, const float* b, unsigned int n) {
    unsigned int idx = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int stride = (unsigned int)(blockDim.x * gridDim.x);
    for (unsigned int i = idx; i < n; i += stride) {
        out[i] = a[i] * b[i];
    }
}

extern "C" __global__ void ew_relu_f32(float* out, const float* x, unsigned int n) {
    unsigned int idx = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int stride = (unsigned int)(blockDim.x * gridDim.x);
    for (unsigned int i = idx; i < n; i += stride) {
        float v = x[i];
        out[i] = v > 0.0f ? v : 0.0f;
    }
}

extern "C" __global__ void ew_pow_f32_scalar(float* out, const float* x, float exp, unsigned int n) {
    unsigned int idx = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int stride = (unsigned int)(blockDim.x * gridDim.x);
    for (unsigned int i = idx; i < n; i += stride) {
        out[i] = powf(x[i], exp);
    }
}

extern "C" __global__ void fill_f32(float* out, float value, unsigned int n) {
    unsigned int idx = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int stride = (unsigned int)(blockDim.x * gridDim.x);
    for (unsigned int i = idx; i < n; i += stride) {
        out[i] = value;
    }
}

extern "C" __global__ void ew_relu_backward_f32(float* out, const float* x, const float* dout, unsigned int n) {
    unsigned int idx = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int stride = (unsigned int)(blockDim.x * gridDim.x);
    for (unsigned int i = idx; i < n; i += stride) {
        float xv = x[i];
        out[i] = xv > 0.0f ? dout[i] : 0.0f;
    }
}

extern "C" __global__ void ew_pow_backward_f32(float* out, const float* x, const float* dout, float exp, unsigned int n) {
    unsigned int idx = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int stride = (unsigned int)(blockDim.x * gridDim.x);
    for (unsigned int i = idx; i < n; i += stride) {
        float xv = x[i];
        out[i] = dout[i] * (exp * powf(xv, exp - 1.0f));
    }
}

extern "C" __global__ void reduce_sum_f32(const float* inp, float* out, unsigned int n) {
    extern __shared__ float sdata[];
    unsigned int tid = (unsigned int)threadIdx.x;
    unsigned int i = (unsigned int)(blockIdx.x * blockDim.x * 2 + threadIdx.x);
    float sum = 0.0f;
    if (i < n) sum += inp[i];
    if (i + blockDim.x < n) sum += inp[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}
"#;

#[derive(Clone)]
struct KernelCache {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    ew_add: CudaFunction,
    ew_mul: CudaFunction,
    ew_relu: CudaFunction,
    ew_pow: CudaFunction,
    fill: CudaFunction,
    relu_bwd: CudaFunction,
    pow_bwd: CudaFunction,
    reduce_sum: CudaFunction,
}

thread_local! {
    static KERNELS: RefCell<Option<KernelCache>> = const { RefCell::new(None) };
}

fn init_kernels(ctx: &Arc<CudaContext>) -> Result<KernelCache, DriverError> {
    let ptx = compile_ptx(KERNEL_SRC).map_err(|_| DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE))?;
    let module = ctx.load_module(ptx)?;
    let ew_add = module.load_function("ew_add_f32")?;
    let ew_mul = module.load_function("ew_mul_f32")?;
    let ew_relu = module.load_function("ew_relu_f32")?;
    let ew_pow = module.load_function("ew_pow_f32_scalar")?;
    let fill = module.load_function("fill_f32")?;
    let relu_bwd = module.load_function("ew_relu_backward_f32")?;
    let pow_bwd = module.load_function("ew_pow_backward_f32")?;
    let reduce_sum = module.load_function("reduce_sum_f32")?;
    Ok(KernelCache {
        module,
        ew_add,
        ew_mul,
        ew_relu,
        ew_pow,
        fill,
        relu_bwd,
        pow_bwd,
        reduce_sum,
    })
}

fn with_kernels<F, R>(f: F) -> Result<R, DriverError>
where
    F: FnOnce(&KernelCache, &Arc<CudaStream>) -> Result<R, DriverError>,
{
    crate::device::cuda_ops::with_cuda_context(|ctx_holder| {
        KERNELS.with(|k| {
            let mut k_ref = k.borrow_mut();
            if k_ref.is_none() {
                *k_ref = Some(init_kernels(&ctx_holder.context)?);
            }
            f(k_ref.as_ref().unwrap(), &ctx_holder.stream)
        })
    })
}

fn launch_for_numel(numel: usize) -> Result<LaunchConfig, DriverError> {
    if numel > u32::MAX as usize {
        return Err(DriverError(cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE));
    }
    Ok(LaunchConfig::for_num_elems(numel as u32))
}

pub fn cuda_add(a: &CudaSlice<f32>, b: &CudaSlice<f32>, numel: usize) -> Result<CudaSlice<f32>, DriverError> {
    with_kernels(|k, stream| {
        let mut out = stream.alloc_zeros::<f32>(numel)?;
        let cfg = launch_for_numel(numel)?;
        let n = numel as u32;
        unsafe { stream.launch_builder(&k.ew_add).arg(&mut out).arg(a).arg(b).arg(&n).launch(cfg) }?;
        Ok(out)
    })
}

pub fn cuda_mul(a: &CudaSlice<f32>, b: &CudaSlice<f32>, numel: usize) -> Result<CudaSlice<f32>, DriverError> {
    with_kernels(|k, stream| {
        let mut out = stream.alloc_zeros::<f32>(numel)?;
        let cfg = launch_for_numel(numel)?;
        let n = numel as u32;
        unsafe { stream.launch_builder(&k.ew_mul).arg(&mut out).arg(a).arg(b).arg(&n).launch(cfg) }?;
        Ok(out)
    })
}

pub fn cuda_relu(x: &CudaSlice<f32>, numel: usize) -> Result<CudaSlice<f32>, DriverError> {
    with_kernels(|k, stream| {
        let mut out = stream.alloc_zeros::<f32>(numel)?;
        let cfg = launch_for_numel(numel)?;
        let n = numel as u32;
        unsafe { stream.launch_builder(&k.ew_relu).arg(&mut out).arg(x).arg(&n).launch(cfg) }?;
        Ok(out)
    })
}

pub fn cuda_pow_scalar(x: &CudaSlice<f32>, exp: f32, numel: usize) -> Result<CudaSlice<f32>, DriverError> {
    with_kernels(|k, stream| {
        let mut out = stream.alloc_zeros::<f32>(numel)?;
        let cfg = launch_for_numel(numel)?;
        let n = numel as u32;
        unsafe { stream.launch_builder(&k.ew_pow).arg(&mut out).arg(x).arg(&exp).arg(&n).launch(cfg) }?;
        Ok(out)
    })
}

pub fn cuda_fill(value: f32, numel: usize) -> Result<CudaSlice<f32>, DriverError> {
    with_kernels(|k, stream| {
        let mut out = stream.alloc_zeros::<f32>(numel)?;
        let cfg = launch_for_numel(numel)?;
        let n = numel as u32;
        unsafe { stream.launch_builder(&k.fill).arg(&mut out).arg(&value).arg(&n).launch(cfg) }?;
        Ok(out)
    })
}

pub fn cuda_relu_backward(x: &CudaSlice<f32>, dout: &CudaSlice<f32>, numel: usize) -> Result<CudaSlice<f32>, DriverError> {
    with_kernels(|k, stream| {
        let mut out = stream.alloc_zeros::<f32>(numel)?;
        let cfg = launch_for_numel(numel)?;
        let n = numel as u32;
        unsafe { stream.launch_builder(&k.relu_bwd).arg(&mut out).arg(x).arg(dout).arg(&n).launch(cfg) }?;
        Ok(out)
    })
}

pub fn cuda_pow_backward(x: &CudaSlice<f32>, dout: &CudaSlice<f32>, exp: f32, numel: usize) -> Result<CudaSlice<f32>, DriverError> {
    with_kernels(|k, stream| {
        let mut out = stream.alloc_zeros::<f32>(numel)?;
        let cfg = launch_for_numel(numel)?;
        let n = numel as u32;
        unsafe { stream.launch_builder(&k.pow_bwd).arg(&mut out).arg(x).arg(dout).arg(&exp).arg(&n).launch(cfg) }?;
        Ok(out)
    })
}

pub fn cuda_sum_all(x: &CudaSlice<f32>, numel: usize) -> Result<CudaSlice<f32>, DriverError> {
    with_kernels(|k, stream| {
        if numel == 0 {
            return Ok(stream.clone_htod(&[0.0f32])?);
        }

        let mut cur_inp: CudaSlice<f32> = x.try_clone()?;
        let mut cur_n = numel;

        // Iteratively reduce until a single value remains.
        loop {
            if cur_n <= 1 {
                break;
            }
            // each block reduces 2*blockDim elements
            let threads: u32 = 1024;
            let blocks = ((cur_n as u64) + (threads as u64 * 2 - 1)) / (threads as u64 * 2);
            let blocks_u32 = blocks.min(u32::MAX as u64) as u32;
            let mut out = stream.alloc_zeros::<f32>(blocks_u32 as usize)?;
            let cfg = LaunchConfig {
                grid_dim: (blocks_u32, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: threads * 4,
            };
            let n_u32 = cur_n as u32;
            unsafe {
                stream
                    .launch_builder(&k.reduce_sum)
                    .arg(&cur_inp)
                    .arg(&mut out)
                    .arg(&n_u32)
                    .launch(cfg)
            }?;
            cur_inp = out;
            cur_n = blocks_u32 as usize;
        }

        // cur_inp now has len=1 (or more if something went wrong); ensure scalar.
        if cur_inp.len() != 1 {
            // If reduction didn't fully converge (shouldn't happen), do a final dtoh.
            let host = stream.clone_dtoh(&cur_inp)?;
            let s: f32 = host.into_iter().sum();
            return Ok(stream.clone_htod(&[s])?);
        }
        Ok(cur_inp)
    })
}


