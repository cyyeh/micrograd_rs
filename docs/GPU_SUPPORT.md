# GPU Support in micrograd_rs

This document provides detailed information about GPU support in micrograd_rs.

## Overview

micrograd_rs includes GPU acceleration support via [wgpu](https://wgpu.rs/), a cross-platform GPU compute library that provides a modern, safe API for GPU programming in Rust.

## Supported Hardware

The wgpu backend supports a wide range of GPUs across different vendors:

### NVIDIA GPUs
- Supported via Vulkan or CUDA backends
- Recommended: GTX 900 series or newer
- Compute capability 3.5 or higher

### AMD GPUs
- Supported via Vulkan backend
- Recommended: GCN architecture or newer (Radeon HD 7000 series and later)

### Intel GPUs
- Supported via Vulkan backend
- Recommended: Intel HD Graphics 500 series or newer

### Apple Silicon (M1/M2/M3)
- Supported via Metal backend
- All Apple Silicon Macs supported

## How It Works

### Device Abstraction

The library provides a `Device` enum with two variants:
- `Device::Cpu`: CPU execution (default)
- `Device::Gpu`: GPU execution (when available)

### Automatic Fallback

If GPU is not available or GPU operations fail, the library automatically falls back to CPU execution. This ensures your code works everywhere without modification.

### Operation Dispatch

When you create a `Value` with a specific device:

```python
from micrograd_rs import Value, Device

# Create value on GPU
v = Value(3.14, device=Device.gpu())
```

All subsequent operations on that value will execute on the same device:

```python
# These operations happen on GPU
result = v * 2.0 + v ** 2
result.backward()  # Backward pass also on GPU
```

## Performance Considerations

### When GPU Helps

GPU acceleration is most beneficial for:
- **Large neural networks**: Models with many parameters (>1000)
- **Batch processing**: Processing multiple samples simultaneously
- **Deep networks**: Networks with many layers

### When CPU is Faster

For these scenarios, CPU execution may be faster:
- **Scalar operations**: Single value computations
- **Small models**: Networks with few parameters (<100)
- **Low iteration counts**: Very few training iterations

### Overhead

GPU operations have overhead from:
- Data transfer between CPU and GPU
- Kernel launch latency
- GPU context creation

For micrograd_rs with scalar values, this overhead is typically larger than the compute time. However, the architecture is ready for future enhancements like batch operations where GPU excels.

## API Reference

### Device Class

```python
class Device:
    @staticmethod
    def cpu() -> Device:
        """Create a CPU device."""
        
    @staticmethod
    def gpu() -> Device:
        """Create a GPU device. Raises RuntimeError if GPU is not available."""
        
    @staticmethod
    def is_gpu_available() -> bool:
        """Check if GPU is available on this system."""
```

### Value with Device

```python
# Create value on specific device
v = Value(data, device=device)

# Get current device
current_device = v.device

# Move to different device
v_gpu = v.to(Device.gpu())
```

### Neural Networks with Device

```python
from micrograd_rs import MLP, Device

# Create model on GPU
device = Device.gpu() if Device.is_gpu_available() else Device.cpu()
model = MLP(input_size, layer_sizes, device=device)

# All parameters will be on the specified device
for param in model.parameters():
    assert param.device == device
```

## Troubleshooting

### GPU Not Available

If `Device.is_gpu_available()` returns `False`:

1. **Check GPU drivers**: Ensure you have the latest GPU drivers installed
2. **Verify Vulkan support**: Run `vulkaninfo` (Linux/Windows) or check System Information (macOS)
3. **Check permissions**: Ensure your user has access to GPU devices

### GPU Operations Slow

If GPU operations are slower than CPU:

1. **Check model size**: Small models may not benefit from GPU
2. **Measure overhead**: Use the benchmark script to identify bottlenecks
3. **Verify GPU usage**: Use GPU monitoring tools (nvidia-smi, AMD tools, Activity Monitor)

### Build Issues

If you encounter build errors:

1. **Missing GPU support**: Rebuild with GPU feature enabled:
   ```bash
   cargo build --release --features gpu
   ```

2. **wgpu version conflicts**: Check Cargo.lock and update dependencies:
   ```bash
   cargo update
   ```

## Future Enhancements

Planned improvements for GPU support:

1. **Batch Operations**: Support for processing multiple values simultaneously
2. **Tensor Operations**: Full tensor support for matrix/vector operations
3. **Custom Kernels**: Optimized CUDA/Metal kernels for specific operations
4. **Memory Management**: Better GPU memory management and caching
5. **Mixed Precision**: FP16/BF16 support for faster training

## Benchmarking

To benchmark GPU performance on your system:

```bash
python benchmarks/benchmark_gpu.py
```

This will compare CPU and GPU performance for:
- Basic operations (add, mul, pow, relu)
- Neural network forward passes
- Neural network backward passes

Example output:
```
================================================================================
micrograd_rs Performance Benchmarks
================================================================================

GPU Available: True

Basic Operations (average time per operation, 1000 iterations)
--------------------------------------------------------------------------------
Operation                       CPU            GPU
--------------------------------------------------------------------------------
Addition                  0.0006 ms      0.0004 ms
Multiplication            0.0006 ms      0.0004 ms
Power                     0.0004 ms      0.0003 ms
ReLU                      0.0004 ms      0.0003 ms

Neural Network Benchmarks (average time per iteration)
--------------------------------------------------------------------------------

Small MLP (10-20-20-1):
----------------------------------------
  CPU Forward:              0.0494 ms
  CPU Forward+Backward:     0.2778 ms
  GPU Forward:              0.0312 ms
  GPU Forward+Backward:     0.1856 ms
```

## Contributing

If you encounter issues with GPU support or have suggestions for improvements, please:
1. Check existing issues on GitHub
2. Run the benchmark script and include results
3. Provide system information (OS, GPU model, driver version)
4. Submit a detailed issue report

## References

- [wgpu documentation](https://wgpu.rs/)
- [WebGPU specification](https://www.w3.org/TR/webgpu/)
- [Vulkan documentation](https://www.vulkan.org/)
- [Metal documentation](https://developer.apple.com/metal/)
