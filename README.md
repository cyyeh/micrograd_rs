# micrograd_rs

A tiny autograd engine implemented in Rust with Python bindings via PyO3.

This is a Rust reimplementation of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd) with **GPU acceleration support**.

## Features

- ✨ Fast autograd engine implemented in Rust
- 🚀 GPU acceleration via wgpu (cross-platform: NVIDIA, AMD, Intel, Apple)
- 🐍 Python bindings via PyO3
- 🧠 Neural network layers (Neuron, Layer, MLP)
- 📊 Automatic differentiation (backward propagation)
- 🎯 Simple and intuitive API

## Installation

```bash
# Using Poetry
poetry install
poetry run maturin develop

# Or using pip with maturin
pip install maturin
maturin develop --release
```

### GPU Support

GPU support is enabled by default. The library uses [wgpu](https://wgpu.rs/) for cross-platform GPU acceleration, which supports:
- NVIDIA GPUs (via Vulkan/CUDA)
- AMD GPUs (via Vulkan)
- Intel GPUs (via Vulkan)
- Apple Silicon (via Metal)

To build without GPU support:
```bash
cargo build --release --no-default-features
```

## Usage

### Basic Operations

```python
from micrograd_rs import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}')  # prints 24.7041
g.backward()
print(f'{a.grad:.4f}')  # prints 138.8338
print(f'{b.grad:.4f}')  # prints 645.5773
```

### GPU Acceleration

```python
from micrograd_rs import Value, Device

# Check GPU availability
if Device.is_gpu_available():
    print("GPU is available!")
    device = Device.gpu()
else:
    print("GPU not available, using CPU")
    device = Device.cpu()

# Create values on GPU
a = Value(2.0, device=device)
b = Value(3.0, device=device)
c = a * b + a**2
c.backward()

print(f"Result: {c.data}")
print(f"Gradient of a: {a.grad}")

# Move values between devices
cpu_value = c.to(Device.cpu())
```

### Neural Networks

```python
from micrograd_rs import Value, MLP, Device

# Create a simple MLP (optionally on GPU)
device = Device.gpu() if Device.is_gpu_available() else Device.cpu()
model = MLP(3, [4, 4, 1], device=device)

# Forward pass
x = [Value(1.0, device=device), 
     Value(2.0, device=device), 
     Value(-1.0, device=device)]
y = model(x)

# Backward pass
y.backward()

# Access gradients
for p in model.parameters():
    print(p.data, p.grad)
```

## Running Tests

```bash
# Run all tests
pytest

# Run GPU-specific tests
pytest tests/test_gpu.py

# Run benchmarks
python benchmarks/benchmark_gpu.py
```

## Performance

The GPU acceleration provides significant speedup for larger models and datasets. Run the benchmarks to see the performance improvement on your hardware:

```bash
python benchmarks/benchmark_gpu.py
```

Example output (performance will vary based on hardware):
```
================================================================================
micrograd_rs Performance Benchmarks
================================================================================

GPU Available: True  # Note: Depends on your system configuration

Basic Operations (average time per operation, 1000 iterations)
--------------------------------------------------------------------------------
Operation                       CPU            GPU
--------------------------------------------------------------------------------
Addition                  0.0006 ms      0.0004 ms
Multiplication            0.0006 ms      0.0004 ms
...
```

**Note**: 
- GPU availability depends on your system configuration (hardware, drivers, etc.)
- For scalar operations (single values), GPU overhead may exceed compute time
- GPU acceleration is most effective for large-scale tensor operations and neural networks

## Architecture

- **Device Abstraction**: `Device` enum for CPU/GPU selection
- **GPU Backend**: wgpu-based compute shader execution
- **Value Operations**: All operations (`add`, `mul`, `pow`, `relu`) support device parameter
- **Neural Networks**: Neurons, Layers, and MLPs can be created on specific devices
- **Automatic Fallback**: If GPU is not available, operations gracefully fall back to CPU

## License

MIT

