# micrograd_rs

A tiny autograd engine implemented in Rust with Python bindings via PyO3.

This is a Rust reimplementation of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd).

## Installation

```bash
# Using Poetry
poetry install
poetry run maturin develop
```

### GPU Support (NVIDIA CUDA)

To enable GPU support for NVIDIA GPUs via CUDA, build with the `cuda` feature:

```bash
# Build with CUDA support
poetry run maturin develop --features cuda
```

**Requirements for CUDA support:**
- NVIDIA GPU with CUDA capability
- CUDA Toolkit installed (11.x or 12.x)
- `nvcc` available in PATH

## Usage

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

### Device Selection

You can specify which device (CPU or GPU) to use for computations:

```python
from micrograd_rs import Value, Device

# Check if CUDA is available
if Device.is_cuda_available():
    device = Device.cuda()
    print("Using CUDA GPU")
else:
    device = Device.cpu()
    print("Using CPU")

# Create values on a specific device
a = Value(2.0, device=device)
b = Value(3.0, device=device)

# Operations preserve device placement
c = a + b  # c is on the same device as a and b
print(f"Result: {c.data}, Device: {c.device}")

# Move values between devices
cpu_val = c.cpu()  # Move to CPU
# cuda_val = c.cuda()  # Move to CUDA (if available)

# Use the .to() method for explicit device transfer
cpu_device = Device.cpu()
d = c.to(cpu_device)
```

## Neural Networks

```python
from micrograd_rs import Value, MLP

# Create a simple MLP
model = MLP(3, [4, 4, 1])

# Forward pass
x = [Value(1.0), Value(2.0), Value(-1.0)]
y = model(x)

# Backward pass
y.backward()

# Access gradients
for p in model.parameters():
    print(p.data, p.grad)
```

## Running Tests

```bash
poetry run pytest
```

## Benchmarks

To compare CPU vs GPU performance:

```bash
poetry run python benchmarks/benchmark_device.py
```

The benchmark script measures various operations including:
- Basic arithmetic (add, mul, pow, relu)
- Chained operations
- Backward pass
- Complex expressions

When CUDA is available, it compares CPU and GPU performance and shows speedup ratios.


