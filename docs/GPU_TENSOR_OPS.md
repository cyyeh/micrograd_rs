---
question: I am new to GPU and tensor operations, please guide me through the project here
---

I'd be happy to guide you through this project! Let me first explore the codebase to understand its structure and then explain the key concepts.

[4 tools called]

Now let me look at the GPU and tensor-specific code to explain those concepts:

[4 tools called]

Now I have a complete picture! Let me guide you through this project focusing on **GPU and tensor operations**.

---

# 🚀 Understanding GPU & Tensor Operations in micrograd_rs

This project is a Rust implementation of **micrograd** (Andrej Karpathy's autograd engine) with Python bindings. Let me break down the key concepts:

---

## 📊 What is This Project?

This is an **automatic differentiation (autograd) engine** that:
1. Computes mathematical operations (add, multiply, power, ReLU)
2. Automatically tracks gradients for backpropagation (the backbone of neural network training)
3. Can run on **CPU** or **GPU (NVIDIA CUDA)**

---

## 🧠 Key Concept 1: Tensors

A **Tensor** is a multi-dimensional array of numbers. Think of it as a generalization of:
- Scalar (0D): a single number like `5.0`
- Vector (1D): `[1, 2, 3]`
- Matrix (2D): `[[1, 2], [3, 4]]`
- Higher dimensions (3D+): video frames, batches of images, etc.

### Tensor Structure in the Codebase

```23:29:/home/cyyeh/repos/rust-py-bindings/micrograd_rs/src/tensor.rs
/// A tensor view into a flat storage buffer.
///
/// - `shape`: lengths per dimension
/// - `strides`: stride (in elements) per dimension
/// - `offset`: starting offset (in elements) into the underlying storage
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorView {
    pub shape: Vec<usize>,
    pub strides: Vec<isize>,
    pub offset: usize,
    pub numel: usize,
}
```

| Field | Meaning | Example |
|-------|---------|---------|
| `shape` | Dimensions of the tensor | `[2, 3]` = 2 rows, 3 columns |
| `strides` | How many elements to skip per dimension | `[3, 1]` = skip 3 for next row, skip 1 for next column |
| `offset` | Starting position in memory | `0` for new tensors |
| `numel` | Total number of elements | `6` for a 2×3 matrix |

### Why Strides Matter

Strides enable **views without copying data**. For example, transposing a matrix just swaps the strides!

```python
# In Python
t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape [2, 3], strides [3, 1]
t.transpose(0, 1)  # shape [3, 2], strides [1, 3] - NO data copy!
```

---

## 🖥️ Key Concept 2: CPU vs GPU Execution

### The Device Abstraction

```9:17:/home/cyyeh/repos/rust-py-bindings/micrograd_rs/src/device.rs
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
```

### Why Use GPU?

| CPU | GPU |
|-----|-----|
| Few powerful cores (4-16) | Thousands of small cores (1000s) |
| Great for sequential tasks | Great for **parallel** tasks |
| Low latency for small data | High throughput for large data |

**GPUs excel at tensor operations** because:
- Matrix multiplication: multiply thousands of elements simultaneously
- Elementwise ops: apply ReLU to millions of values at once

---

## ⚡ Key Concept 3: How Data Moves Between CPU & GPU

```168:173:/home/cyyeh/repos/rust-py-bindings/micrograd_rs/src/tensor.rs
#[derive(Clone, Debug)]
pub enum TensorData {
    Cpu(Arc<Vec<f32>>),
    #[cfg(feature = "cuda")]
    Cuda(Arc<CudaSlice<f32>>),
}
```

| Location | Storage Type | Description |
|----------|--------------|-------------|
| CPU | `Vec<f32>` | Standard Rust vector in RAM |
| GPU | `CudaSlice<f32>` | Memory allocated on GPU (VRAM) |

### Data Transfer Methods

```python
# Move to GPU
gpu_tensor = tensor.cuda()

# Move back to CPU  
cpu_tensor = gpu_tensor.cpu()

# Or use .to() for explicit control
tensor.to(Device.cuda())
```

**⚠️ Important:** Data transfers between CPU↔GPU are *expensive*! The project avoids implicit transfers:

```496:505:/home/cyyeh/repos/rust-py-bindings/micrograd_rs/src/tensor.rs
    fn ensure_same_device(&self, other: &Tensor) -> PyResult<DeviceType> {
        let a = self.device_type();
        let b = other.device_type();
        if a != b {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "tensors must be on the same device (no implicit device transfers)",
            ));
        }
        Ok(a)
    }
```

---

## 🔧 Key Concept 4: Kernels (The GPU Programs)

### What's a Kernel?

A **kernel** is a small program that runs on the GPU. Each GPU core executes the same code on different data simultaneously.

### CPU Kernels (`cpu_kernels.rs`)

The CPU version iterates through elements:

```52:77:/home/cyyeh/repos/rust-py-bindings/micrograd_rs/src/cpu_kernels.rs
/// Elementwise binary op on two same-shaped tensors (no broadcasting), producing a contiguous output.
pub fn cpu_ew_binary(
    out: &mut [f32],
    a_buf: &Arc<Vec<f32>>,
    a_view: &TensorView,
    b_buf: &Arc<Vec<f32>>,
    b_view: &TensorView,
    op: impl Fn(f32, f32) -> f32,
) -> PyResult<()> {
    // ... iterates through all elements with a loop
}
```

### CUDA Kernels (`cuda_kernels.rs`)

The GPU version runs **in parallel** across thousands of threads:

```14:21:/home/cyyeh/repos/rust-py-bindings/micrograd_rs/src/cuda_kernels.rs
const KERNEL_SRC: &str = r#"
extern "C" __global__ void ew_add_f32(float* out, const float* a, const float* b, unsigned int n) {
    unsigned int idx = (unsigned int)(blockIdx.x * blockDim.x + threadIdx.x);
    unsigned int stride = (unsigned int)(blockDim.x * gridDim.x);
    for (unsigned int i = idx; i < n; i += stride) {
        out[i] = a[i] + b[i];
    }
}
```

### Understanding CUDA Terminology

```
Grid (all blocks)
├── Block 0
│   ├── Thread 0  ──→ processes element 0
│   ├── Thread 1  ──→ processes element 1
│   └── ...
├── Block 1
│   ├── Thread 0  ──→ processes element 256
│   └── ...
└── ...
```

| Term | Meaning |
|------|---------|
| `threadIdx.x` | Thread ID within a block (0-255 typically) |
| `blockIdx.x` | Block ID |
| `blockDim.x` | Number of threads per block |
| `gridDim.x` | Number of blocks |

Each thread computes: `idx = blockIdx * blockDim + threadIdx` to get its unique global index.

---

## 🔄 Key Concept 5: Autograd (Automatic Differentiation)

### The Computation Graph

```182:190:/home/cyyeh/repos/rust-py-bindings/micrograd_rs/src/tensor.rs
#[derive(Clone, Debug)]
pub enum TensorOp {
    None,
    Add(Rc<RefCell<TensorInner>>, Rc<RefCell<TensorInner>>),
    Mul(Rc<RefCell<TensorInner>>, Rc<RefCell<TensorInner>>),
    Pow(Rc<RefCell<TensorInner>>, f32),
    ReLU(Rc<RefCell<TensorInner>>),
    Sum(Rc<RefCell<TensorInner>>),
}
```

Every operation stores **what created it** (its parents). This builds a **computation graph**:

```
a ──┐
    ├──→ (a + b) = c ──┐
b ──┘                   ├──→ (c * d) = e
d ─────────────────────┘
```

### Backward Pass

When you call `.backward()`:
1. Start from the output (gradient = 1.0)
2. Walk backwards through the graph
3. Apply the **chain rule** to compute gradients

For example, with `add`:
- Forward: `c = a + b`
- Backward: `grad_a += grad_c`, `grad_b += grad_c`

For `mul`:
- Forward: `c = a * b`
- Backward: `grad_a += b * grad_c`, `grad_b += a * grad_c`

---

## 📂 Project Architecture Overview

```
src/
├── lib.rs           # Module declarations & Python exports
├── device.rs        # CPU/GPU device abstraction + CUDA context
├── value.rs         # Scalar values with autograd (original micrograd)
├── tensor.rs        # N-dimensional tensors with autograd
├── cpu_kernels.rs   # CPU implementations of tensor ops
├── cuda_kernels.rs  # CUDA GPU kernels (compiled with NVRTC)
└── nn.rs            # Neural network layers (Neuron, Layer, MLP)
```

---

## 🎓 Quick Example Flow

```python
from micrograd_rs import Tensor, Device

# 1. Create tensor on GPU
x = Tensor([[1.0, -2.0], [3.0, 4.0]], device=Device.cuda())

# 2. Forward pass (runs CUDA kernels)
y = (x.relu() * x).sum()

# 3. Backward pass (computes gradients on GPU)
y.backward()

# 4. Get gradients (transfers back to CPU for Python access)
print(x.grad.tolist())  # [[2.0, 0.0], [6.0, 8.0]]
```

---

## 🔑 Key Takeaways

1. **Tensors** = multi-dimensional arrays with shape/strides for efficient views
2. **Device abstraction** lets code run on CPU or GPU transparently
3. **GPU kernels** are small programs that run in parallel across thousands of threads
4. **Data lives where you put it** - explicit control, no automatic transfers
5. **Autograd** tracks operations to automatically compute gradients for training

Would you like me to dive deeper into any specific area? I can explain:
- How the backward pass works in detail
- CUDA memory management
- How `cudarc` (the Rust CUDA library) works
- The neural network module (`nn.rs`)