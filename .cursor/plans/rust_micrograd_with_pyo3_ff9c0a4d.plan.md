---
name: Rust Micrograd with PyO3
overview: Reimplement the micrograd autograd engine and neural network library in Rust, exposing a Python API via PyO3 that matches the original micrograd interface.
todos:
  - id: setup-project
    content: Initialize Cargo project with PyO3, maturin, and configure Cargo.toml/pyproject.toml
    status: pending
  - id: impl-value-core
    content: Implement ValueInner struct with data, grad, graph edges, and backward closures
    status: pending
    dependencies:
      - setup-project
  - id: impl-value-ops
    content: Implement arithmetic ops (+, -, *, /, **) with gradient computation
    status: pending
    dependencies:
      - impl-value-core
  - id: impl-backward
    content: Implement topological sort and backward() method for autograd
    status: pending
    dependencies:
      - impl-value-ops
  - id: pyo3-value
    content: Create PyO3 bindings for Value class with all Python magic methods
    status: pending
    dependencies:
      - impl-backward
  - id: impl-nn
    content: Implement Neuron, Layer, MLP structs with Module trait
    status: pending
    dependencies:
      - pyo3-value
  - id: pyo3-nn
    content: Create PyO3 bindings for neural network classes
    status: pending
    dependencies:
      - impl-nn
  - id: test-validate
    content: Port tests and validate gradients match original micrograd
    status: pending
    dependencies:
      - pyo3-nn
---

# Rust Micrograd Implementation with PyO3 Bindings

## Overview

Reimplement the micrograd autograd engine in Rust with PyO3 bindings. The implementation will provide the same Python API as the original library while leveraging Rust's performance and safety.

## Architecture

```mermaid
flowchart TB
    subgraph python_api [Python API]
        PyValue[Value class]
        PyNeuron[Neuron class]
        PyLayer[Layer class]
        PyMLP[MLP class]
    end
    
    subgraph rust_core [Rust Core]
        RustValue[ValueInner struct]
        RustNode[Computational Graph]
        RustBackward[Backward Pass]
    end
    
    PyValue -->|PyO3 bindings| RustValue
    PyNeuron --> PyValue
    PyLayer --> PyNeuron
    PyMLP --> PyLayer
    RustValue --> RustNode
    RustNode --> RustBackward
```



## Key Implementation Details

### 1. Value Class (Core Autograd Engine)

The `Value` class requires shared ownership for the computational graph since multiple nodes can reference the same parent. In Rust, this will use `Rc<RefCell<ValueInner>>` (or `Arc<RwLock>` for thread safety).**Target API:**

```python
from micrograd_rs import Value
a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
d += 3 * d + (b - a).relu()
g.backward()
print(a.grad, b.grad)
```

**Operations to implement:**

- Arithmetic: `__add__`, `__mul__`, `__pow__`, `__neg__`, `__sub__`, `__truediv__`
- Reverse ops: `__radd__`, `__rmul__`, `__rsub__`, `__rtruediv__`
- Activation: `relu()`
- Autograd: `backward()`
- Properties: `data`, `grad`

### 2. Neural Network Module

Build on top of `Value` to create:

- `Neuron(nin, nonlin=True)` - single neuron
- `Layer(nin, nout, **kwargs)` - layer of neurons
- `MLP(nin, nouts)` - multi-layer perceptron

All inherit from `Module` base with `zero_grad()` and `parameters()`.

## Project Structure

```javascript
micrograd_rs/
├── Cargo.toml
├── pyproject.toml
├── src/
│   ├── lib.rs          # PyO3 module definition
│   ├── value.rs        # Value struct and autograd
│   └── nn.rs           # Neuron, Layer, MLP
└── python/
    └── micrograd_rs/
        └── __init__.py # Re-exports
```



## Implementation Steps

### Phase 1: Project Setup

- Initialize Cargo project with PyO3 and maturin
- Configure `Cargo.toml` with dependencies: `pyo3`, `rand`
- Create `pyproject.toml` for Python packaging

### Phase 2: Value Implementation

- Create `ValueInner` struct with `data`, `grad`, `_prev`, `_op`, and backward closure
- Use `Rc<RefCell<ValueInner>>` for shared ownership in computational graph
- Implement all arithmetic operations with gradient computation
- Implement topological sort for `backward()`

### Phase 3: Python Bindings for Value