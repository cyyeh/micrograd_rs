# micrograd_rs

A tiny autograd engine implemented in Rust with Python bindings via PyO3.

This is a Rust reimplementation of [Andrej Karpathy's micrograd](https://github.com/karpathy/micrograd).

## Installation

```bash
# Using Poetry
poetry install
poetry run maturin develop
```

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

