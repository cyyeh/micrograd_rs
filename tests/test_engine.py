"""
Tests for micrograd_rs - ported from the original micrograd test suite.
These tests verify that the Rust implementation produces the same results as PyTorch.
"""

import torch
from micrograd_rs import Value


def test_sanity_check():
    """Basic sanity check comparing micrograd_rs with PyTorch."""
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item(), f"Forward: {ymg.data} != {ypt.data.item()}"
    # backward pass went well
    assert xmg.grad == xpt.grad.item(), f"Backward: {xmg.grad} != {xpt.grad.item()}"


def test_more_ops():
    """Test more operations including power, division, and compound expressions."""
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
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol, f"Forward: {gmg.data} != {gpt.data.item()}"
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol, f"Backward a: {amg.grad} != {apt.grad.item()}"
    assert abs(bmg.grad - bpt.grad.item()) < tol, f"Backward b: {bmg.grad} != {bpt.grad.item()}"


def test_value_basic():
    """Test basic Value operations without PyTorch comparison."""
    a = Value(2.0)
    b = Value(3.0)
    
    # Test addition
    c = a + b
    assert c.data == 5.0
    
    # Test multiplication
    d = a * b
    assert d.data == 6.0
    
    # Test power
    e = a ** 2
    assert e.data == 4.0
    
    # Test negation
    f = -a
    assert f.data == -2.0
    
    # Test subtraction
    g = b - a
    assert g.data == 1.0
    
    # Test division
    h = b / a
    assert abs(h.data - 1.5) < 1e-10
    
    # Test relu
    i = Value(-5.0)
    assert i.relu().data == 0.0
    assert a.relu().data == 2.0


def test_backward_simple():
    """Test simple backward pass."""
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    
    # dc/da = b = 3.0
    assert a.grad == 3.0
    # dc/db = a = 2.0
    assert b.grad == 2.0


def test_backward_chain():
    """Test backward pass through a chain of operations."""
    a = Value(2.0)
    b = a * a  # b = a^2, db/da = 2a = 4
    c = b * a  # c = a^3, dc/da = 3a^2 = 12
    c.backward()
    
    # dc/da through chain rule
    assert a.grad == 12.0


def test_repr():
    """Test string representation."""
    a = Value(3.14)
    repr_str = repr(a)
    assert "3.14" in repr_str
    assert "grad=" in repr_str


if __name__ == "__main__":
    test_value_basic()
    print("test_value_basic passed!")
    
    test_backward_simple()
    print("test_backward_simple passed!")
    
    test_backward_chain()
    print("test_backward_chain passed!")
    
    test_repr()
    print("test_repr passed!")
    
    test_sanity_check()
    print("test_sanity_check passed!")
    
    test_more_ops()
    print("test_more_ops passed!")
    
    print("\nAll tests passed!")

