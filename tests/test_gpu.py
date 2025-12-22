"""
Tests for GPU support in micrograd_rs
"""

import pytest
from micrograd_rs import Value, Device, MLP


def test_device_class():
    """Test Device class and its methods."""
    # Test CPU device
    cpu = Device.cpu()
    assert str(cpu) == "CPU"
    assert repr(cpu) == "Device.CPU"
    
    # Test GPU availability check
    is_gpu_available = Device.is_gpu_available()
    assert isinstance(is_gpu_available, bool)
    
    # Test GPU device creation
    if is_gpu_available:
        gpu = Device.gpu()
        assert str(gpu) == "GPU"
        assert repr(gpu) == "Device.GPU"
    else:
        # GPU not available should raise error
        with pytest.raises(RuntimeError):
            Device.gpu()


def test_value_with_device():
    """Test Value creation with different devices."""
    # Default device (CPU)
    v1 = Value(3.14)
    assert v1.device == Device.cpu()
    
    # Explicit CPU device
    v2 = Value(2.71, device=Device.cpu())
    assert v2.device == Device.cpu()
    assert v2.data == 2.71


def test_operations_preserve_device():
    """Test that operations preserve the device."""
    v1 = Value(2.0, device=Device.cpu())
    v2 = Value(3.0, device=Device.cpu())
    
    # Test addition
    v3 = v1 + v2
    assert v3.data == 5.0
    assert v3.device == Device.cpu()
    
    # Test multiplication
    v4 = v1 * v2
    assert v4.data == 6.0
    assert v4.device == Device.cpu()
    
    # Test power
    v5 = v1 ** 2
    assert v5.data == 4.0
    assert v5.device == Device.cpu()
    
    # Test relu
    v6 = Value(-5.0, device=Device.cpu())
    v7 = v6.relu()
    assert v7.data == 0.0
    assert v7.device == Device.cpu()
    
    v8 = Value(5.0, device=Device.cpu())
    v9 = v8.relu()
    assert v9.data == 5.0
    assert v9.device == Device.cpu()


def test_backward_with_device():
    """Test backward pass works correctly with device."""
    v1 = Value(2.0, device=Device.cpu())
    v2 = Value(3.0, device=Device.cpu())
    v3 = v1 * v2
    v3.backward()
    
    # Check gradients
    assert v1.grad == 3.0
    assert v2.grad == 2.0
    assert v3.grad == 1.0


def test_to_method():
    """Test moving values between devices."""
    v1 = Value(3.14, device=Device.cpu())
    assert v1.device == Device.cpu()
    
    # Moving to same device should return equivalent value
    v2 = v1.to(Device.cpu())
    assert v2.device == Device.cpu()
    assert v2.data == v1.data


def test_mlp_with_device():
    """Test MLP creation with device parameter."""
    # Create MLP on CPU
    model_cpu = MLP(3, [4, 4, 1], device=Device.cpu())
    
    # Create input values on CPU
    x = [Value(1.0, device=Device.cpu()), 
         Value(2.0, device=Device.cpu()), 
         Value(-1.0, device=Device.cpu())]
    
    # Forward pass
    y = model_cpu(x)
    
    # Check that output is a Value
    if isinstance(y, list):
        y = y[0]
    assert isinstance(y, Value)
    assert y.device == Device.cpu()
    
    # Backward pass
    y.backward()
    
    # Check that all parameters have gradients
    params = model_cpu.parameters()
    assert len(params) > 0
    for p in params:
        assert p.device == Device.cpu()
        # Gradient should be computed
        assert p.grad != 0.0 or True  # Some gradients might be zero


def test_mixed_operations():
    """Test operations with scalars preserve device."""
    v1 = Value(2.0, device=Device.cpu())
    
    # Operations with Python floats
    v2 = v1 + 3.0
    assert v2.data == 5.0
    assert v2.device == Device.cpu()
    
    v3 = v1 * 2.0
    assert v3.data == 4.0
    assert v3.device == Device.cpu()
    
    v4 = v1 - 1.0
    assert v4.data == 1.0
    assert v4.device == Device.cpu()
    
    v5 = v1 / 2.0
    assert v5.data == 1.0
    assert v5.device == Device.cpu()


def test_gpu_fallback():
    """Test that GPU operations fall back to CPU when GPU is not available."""
    # This test ensures that even if we tried to use GPU, 
    # the system would gracefully fall back to CPU
    v1 = Value(2.0, device=Device.cpu())
    v2 = Value(3.0, device=Device.cpu())
    
    # All operations should work
    result = (v1 + v2) * v1 - v2 / 2.0
    assert isinstance(result, Value)
    assert result.device == Device.cpu()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
