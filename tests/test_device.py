"""
Tests for device support in micrograd_rs.
These tests verify the device selection API and CPU/GPU functionality.
"""

from micrograd_rs import Value, Device


def test_device_cpu_creation():
    """Test creating a CPU device."""
    device = Device.cpu()
    assert device.is_cpu()
    assert not device.is_cuda()
    assert str(device) == "cpu"


def test_device_is_cuda_available():
    """Test CUDA availability check."""
    # This should return False if CUDA is not available
    available = Device.is_cuda_available()
    assert isinstance(available, bool)


def test_value_default_device():
    """Test that Value defaults to CPU device."""
    a = Value(3.14)
    assert a.device.is_cpu()


def test_value_with_cpu_device():
    """Test creating a Value with explicit CPU device."""
    device = Device.cpu()
    a = Value(2.0, device=device)
    assert a.device.is_cpu()
    assert a.data == 2.0


def test_value_to_cpu():
    """Test moving a Value to CPU."""
    a = Value(5.0)
    b = a.cpu()
    assert b.device.is_cpu()
    assert b.data == 5.0


def test_value_operations_preserve_device():
    """Test that operations preserve device placement."""
    device = Device.cpu()
    a = Value(2.0, device=device)
    b = Value(3.0, device=device)
    
    # Test addition
    c = a + b
    assert c.device.is_cpu()
    assert c.data == 5.0
    
    # Test multiplication
    d = a * b
    assert d.device.is_cpu()
    assert d.data == 6.0
    
    # Test power
    e = a ** 2
    assert e.device.is_cpu()
    assert e.data == 4.0
    
    # Test relu
    f = Value(-5.0, device=device)
    g = f.relu()
    assert g.device.is_cpu()
    assert g.data == 0.0


def test_value_backward_on_device():
    """Test backward pass works correctly on CPU device."""
    device = Device.cpu()
    a = Value(2.0, device=device)
    b = Value(3.0, device=device)
    c = a * b
    c.backward()
    
    # dc/da = b = 3.0
    assert a.grad == 3.0
    # dc/db = a = 2.0
    assert b.grad == 2.0


def test_value_repr_includes_device():
    """Test that Value repr includes device information."""
    a = Value(3.14)
    repr_str = repr(a)
    assert "3.14" in repr_str
    assert "cpu" in repr_str


def test_value_to_device():
    """Test moving a Value to a specific device."""
    device = Device.cpu()
    a = Value(5.0)
    b = a.to(device)
    assert b.device.is_cpu()
    assert b.data == 5.0


def test_cuda_unavailable_error():
    """Test that cuda() returns error when CUDA is not available."""
    a = Value(5.0)
    try:
        # This should raise an error since CUDA is not available in test environment
        b = a.cuda()
        # If we get here, CUDA is available, which is also fine
        assert b.device.is_cuda()
    except RuntimeError as e:
        # Expected error when CUDA is not available
        assert "CUDA" in str(e)


def test_device_cuda_unavailable():
    """Test Device.cuda() returns error when CUDA is not available."""
    try:
        device = Device.cuda()
        # If we get here, CUDA is available
        assert device.is_cuda()
    except RuntimeError as e:
        # Expected error when CUDA is not available
        assert "CUDA" in str(e)


if __name__ == "__main__":
    test_device_cpu_creation()
    print("test_device_cpu_creation passed!")
    
    test_device_is_cuda_available()
    print("test_device_is_cuda_available passed!")
    
    test_value_default_device()
    print("test_value_default_device passed!")
    
    test_value_with_cpu_device()
    print("test_value_with_cpu_device passed!")
    
    test_value_to_cpu()
    print("test_value_to_cpu passed!")
    
    test_value_operations_preserve_device()
    print("test_value_operations_preserve_device passed!")
    
    test_value_backward_on_device()
    print("test_value_backward_on_device passed!")
    
    test_value_repr_includes_device()
    print("test_value_repr_includes_device passed!")
    
    test_value_to_device()
    print("test_value_to_device passed!")
    
    test_cuda_unavailable_error()
    print("test_cuda_unavailable_error passed!")
    
    test_device_cuda_unavailable()
    print("test_device_cuda_unavailable passed!")
    
    print("\nAll device tests passed!")
