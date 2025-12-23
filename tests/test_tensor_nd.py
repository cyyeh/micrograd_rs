import math
import pytest

from micrograd_rs import Tensor, Device


def assert_allclose(a, b, rtol=1e-5, atol=1e-6):
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_allclose(x, y, rtol=rtol, atol=atol)
    else:
        assert math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol)


def test_tensor_from_nested_lists_shape_strides():
    t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert t.shape == [2, 3]
    assert t.strides == [3, 1]
    assert t.is_contiguous()
    assert t.tolist() == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_tensor_transpose_and_slice_views():
    t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tt = t.transpose(0, 1)
    assert tt.shape == [3, 2]
    assert tt.tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert not tt.is_contiguous()

    s = tt.slice(0, 0, 3, 2)
    assert s.shape == [2, 2]
    assert s.tolist() == [[1.0, 4.0], [3.0, 6.0]]
    assert not s.is_contiguous()


def test_tensor_reshape_requires_contiguous_or_materializes():
    t = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    tt = t.transpose(0, 1)  # non-contiguous
    r = tt.reshape([6])  # should materialize via .contiguous() internally
    assert r.shape == [6]
    assert r.tolist() == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]


def test_tensor_ops_no_broadcast():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[10.0, 20.0], [30.0, 40.0]])
    c = a + b
    d = a * b
    assert c.tolist() == [[11.0, 22.0], [33.0, 44.0]]
    assert d.tolist() == [[10.0, 40.0], [90.0, 160.0]]

    with pytest.raises(ValueError):
        _ = a + Tensor([1.0, 2.0])


def test_tensor_sum_and_backward_cpu():
    x = Tensor([[1.0, -2.0], [3.0, 4.0]], device=Device.cpu())
    y = (x.relu() * x).sum()
    # y = sum(relu(x) * x) => grad = 2*x where x>0 else 0
    y.backward()
    g = x.grad.tolist()
    assert_allclose(g, [[2.0, 0.0], [6.0, 8.0]])


def test_tensor_pow_backward_cpu():
    x = Tensor([1.5, 2.0, -3.0], device=Device.cpu())
    y = (x ** 3.0).sum()
    y.backward()
    # d/dx x^3 = 3x^2
    g = x.grad.tolist()
    assert_allclose(g, [3.0 * 1.5 * 1.5, 3.0 * 4.0, 3.0 * 9.0])


def test_tensor_cuda_smoke_if_available():
    if not Device.is_cuda_available():
        pytest.skip("CUDA not available")
    dev = Device.cuda()
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], device=dev)
    b = Tensor([[10.0, 20.0], [30.0, 40.0]], device=dev)
    y = ((a + b) * a).sum()
    y.backward()
    # grad of y wrt a: (a+b) + a*1 => 2a + b
    g = a.grad.cpu().tolist()
    assert_allclose(g, [[2.0 * 1.0 + 10.0, 2.0 * 2.0 + 20.0], [2.0 * 3.0 + 30.0, 2.0 * 4.0 + 40.0]])


