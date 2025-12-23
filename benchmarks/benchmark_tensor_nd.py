"""
Benchmarks for ND Tensor CPU vs CUDA.

Run with:
  poetry run python benchmarks/benchmark_tensor_nd.py
"""

import time
from micrograd_rs import Tensor, Device


def bench(name, fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0 / iters
    print(f"{name:<25} {ms:>10.3f} ms")
    return ms


def main():
    n = 2_000_000
    shape = [n]

    cpu = Device.cpu()
    a_cpu = Tensor([1.0] * n, shape=shape, device=cpu)
    b_cpu = Tensor([2.0] * n, shape=shape, device=cpu)

    print("\n--- CPU ---")
    bench("add", lambda: (a_cpu + b_cpu))
    bench("mul", lambda: (a_cpu * b_cpu))
    bench("relu", lambda: a_cpu.relu())
    bench("pow", lambda: (a_cpu ** 2.5))
    bench("sum", lambda: a_cpu.sum())
    bench("backward", lambda: ((a_cpu * b_cpu).sum().backward()))

    if not Device.is_cuda_available():
        print("\nCUDA not available; skipping GPU benchmarks.")
        return

    dev = Device.cuda()
    a = Tensor([1.0] * n, shape=shape, device=dev)
    b = Tensor([2.0] * n, shape=shape, device=dev)

    print("\n--- CUDA ---")
    bench("add", lambda: (a + b))
    bench("mul", lambda: (a * b))
    bench("relu", lambda: a.relu())
    bench("pow", lambda: (a ** 2.5))
    bench("sum", lambda: a.sum())
    bench("backward", lambda: ((a * b).sum().backward()))


if __name__ == "__main__":
    main()


