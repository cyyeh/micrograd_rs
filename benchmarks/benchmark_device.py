"""
Benchmarks comparing CPU vs GPU performance for micrograd_rs.
These benchmarks measure the performance of various operations on different devices.

Run with: python benchmarks/benchmark_device.py
"""

import time
from typing import Callable
from micrograd_rs import Value, Device


def benchmark_operation(
    name: str,
    operation: Callable[[], None],
    iterations: int = 1000,
    warmup: int = 10
) -> float:
    """
    Benchmark an operation and return the average time in milliseconds.
    
    Args:
        name: Name of the operation being benchmarked
        operation: Function to benchmark
        iterations: Number of iterations to run
        warmup: Number of warmup iterations
    
    Returns:
        Average time per iteration in milliseconds
    """
    # Warmup
    for _ in range(warmup):
        operation()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        operation()
    end = time.perf_counter()
    
    avg_time_ms = ((end - start) / iterations) * 1000
    return avg_time_ms


def benchmark_basic_operations(device: Device, device_name: str) -> dict:
    """Benchmark basic arithmetic operations."""
    results = {}
    
    # Addition
    def add_op():
        a = Value(2.0, device=device)
        b = Value(3.0, device=device)
        c = a + b
        return c.data
    
    results['add'] = benchmark_operation(f'{device_name} add', add_op)
    
    # Multiplication
    def mul_op():
        a = Value(2.0, device=device)
        b = Value(3.0, device=device)
        c = a * b
        return c.data
    
    results['mul'] = benchmark_operation(f'{device_name} mul', mul_op)
    
    # Power
    def pow_op():
        a = Value(2.0, device=device)
        b = a ** 3
        return b.data
    
    results['pow'] = benchmark_operation(f'{device_name} pow', pow_op)
    
    # ReLU
    def relu_op():
        a = Value(-2.0, device=device)
        b = a.relu()
        return b.data
    
    results['relu'] = benchmark_operation(f'{device_name} relu', relu_op)
    
    return results


def benchmark_chain_operations(device: Device, device_name: str, chain_length: int = 10) -> float:
    """Benchmark chained operations."""
    def chain_op():
        x = Value(1.0, device=device)
        for i in range(chain_length):
            x = x * Value(1.1, device=device)
            x = x + Value(0.1, device=device)
        return x.data
    
    return benchmark_operation(f'{device_name} chain({chain_length})', chain_op, iterations=100)


def benchmark_backward_pass(device: Device, device_name: str) -> float:
    """Benchmark backward pass."""
    def backward_op():
        a = Value(2.0, device=device)
        b = Value(3.0, device=device)
        c = a * b + b ** 2
        d = c.relu()
        e = d * Value(2.0, device=device)
        e.backward()
        return a.grad
    
    return benchmark_operation(f'{device_name} backward', backward_op, iterations=100)


def benchmark_complex_expression(device: Device, device_name: str) -> float:
    """Benchmark complex expression (from micrograd's test suite)."""
    def complex_op():
        a = Value(-4.0, device=device)
        b = Value(2.0, device=device)
        c = a + b
        d = a * b + b ** 3
        c = c + c + Value(1.0, device=device)
        c = c + Value(1.0, device=device) + c + (-a)
        d = d + d * Value(2.0, device=device) + (b + a).relu()
        d = d + Value(3.0, device=device) * d + (b - a).relu()
        e = c - d
        f = e ** 2
        g = f / Value(2.0, device=device)
        g = g + Value(10.0, device=device) / f
        g.backward()
        return g.data
    
    return benchmark_operation(f'{device_name} complex', complex_op, iterations=100)


def print_results_table(cpu_results: dict, cuda_results: dict = None):
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    if cuda_results:
        print(f"{'Operation':<25} {'CPU (ms)':<15} {'CUDA (ms)':<15} {'Speedup':<15}")
        print("-" * 70)
        for op in cpu_results:
            cpu_time = cpu_results[op]
            cuda_time = cuda_results.get(op)
            if cuda_time is not None:
                speedup = cpu_time / cuda_time if cuda_time > 0 else float('inf')
                print(f"{op:<25} {cpu_time:<15.4f} {cuda_time:<15.4f} {speedup:<15.2f}x")
            else:
                print(f"{op:<25} {cpu_time:<15.4f} {'N/A':<15} {'N/A':<15}")
    else:
        print(f"{'Operation':<25} {'CPU (ms)':<15}")
        print("-" * 40)
        for op in cpu_results:
            print(f"{op:<25} {cpu_results[op]:<15.4f}")
    
    print("=" * 70)


def main():
    print("=" * 70)
    print("micrograd_rs Device Performance Benchmarks")
    print("=" * 70)
    
    # Check CUDA availability
    cuda_available = Device.is_cuda_available()
    print(f"\nCUDA Available: {cuda_available}")
    
    # CPU benchmarks
    print("\n--- Running CPU Benchmarks ---")
    cpu_device = Device.cpu()
    
    cpu_results = benchmark_basic_operations(cpu_device, "CPU")
    cpu_results['chain_10'] = benchmark_chain_operations(cpu_device, "CPU", 10)
    cpu_results['backward'] = benchmark_backward_pass(cpu_device, "CPU")
    cpu_results['complex'] = benchmark_complex_expression(cpu_device, "CPU")
    
    # CUDA benchmarks (if available)
    cuda_results = None
    if cuda_available:
        print("\n--- Running CUDA Benchmarks ---")
        try:
            cuda_device = Device.cuda()
            
            cuda_results = benchmark_basic_operations(cuda_device, "CUDA")
            cuda_results['chain_10'] = benchmark_chain_operations(cuda_device, "CUDA", 10)
            cuda_results['backward'] = benchmark_backward_pass(cuda_device, "CUDA")
            cuda_results['complex'] = benchmark_complex_expression(cuda_device, "CUDA")
        except RuntimeError as e:
            print(f"Failed to create CUDA device: {e}")
            cuda_results = None
    else:
        print("\n--- Skipping CUDA Benchmarks (CUDA not available) ---")
    
    # Print results
    print_results_table(cpu_results, cuda_results)
    
    # Summary
    print("\nNotes:")
    print("- Times are in milliseconds (ms)")
    print("- Speedup > 1.0 means CUDA is faster than CPU")
    print("- For scalar operations, CPU and CUDA have similar performance")
    print("- True GPU acceleration benefits appear with batched tensor operations")
    if not cuda_available:
        print("- CUDA benchmarks skipped (no CUDA device available)")


if __name__ == "__main__":
    main()
