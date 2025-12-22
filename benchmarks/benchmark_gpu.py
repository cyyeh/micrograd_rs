"""
Benchmarks for comparing CPU and GPU performance in micrograd_rs

This script benchmarks the performance of basic operations and neural network
forward/backward passes on CPU vs GPU (when available).
"""

import time
from micrograd_rs import Value, MLP, Device


def benchmark_operation(name, operation_fn, device, iterations=1000):
    """Benchmark a single operation."""
    start = time.time()
    for _ in range(iterations):
        operation_fn(device)
    elapsed = time.time() - start
    avg_time = elapsed / iterations * 1000  # Convert to milliseconds
    return avg_time


def test_addition(device):
    """Test addition operation."""
    v1 = Value(2.0, device=device)
    v2 = Value(3.0, device=device)
    result = v1 + v2
    return result


def test_multiplication(device):
    """Test multiplication operation."""
    v1 = Value(2.0, device=device)
    v2 = Value(3.0, device=device)
    result = v1 * v2
    return result


def test_power(device):
    """Test power operation."""
    v = Value(2.0, device=device)
    result = v ** 2
    return result


def test_relu(device):
    """Test ReLU operation."""
    v = Value(-5.0, device=device)
    result = v.relu()
    return result


def test_mlp_forward(device, input_size=10, hidden_sizes=[20, 20], iterations=100):
    """Test MLP forward pass."""
    model = MLP(input_size, hidden_sizes + [1], device=device)
    
    start = time.time()
    for _ in range(iterations):
        x = [Value(float(i), device=device) for i in range(input_size)]
        y = model(x)
    elapsed = time.time() - start
    
    return elapsed / iterations * 1000  # Convert to milliseconds


def test_mlp_backward(device, input_size=10, hidden_sizes=[20, 20], iterations=100):
    """Test MLP forward and backward pass."""
    model = MLP(input_size, hidden_sizes + [1], device=device)
    
    start = time.time()
    for _ in range(iterations):
        # Zero gradients
        for p in model.parameters():
            p.grad = 0.0
        
        # Forward pass
        x = [Value(float(i), device=device) for i in range(input_size)]
        y = model(x)
        
        # Backward pass
        if isinstance(y, list):
            y = y[0]
        y.backward()
    
    elapsed = time.time() - start
    return elapsed / iterations * 1000  # Convert to milliseconds


def run_benchmarks():
    """Run all benchmarks and display results."""
    print("=" * 80)
    print("micrograd_rs Performance Benchmarks")
    print("=" * 80)
    print()
    
    # Check GPU availability
    gpu_available = Device.is_gpu_available()
    print(f"GPU Available: {gpu_available}")
    print()
    
    devices = [Device.cpu()]
    device_names = ["CPU"]
    
    if gpu_available:
        devices.append(Device.gpu())
        device_names.append("GPU")
    
    # Basic operations benchmarks
    print("Basic Operations (average time per operation, 1000 iterations)")
    print("-" * 80)
    
    operations = [
        ("Addition", test_addition),
        ("Multiplication", test_multiplication),
        ("Power", test_power),
        ("ReLU", test_relu),
    ]
    
    results = {}
    for op_name, op_fn in operations:
        results[op_name] = {}
        for device, dev_name in zip(devices, device_names):
            avg_time = benchmark_operation(op_name, op_fn, device)
            results[op_name][dev_name] = avg_time
    
    # Display basic operations results
    header = f"{'Operation':<20}" + "".join(f"{name:>15}" for name in device_names)
    print(header)
    print("-" * 80)
    
    for op_name in results:
        row = f"{op_name:<20}"
        for dev_name in device_names:
            time_ms = results[op_name][dev_name]
            row += f"{time_ms:>12.4f} ms"
        print(row)
    
    print()
    
    # Neural network benchmarks
    print("Neural Network Benchmarks (average time per iteration)")
    print("-" * 80)
    
    nn_configs = [
        ("Small MLP (10-20-20-1)", 10, [20, 20], 100),
        ("Medium MLP (50-100-100-1)", 50, [100, 100], 50),
    ]
    
    nn_results = {}
    for config_name, input_size, hidden_sizes, iterations in nn_configs:
        nn_results[config_name] = {}
        
        for device, dev_name in zip(devices, device_names):
            # Forward pass only
            forward_time = test_mlp_forward(device, input_size, hidden_sizes, iterations)
            nn_results[config_name][f"{dev_name}_forward"] = forward_time
            
            # Forward + backward pass
            backward_time = test_mlp_backward(device, input_size, hidden_sizes, iterations)
            nn_results[config_name][f"{dev_name}_backward"] = backward_time
    
    # Display neural network results
    for config_name in nn_results:
        print(f"\n{config_name}:")
        print("-" * 40)
        for dev_name in device_names:
            forward_key = f"{dev_name}_forward"
            backward_key = f"{dev_name}_backward"
            forward_time = nn_results[config_name][forward_key]
            backward_time = nn_results[config_name][backward_key]
            print(f"  {dev_name} Forward:          {forward_time:>10.4f} ms")
            print(f"  {dev_name} Forward+Backward: {backward_time:>10.4f} ms")
    
    print()
    print("=" * 80)
    
    # Performance comparison
    if len(devices) > 1:
        print("\nPerformance Comparison (CPU baseline)")
        print("-" * 80)
        
        for op_name in results:
            cpu_time = results[op_name]["CPU"]
            gpu_time = results[op_name]["GPU"]
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"{op_name:<20} Speedup: {speedup:.2f}x")
        
        print()
        for config_name in nn_results:
            print(f"\n{config_name}:")
            for phase in ["forward", "backward"]:
                cpu_key = f"CPU_{phase}"
                gpu_key = f"GPU_{phase}"
                cpu_time = nn_results[config_name][cpu_key]
                gpu_time = nn_results[config_name][gpu_key]
                speedup = cpu_time / gpu_time if gpu_time > 0 else 0
                print(f"  {phase.capitalize():>15}: {speedup:.2f}x speedup")
        
        print()
        print("=" * 80)
    
    print("\nNote: For scalar operations, GPU overhead may exceed compute time.")
    print("GPU acceleration is most effective for large-scale tensor operations.")


if __name__ == "__main__":
    run_benchmarks()
