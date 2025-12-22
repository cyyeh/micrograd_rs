#!/usr/bin/env python3
"""
Example demonstrating GPU support in micrograd_rs

This script shows how to:
1. Check GPU availability
2. Create values on GPU
3. Perform operations on GPU
4. Train a neural network on GPU
"""

from micrograd_rs import Value, MLP, Device


def demo_basic_operations():
    """Demonstrate basic operations with GPU support."""
    print("=" * 80)
    print("Basic Operations Demo")
    print("=" * 80)
    
    # Check GPU availability
    gpu_available = Device.is_gpu_available()
    print(f"\nGPU Available: {gpu_available}")
    
    if gpu_available:
        device = Device.gpu()
        print(f"Using device: {device}")
    else:
        device = Device.cpu()
        print(f"GPU not available, using: {device}")
    
    # Create values on device
    print("\n1. Creating values on device:")
    a = Value(2.0, device=device)
    b = Value(3.0, device=device)
    print(f"   a = {a.data} (device: {a.device})")
    print(f"   b = {b.data} (device: {b.device})")
    
    # Perform operations
    print("\n2. Performing operations:")
    c = a + b
    print(f"   a + b = {c.data} (device: {c.device})")
    
    d = a * b
    print(f"   a * b = {d.data} (device: {d.device})")
    
    e = a ** 2
    print(f"   a ** 2 = {e.data} (device: {e.device})")
    
    f = b.relu()
    print(f"   b.relu() = {f.data} (device: {f.device})")
    
    # Complex expression
    print("\n3. Complex expression:")
    result = (a * b + a**2) / (b - a).relu()
    print(f"   (a*b + a^2) / relu(b-a) = {result.data} (device: {result.device})")
    
    # Backward pass
    print("\n4. Backward pass:")
    result.backward()
    print(f"   ∂result/∂a = {a.grad}")
    print(f"   ∂result/∂b = {b.grad}")


def demo_neural_network():
    """Demonstrate neural network training with GPU support."""
    print("\n" + "=" * 80)
    print("Neural Network Demo")
    print("=" * 80)
    
    # Select device
    gpu_available = Device.is_gpu_available()
    device = Device.gpu() if gpu_available else Device.cpu()
    print(f"\nUsing device: {device}")
    
    # Create a simple MLP
    print("\n1. Creating MLP:")
    model = MLP(3, [4, 4, 1], device=device)
    print(f"   Model: {model}")
    print(f"   Parameters: {len(model.parameters())}")
    
    # Sample data
    print("\n2. Sample data:")
    X = [
        [Value(2.0, device=device), Value(3.0, device=device), Value(-1.0, device=device)],
        [Value(3.0, device=device), Value(-1.0, device=device), Value(0.5, device=device)],
        [Value(0.5, device=device), Value(1.0, device=device), Value(1.0, device=device)],
        [Value(1.0, device=device), Value(1.0, device=device), Value(-1.0, device=device)],
    ]
    
    # Target values
    y_true = [
        Value(1.0, device=device),
        Value(-1.0, device=device),
        Value(-1.0, device=device),
        Value(1.0, device=device),
    ]
    
    print(f"   Training samples: {len(X)}")
    
    # Training loop
    print("\n3. Training:")
    learning_rate = 0.01
    
    for epoch in range(100):
        # Forward pass
        y_pred = []
        for x in X:
            output = model(x)
            y_pred.append(output if isinstance(output, Value) else output[0])
        
        # Compute loss (MSE)
        loss_val = Value(0.0, device=device)
        for yp, yt in zip(y_pred, y_true):
            diff = yp - yt
            loss_val = loss_val + diff * diff
        
        # Zero gradients
        for p in model.parameters():
            p.grad = 0.0
        
        # Backward pass
        loss_val.backward()
        
        # Update parameters
        for p in model.parameters():
            p.data -= learning_rate * p.grad
        
        if epoch % 10 == 0 or epoch == 99:
            print(f"   Epoch {epoch:3d}: Loss = {loss_val.data:.6f}")
    
    # Test predictions
    print("\n4. Final predictions:")
    for i, x in enumerate(X):
        output = model(x)
        y_pred = output if isinstance(output, Value) else output[0]
        print(f"   Sample {i}: Predicted = {y_pred.data:7.4f}, Target = {y_true[i].data:7.4f}")


def demo_device_movement():
    """Demonstrate moving values between devices."""
    print("\n" + "=" * 80)
    print("Device Movement Demo")
    print("=" * 80)
    
    # Create value on CPU
    print("\n1. Create value on CPU:")
    v_cpu = Value(3.14, device=Device.cpu())
    print(f"   Value: {v_cpu.data}, Device: {v_cpu.device}")
    
    # Try to move to GPU (will stay on CPU if GPU not available)
    if Device.is_gpu_available():
        print("\n2. Move to GPU:")
        v_gpu = v_cpu.to(Device.gpu())
        print(f"   Value: {v_gpu.data}, Device: {v_gpu.device}")
        
        print("\n3. Perform operations on GPU:")
        result = v_gpu * 2.0 + v_gpu ** 2
        print(f"   Result: {result.data}, Device: {result.device}")
        
        print("\n4. Move result back to CPU:")
        result_cpu = result.to(Device.cpu())
        print(f"   Result: {result_cpu.data}, Device: {result_cpu.device}")
    else:
        print("\n2. GPU not available, skipping GPU operations")
        result = v_cpu * 2.0 + v_cpu ** 2
        print(f"   Result on CPU: {result.data}, Device: {result.device}")


if __name__ == "__main__":
    print("\nmicrograd_rs GPU Support Examples")
    print("=" * 80)
    
    # Run demos
    demo_basic_operations()
    demo_neural_network()
    demo_device_movement()
    
    print("\n" + "=" * 80)
    print("All demos completed successfully!")
    print("=" * 80 + "\n")
