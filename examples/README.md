# Examples

This directory contains example scripts demonstrating various features of micrograd_rs.

## GPU Demo (`gpu_demo.py`)

Comprehensive demonstration of GPU support in micrograd_rs.

**What it demonstrates:**
- Checking GPU availability
- Creating values on different devices (CPU/GPU)
- Performing operations that preserve device
- Training a neural network on GPU
- Moving values between devices

**How to run:**
```bash
# Using Python
python examples/gpu_demo.py

# Or make it executable
chmod +x examples/gpu_demo.py
./examples/gpu_demo.py
```

**Expected output:**
The script will automatically detect if GPU is available and use it. If GPU is not available, it will gracefully fall back to CPU.

The demo includes:
1. **Basic Operations**: Simple arithmetic and activation functions
2. **Neural Network Training**: A small MLP trained on sample data
3. **Device Movement**: Moving values between CPU and GPU

## More Examples

More examples will be added in the future:
- Batch processing
- Transfer learning
- Custom loss functions
- Model serialization
- Advanced training techniques

## Contributing

Feel free to contribute new examples! Please ensure they:
- Are well-documented
- Include comments explaining key concepts
- Work on both CPU and GPU (with appropriate fallbacks)
- Follow the existing code style
