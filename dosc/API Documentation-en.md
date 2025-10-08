# LowMind - Raspberry Pi Deep Learning Framework API Documentation

## 📖 Overview

LowMind 2.2 is an **ultra-optimized deep learning framework** specifically designed for **Raspberry Pi's limited resources**. It provides memory-efficient tensor operations, advanced system monitoring, and lightweight model architectures.

## 🏗️ Core Architecture

### Memory Management System

```python
class MemoryManager:
    """
    Advanced memory manager optimized for Raspberry Pi's limited resources
    """
    
    def __init__(self, max_memory_mb=128)
    def allocate(self, tensor, name=None)
    def free(self, name)
    def free_unused(self)
    def free_all_non_essential(self)
    def clear_cache(self)
    def get_memory_info(self)
    def optimize_for_inference(self)
```

### Tensor Operations

```python
class Tensor:
    """
    Ultra-Optimized Tensor Class for Raspberry Pi
    """
    
    # Core Operations
    def __add__(self, other)        # Addition
    def __mul__(self, other)        # Multiplication  
    def __matmul__(self, other)     # Matrix multiplication
    def relu(self)                  # ReLU activation
    def sigmoid(self)               # Sigmoid activation
    def backward(self, grad=None)   # Memory-optimized backprop
    
    # Memory-Efficient Methods
    def matmul_memory_efficient(self, other)
    def _chunked_matmul(self, other, chunk_size=512)
```

## 🧩 Core Modules

### Neural Network Layers

```python
class Module:
    """Base class for all neural network modules"""
    def parameters(self)
    def named_parameters(self) 
    def train()
    def eval()
    def forward(x)

class Linear(Module):
    """Fully connected layer"""
    def __init__(self, in_features, out_features, bias=True, device='cpu')

class Conv2d(Module):
    """2D Convolutional layer"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, device='cpu')

class Dropout(Module):
    """Dropout layer for regularization"""
    def __init__(self, p=0.5)
```

### Loss Functions

```python
def cross_entropy_loss(output, target)
def mse_loss(output, target)
```

### Optimizers

```python
class SGD:
    """Stochastic Gradient Descent with momentum"""
    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0)
    def zero_grad()
    def step()
```

## 🔍 Monitoring & Profiling

### System Monitoring

```python
class RaspberryPiAdvancedMonitor:
    """Comprehensive system monitoring for Raspberry Pi"""
    
    def get_system_stats(self)
    def update_monitoring(self) 
    def print_detailed_status(self)
    def get_health_score(self)
```

### Memory Profiling

```python
class memory_trace:
    """Context manager for memory usage tracing"""
    def __init__(self, name)
    def __enter__()
    def __exit__()
```

## 🚀 Pre-built Models

### Lightweight Architectures

```python
class MicroCNN(Module):
    """Ultra-lightweight CNN for Raspberry Pi"""
    def __init__(self, num_classes=10, device='cpu')
```

## 💡 Usage Examples

### Basic Tensor Operations

```python
# Create tensors with memory management
a = Tensor(np.random.randn(50, 50), requires_grad=True, name='tensor_a')
b = Tensor(np.random.randn(50, 50), requires_grad=True, name='tensor_b')

# Memory-efficient operations
c = a.matmul_memory_efficient(b)
result = c.relu()
```

### Building Neural Networks

```python
# Create a simple model
model = MicroCNN(num_classes=10, device='cpu')

# Training mode
model.train()

# Inference mode  
model.eval()

# Forward pass
output = model(input_tensor)
loss = cross_entropy_loss(output, target)
```

### Training Loop

```python
# Initialize optimizer
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(data)
    loss = cross_entropy_loss(output, target)
    loss.backward()
    optimizer.step()
```

### System Monitoring

```python
# Initialize monitoring
monitor = RaspberryPiAdvancedMonitor()

# Get comprehensive status
monitor.print_detailed_status()

# Health assessment
health_score = monitor.get_health_score()
```

## ⚙️ Configuration

### Memory Settings

```python
# Global memory manager configuration
memory_manager = MemoryManager(max_memory_mb=64)  # Conservative limit for RPi

# Tensor creation with optimization
tensor = Tensor(data, requires_grad=True, device='cpu', persistent=False)
```

### Performance Optimization

```python
# Enable memory tracing
with memory_trace("Operation Name"):
    # Your operations here
    result = expensive_operation()

# Force memory cleanup
memory_manager.free_unused()
gc.collect()

# Optimize for inference
memory_manager.optimize_for_inference()
```

## 📊 Advanced Features

### 1. **Memory-Efficient Backpropagation**
- Lazy gradient allocation
- Chunked matrix operations
- Automatic memory cleanup during backward pass

### 2. **System Health Monitoring**
- Real-time CPU temperature tracking
- Memory usage analytics
- Health scoring system
- Automatic warnings for critical conditions

### 3. **Raspberry Pi Optimization**
- Conservative memory limits
- Temperature-aware operations
- Process priority management
- Dynamic batch size adjustment

### 4. **Debugging & Profiling**
- Memory usage tracing
- Operation timing
- Gradient flow visualization
- System resource reporting

## 🛠️ Best Practices

### Memory Management
```python
# Use memory-efficient operations
result = a.matmul_memory_efficient(b)  # Instead of a @ b

# Free unused tensors regularly
memory_manager.free_unused()

# Use context managers for profiling
with memory_trace("Training Step"):
    train_step()
```

### Model Design
```python
# Use lightweight architectures
model = MicroCNN(num_classes=10)

# Enable/disable gradients as needed
with torch.no_grad():
    inference_output = model(input_data)
```

### System Monitoring
```python
# Regular health checks
if monitor.get_health_score() < 60:
    print("Warning: System health critical")
    # Reduce batch size or model complexity
```

## 🔧 Troubleshooting

### Common Issues & Solutions

1. **Memory Errors**
   ```python
   # Reduce memory limits
   memory_manager = MemoryManager(max_memory_mb=32)
   # Use chunked operations
   result = tensor._chunked_matmul(other, chunk_size=256)
   ```

2. **Performance Issues**
   ```python
   # Monitor system health
   monitor.print_detailed_status()
   # Optimize for inference
   memory_manager.optimize_for_inference()
   ```

3. **Gradient Problems**
   ```python
   # Check gradient flow
   print(f"Gradient norm: {np.linalg.norm(tensor.grad)}")
   # Use gradient clipping if needed
   ```

## 📈 Performance Metrics

The framework provides comprehensive monitoring:
- Memory usage (allocated/peak/max)
- CPU temperature and usage
- System memory availability
- Operation timing
- Health scoring (0-100)

## 🎯 Use Cases

### Ideal For:
- **Edge AI applications** on Raspberry Pi
- **Educational projects** with resource constraints
- **Prototyping** lightweight neural networks
- **IoT deployments** with limited memory
- **Research** on efficient deep learning

### Not Recommended For:
- Large-scale training (use PyTorch/TensorFlow instead)
- High-performance computing
- Models with billions of parameters

## 🔮 Future Extensions

The framework is designed to be extensible:
- Custom layer implementations
- Additional optimizer algorithms
- Hardware-specific optimizations
- Distributed training support
- Quantization and pruning techniques

---

**Note**: This framework is specifically optimized for **Raspberry Pi 3/4/Zero** with limited RAM (1GB-4GB). For larger Raspberry Pi models (8GB), you can increase the `max_memory_mb` parameter accordingly.

**🚀 Start building efficient AI applications on your Raspberry Pi today!**