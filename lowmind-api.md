
# LowMind Deep Learning Framework - Complete API Reference

## 🧠 Core Tensor Operations

### Tensor Initialization
```python
Tensor(data, requires_grad=False, device='cpu', name=None, persistent=False)
```
**Parameters:**
- `data`: numpy array or list
- `requires_grad`: Enable gradient computation
- `device`: 'cpu' (Raspberry Pi optimized)
- `name`: Memory management identifier
- `persistent`: Prevent automatic cleanup

### Basic Mathematical Operations
```python
# Arithmetic operations
a + b, a - b, a * b, a / b, a ** power
-a  # Negation
a @ b  # Matrix multiplication

# Comparison operations (returns numpy arrays)
a.data == b.data, a.data > b.data, etc.
```

### Tensor Operations
```python
# Reduction operations
tensor.sum(axis=None, keepdims=False)
tensor.mean(axis=None, keepdims=False)

# Shape operations
tensor.reshape(shape)
tensor.transpose(axes=None)
tensor.T  # Transpose property
tensor.squeeze(axis=None)
tensor.unsqueeze(axis)

# Activation functions
tensor.relu()
tensor.sigmoid()
tensor.tanh()
tensor.exp()
tensor.log()

# Advanced operations
tensor.matmul_memory_efficient(other)  # For large matrices
tensor.backward(grad=None)  # Memory-optimized backpropagation
```

### Tensor Properties
```python
tensor.shape    # Get tensor shape
tensor.ndim     # Get number of dimensions
tensor.data     # Access underlying numpy array
tensor.grad     # Access gradients
tensor.item()   # Get scalar value
tensor.requires_grad  # Check if gradient tracking enabled
```

## 🏗️ Neural Network Modules

### Base Module Class
```python
class Module:
    def parameters(self)          # Generator for all parameters
    def named_parameters(self)    # Generator for (name, param) pairs
    def train()                  # Set training mode
    def eval()                   # Set evaluation mode
    def forward(x)               # Abstract method - implement in subclasses
    def __call__(x)              # Enables module(x) syntax
```

### Layer Classes

#### Linear Layer
```python
Linear(in_features, out_features, bias=True, device='cpu')
```
**Usage:**
```python
layer = Linear(128, 64)  # 128 input features, 64 output features
output = layer(input_tensor)
```

#### Convolutional Layer
```python
Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, device='cpu')
```
**Usage:**
```python
conv = Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
output = conv(input_4d_tensor)  # Shape: (batch, channels, height, width)
```

#### Dropout Layer
```python
Dropout(p=0.5)  # Dropout probability
```
**Usage:**
```python
dropout = Dropout(0.3)
output = dropout(tensor)  # Only active during training
```

## 📉 Loss Functions

### Cross Entropy Loss
```python
cross_entropy_loss(output, target)
```
**Parameters:**
- `output`: Raw logits from model (shape: [batch, classes])
- `target`: Ground truth labels (shape: [batch])

**Usage:**
```python
loss = cross_entropy_loss(predictions, targets)
loss.backward()
```

### Mean Squared Error Loss
```python
mse_loss(output, target)
```
**Usage:**
```python
loss = mse_loss(predictions, targets)
```

## ⚙️ Optimizers

### Stochastic Gradient Descent
```python
SGD(params, lr=0.01, momentum=0, weight_decay=0)
```
**Methods:**
- `zero_grad()`: Reset all gradients to zero
- `step()`: Update parameters using computed gradients

**Usage:**
```python
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 🎯 Pre-defined Models

### MicroCNN (Raspberry Pi Optimized)
```python
MicroCNN(num_classes=10, device='cpu')
```
**Architecture:**
- Conv2d(3, 8, 3) → ReLU
- Conv2d(8, 16, 3) → ReLU
- Global Average Pooling
- Linear(16, num_classes)
- Dropout(0.1)

**Usage:**
```python
model = MicroCNN(num_classes=10)
output = model(input_tensor)  # Input: (batch, 3, 32, 32)
```

## 💾 Memory Management

### Memory Manager
```python
memory_manager = MemoryManager(max_memory_mb=64)
```

**Methods:**
```python
memory_manager.allocate(tensor, name)     # Register tensor
memory_manager.free(name)                 # Free specific tensor
memory_manager.free_unused()              # LRU cleanup
memory_manager.free_all_non_essential()   # Aggressive cleanup
memory_manager.clear_cache()              # Clear all cached tensors
memory_manager.optimize_for_inference()   # Remove training overhead
memory_manager.get_memory_info()          # Detailed memory stats
```

### Memory Tracing Context
```python
with memory_trace("Operation Name"):
    # Your operations here
    result = some_expensive_operation()
```

## 📊 System Monitoring

### Raspberry Pi Advanced Monitor
```python
monitor = RaspberryPiAdvancedMonitor()
```

**Methods:**
```python
monitor.get_system_stats()        # Comprehensive system stats
monitor.update_monitoring()       # Update all metrics
monitor.print_detailed_status()   # Print status report
monitor.get_health_score()        # System health score (0-100)
```

## 🚀 Utility Functions

### Advanced Testing
```python
advanced_raspberry_pi_test()  # Comprehensive framework test
```

### Shape Compatibility
```python
# Automatic broadcasting in operations
tensor_a = Tensor(np.random.randn(32, 1))    # Shape: (32, 1)
tensor_b = Tensor(np.random.randn(32, 64))   # Shape: (32, 64)
result = tensor_a + tensor_b                 # Broadcasts to (32, 64)
```

## 💡 Usage Examples

### Complete Training Loop
```python
# Model definition
model = MicroCNN(num_classes=10)
optimizer = SGD(model.parameters(), lr=0.01)

# Training mode
model.train()
for epoch in range(10):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = cross_entropy_loss(output, batch_y)
        loss.backward()
        optimizer.step()

# Inference mode
model.eval()
with memory_trace("Inference"):
    predictions = model(test_data)
```

### Memory-Efficient Inference
```python
# Optimize for inference
memory_manager.optimize_for_inference()

with memory_trace("Batch Inference"):
    for batch in test_batches:
        output = model(batch)
        predictions.append(output.data)
```

## 🛠️ Advanced Features

### Gradient Clipping (Manual)
```python
# Manual gradient clipping
max_grad_norm = 1.0
for param in model.parameters():
    if param.grad is not None:
        grad_norm = np.linalg.norm(param.grad)
        if grad_norm > max_grad_norm:
            param.grad = param.grad * (max_grad_norm / grad_norm)
```

### Custom Layer Implementation
```python
class CustomLayer(Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weights = Tensor(
            np.random.randn(output_size, input_size) * 0.01,
            requires_grad=True,
            name='custom_weights'
        )
        self.bias = Tensor(
            np.zeros(output_size),
            requires_grad=True, 
            name='custom_bias'
        )
        self._parameters['weights'] = self.weights
        self._parameters['bias'] = self.bias
    
    def forward(self, x):
        return x @ self.weights.T + self.bias
```

## 📈 Performance Tips

1. **Use `matmul_memory_efficient`** for large matrices
2. **Enable `persistent=True`** for frequently used tensors
3. **Call `memory_manager.free_unused()`** periodically during training
4. **Use `memory_trace` context** to identify memory bottlenecks
5. **Set `requires_grad=False`** for inference to save memory
6. **Use `model.eval()`** during inference to disable dropout
