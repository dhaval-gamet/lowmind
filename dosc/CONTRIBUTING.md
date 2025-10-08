# CONTRIBUTING.md

```markdown
# Contributing to LowMind Framework 🧠

First off, thank you for considering contributing to LowMind! It's people like you that make LowMind such a great framework for Raspberry Pi and edge device AI.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct:
- Be respectful and inclusive
- Exercise consideration and respect in your speech and actions
- Attempt collaboration before conflict
- Refrain from demeaning, discriminatory, or harassing behavior

## Getting Started

### Prerequisites
- Python 3.6 or higher
- Raspberry Pi (for testing Raspberry Pi specific features)
- Basic understanding of deep learning concepts

### Setting Up Development Environment

1. **Fork the repository**
   ```bash
   git clone https://github.com/dhaval-gamet/lowmind.git
   cd lowmind
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## How Can I Contribute?

### 🐛 Reporting Bugs
When reporting bugs, please include:
- Clear description of the issue
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, Raspberry Pi model)
- Error logs or screenshots

### 💡 Suggesting Enhancements
We welcome feature ideas! Please:
- Use a clear and descriptive title
- Provide a detailed description of the proposed feature
- Explain why this feature would be useful
- Include examples of how it would be used

### 🔧 Types of Contributions We're Looking For

#### High Priority:
- Memory optimization for Raspberry Pi
- New activation functions
- Additional loss functions
- Performance improvements
- Bug fixes

#### Medium Priority:
- New layer types (LSTM, GRU, etc.)
- Additional optimizers
- Model examples and tutorials
- Documentation improvements

#### Experimental:
- Quantization support
- Pruning techniques
- Hardware-specific optimizations

## Development Workflow

### Branch Naming Convention
```
feature/description    # For new features
bugfix/description     # For bug fixes
docs/description       # For documentation
refactor/description   # For code refactoring
```

### Commit Message Guidelines
We follow conventional commit messages:
```
feat: add new convolution layer
fix: memory leak in tensor operations
docs: update installation guide
refactor: optimize matrix multiplication
test: add tests for dropout layer
```

Example:
```bash
git commit -m "feat: add batch normalization layer for better training stability"
```

## Coding Standards

### Python Code Style
We follow PEP 8 with some specific guidelines for deep learning code:

```python
# ✅ Good
class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self._init_grad()
    
    def backward(self, grad=None):
        """Memory-optimized backward pass"""
        # Implementation here

# ❌ Avoid
class tensor:
    def __init__(self,data,requires_grad=False):
        self.data = np.array(data,dtype=np.float32)
        self.requires_grad=requires_grad
```

### Memory Efficiency Guidelines
Since LowMind targets Raspberry Pi, memory efficiency is crucial:

1. **Use NumPy efficiently:**
   ```python
   # ✅ Good - no unnecessary copies
   result = np.zeros(shape, dtype=np.float32)
   
   # ❌ Avoid - creates intermediate arrays
   result = np.array(some_list).reshape(shape)
   ```

2. **Lazy initialization:**
   ```python
   def _init_grad(self):
       """Initialize gradient only when needed"""
       if self.grad is None and self.requires_grad:
           self.grad = np.zeros_like(self.data)
   ```

3. **Clean up properly:**
   ```python
   def __del__(self):
       """Clean up memory when tensor is deleted"""
       if hasattr(self, 'name') and self.name:
           memory_manager.free(self.name)
   ```

### Documentation Standards

#### Class Documentation:
```python
class Linear(Module):
    """
    A linear (fully connected) layer.
    
    Parameters:
    -----------
    in_features : int
        Size of each input sample
    out_features : int
        Size of each output sample
    bias : bool, default=True
        If set to False, the layer will not learn an additive bias
    
    Examples:
    ---------
    >>> layer = Linear(10, 5)
    >>> input_tensor = Tensor(np.random.randn(32, 10))
    >>> output = layer(input_tensor)
    >>> print(output.shape)
    (32, 5)
    """
```

#### Function Documentation:
```python
def relu(self):
    """
    Apply Rectified Linear Unit (ReLU) activation function.
    
    Returns:
    --------
    Tensor
        A new tensor with ReLU applied element-wise
    
    Notes:
    ------
    ReLU is defined as: f(x) = max(0, x)
    
    This operation is memory-efficient and in-place where possible.
    """
    return Tensor(np.maximum(0, self.data), 
                 requires_grad=self.requires_grad)
```


### Writing Tests
```python
def test_tensor_addition():
    """Test basic tensor addition with gradient tracking"""
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5, 6], requires_grad=True)
    c = a + b
    
    # Test forward pass
    assert np.array_equal(c.data, [5, 7, 9])
    
    # Test backward pass
    c.backward()
    assert np.array_equal(a.grad, [1, 1, 1])
    assert np.array_equal(b.grad, [1, 1, 1])

def test_memory_usage():
    """Test that operations don't leak memory"""
    initial_memory = memory_manager.allocated_memory
    
    # Perform operations
    a = Tensor(np.random.randn(100, 100))
    b = Tensor(np.random.randn(100, 100))
    c = a @ b
    
    # Clean up
    del a, b, c
    memory_manager.free_unused()
    
    # Memory should return to similar level
    final_memory = memory_manager.allocated_memory
    assert abs(final_memory - initial_memory) < 1024  # Within 1KB
```

### Raspberry Pi Specific Testing
If you have a Raspberry Pi, please test:
- Memory usage under load
- Temperature during training
- Performance with different model sizes
- Compatibility with different Raspberry Pi models


## Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**
3. **Make your changes**
4. **Add or update tests**
5. **Update documentation**
6. **Ensure all tests pass**
7. **Submit pull request**

### PR Checklist:
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows coding standards
- [ ] Commit messages follow conventions
- [ ] All CI checks pass
- [ ] Memory usage optimized for Raspberry Pi

### PR Template:
```markdown
## Description
Brief description of the changes

## Related Issue
Fixes # (issue number)

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing Performed
- [ ] Unit tests added
- [ ] Raspberry Pi testing performed
- [ ] Memory usage tested
- [ ] Performance benchmarks

## Screenshots/Logs
(If applicable)

## Additional Notes
Any additional information reviewers should know
```

## Community

### Getting Help
- 📚 Check the [documentation](README.md)
- 🐛 Create an [issue](https://github.com/dhaval-gamet/lowmind/issues)
- 💬 Join our [Discord community](link-to-discord)
- 🐦 Follow us on [Twitter](link-to-twitter)

### Recognition
Great contributors will be:
- Featured in our contributors list
- Mentioned in release notes
- Given commit access (for regular contributors)

### Raspberry Pi Community
We especially value contributions from Raspberry Pi users! Share:
- Your use cases
- Performance benchmarks
- Real-world applications
- Tutorials and examples

## License

By contributing to LowMind, you agree that your contributions will be licensed under the same [MIT License](LICENSE) that covers the project.

---

## Thank You! 🎉

Your contributions are what make open source amazing. Thank you for helping make LowMind better for everyone, especially the Raspberry Pi and edge AI community!

If you have any questions, don't hesitate to ask in the issues or reach out to the maintainers.

Happy coding! 🚀