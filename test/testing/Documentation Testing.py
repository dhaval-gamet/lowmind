def test_examples_in_readme():
    """Test that all examples in README work correctly"""
    print("🧪 Testing README Examples")
    print("=" * 40)
    
    # Example 1: Basic tensor operations
    print("1. Basic Tensor Operations:")
    a = lm.Tensor([1, 2, 3])
    b = lm.Tensor([4, 5, 6])
    c = a + b
    print(f"   ✅ {a.data} + {b.data} = {c.data}")
    
    # Example 2: Neural network
    print("\n2. Neural Network:")
    model = lm.Linear(10, 5)
    x = lm.Tensor(np.random.randn(1, 10))
    y = model(x)
    print(f"   ✅ Input: {x.shape} -> Output: {y.shape}")
    
    # Example 3: Training loop
    print("\n3. Training Loop:")
    optimizer = lm.SGD(model.parameters(), lr=0.01)
    output = model(x)
    loss = lm.cross_entropy_loss(output, lm.Tensor([0]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("   ✅ Training step completed")
    
    print("\n✅ All README examples work correctly!")

test_examples_in_readme()