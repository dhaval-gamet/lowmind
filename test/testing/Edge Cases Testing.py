import lowmind as lm
import numpy as np

def test_edge_cases():
    """Test edge cases and error handling"""
    print("🧪 Testing Edge Cases")
    print("=" * 40)
    
    # Test 1: Empty tensor
    try:
        empty_tensor = lm.Tensor([])
        print("✅ Empty tensor handled")
    except Exception as e:
        print(f"❌ Empty tensor error: {e}")
    
    # Test 2: Very large tensor
    try:
        large_tensor = lm.Tensor(np.random.randn(1000, 1000))
        print("✅ Large tensor handled")
    except MemoryError as e:
        print(f"⚠️ Large tensor memory issue: {e}")
    
    # Test 3: Gradient computation without requires_grad
    try:
        a = lm.Tensor([1, 2, 3])  # requires_grad=False by default
        b = a * 2
        b.backward()  # This should not work
        print("❌ Should have failed - no requires_grad")
    except Exception as e:
        print(f"✅ Correctly caught error: {type(e).__name__}")
    
    # Test 4: Broadcasting
    try:
        a = lm.Tensor([[1, 2, 3]])  # shape (1, 3)
        b = lm.Tensor([1, 1, 1])    # shape (3,)
        c = a + b  # Should broadcast
        print(f"✅ Broadcasting worked: {c.shape}")
    except Exception as e:
        print(f"❌ Broadcasting failed: {e}")
    
    # Test 5: In-place operations
    try:
        a = lm.Tensor([1, 2, 3])
        a.data += 1  # In-place modification
        print("✅ In-place operations work")
    except Exception as e:
        print(f"❌ In-place operations failed: {e}")

test_edge_cases()