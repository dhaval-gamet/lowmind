import unittest
import lowmind as lm
import numpy as np
import sys

class TestLowMind(unittest.TestCase):
    
    def test_tensor_creation(self):
        """Test tensor creation with different data types"""
        print("Testing tensor creation...")
        
        # Test with list
        t1 = lm.Tensor([1, 2, 3])
        self.assertEqual(t1.shape, (3,))
        
        # Test with numpy array
        t2 = lm.Tensor(np.array([1, 2, 3]))
        self.assertEqual(t2.shape, (3,))
        
        # Test with requires_grad
        t3 = lm.Tensor([1, 2, 3], requires_grad=True)
        self.assertTrue(t3.requires_grad)
    
    def test_operations(self):
        """Test basic tensor operations"""
        print("Testing operations...")
        
        a = lm.Tensor([1, 2])
        b = lm.Tensor([3, 4])
        
        # Addition
        c = a + b
        self.assertTrue(np.array_equal(c.data, [4, 6]))
        
        # Multiplication
        d = a * b
        self.assertTrue(np.array_equal(d.data, [3, 8]))
    
    def test_backpropagation(self):
        """Test gradient computation"""
        print("Testing backpropagation...")
        
        a = lm.Tensor([2.0], requires_grad=True)
        b = lm.Tensor([3.0], requires_grad=True)
        
        c = a * b
        c.backward()
        
        self.assertEqual(a.grad[0], 3.0)
        self.assertEqual(b.grad[0], 2.0)
    
    def test_memory_manager(self):
        """Test memory management"""
        print("Testing memory manager...")
        
        # Create multiple tensors
        tensors = []
        for i in range(5):
            t = lm.Tensor(np.random.randn(100, 100), name=f"test_tensor_{i}")
            tensors.append(t)
        
        mem_info = lm.memory_manager.get_memory_info()
        self.assertGreater(mem_info['allocated_mb'], 0)
        self.assertEqual(mem_info['tensors_count'], 5)
        
        # Test cleanup
        del tensors
        lm.memory_manager.clear_cache()
        mem_info = lm.memory_manager.get_memory_info()
        self.assertEqual(mem_info['tensors_count'], 0)

def run_unittest_safely():
    """Safely run unittest in IPython environment"""
    print("🧪 Running LowMind Unit Tests")
    print("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestLowMind)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("🎉 ALL UNIT TESTS PASSED! ✅")
    else:
        print("❌ SOME UNIT TESTS FAILED!")
    
    return result.wasSuccessful()

# Run this instead of unittest.main()
if __name__ == "__main__":
    run_unittest_safely()