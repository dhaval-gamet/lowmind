import lowmind as lm
import numpy as np

def test_api_consistency():
    """Test that API is consistent and intuitive"""
    print("🧪 API Consistency Testing")
    print("=" * 40)
    
    # Test 1: Method names and signatures
    test_tensor = lm.Tensor([1, 2, 3])
    
    # Check if essential methods exist
    essential_methods = ['__add__', '__mul__', 'backward', 'reshape', 'sum']
    for method in essential_methods:
        if hasattr(test_tensor, method):
            print(f"✅ {method} exists")
        else:
            print(f"❌ {method} missing!")
    
    # Test 2: Module consistency
    test_module = lm.Linear(10, 5)
    if hasattr(test_module, 'parameters'):
        print("✅ Module.parameters() exists")
    else:
        print("❌ Module.parameters() missing")
    
    # Test 3: Optimizer consistency
    optimizer = lm.SGD(test_module.parameters(), lr=0.01)
    if hasattr(optimizer, 'step') and hasattr(optimizer, 'zero_grad'):
        print("✅ Optimizer methods exist")
    else:
        print("❌ Optimizer methods missing")
    
    # Test 4: Loss functions
    output = lm.Tensor([[1.0, 2.0, 3.0]])
    target = lm.Tensor([1])
    try:
        loss = lm.cross_entropy_loss(output, target)
        print("✅ Loss functions work")
    except Exception as e:
        print(f"❌ Loss functions failed: {e}")

test_api_consistency()