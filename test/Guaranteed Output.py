import lowmind as lm
import numpy as np
import time

def simple_demo_with_output():
    print("🚀 Starting Simple Demo with Guaranteed Output...")
    print("=" * 50)
    
    # Initialize monitoring
    monitor = lm.RaspberryPiAdvancedMonitor()
    
    # Step 1: Create simple model
    print("1. Creating Simple Neural Network...")
    model = lm.Linear(10, 5)  # Simple linear layer
    print(f"   ✅ Model created: {model}")
    print(f"   📊 Weight shape: {model.weight.shape}")
    
    # Step 2: Create sample data
    print("\n2. Creating Sample Data...")
    x_data = lm.Tensor(np.random.randn(8, 10), name='input_data')
    y_data = lm.Tensor(np.random.randint(0, 5, (8,)), name='target_data')
    print(f"   ✅ Input shape: {x_data.shape}")
    print(f"   ✅ Target shape: {y_data.shape}")
    
    # Step 3: Single training step
    print("\n3. Training Step...")
    optimizer = lm.SGD(model.parameters(), lr=0.01)
    
    # Forward pass
    output = model(x_data)
    print(f"   ✅ Output shape: {output.shape}")
    
    # Calculate loss
    loss = lm.cross_entropy_loss(output, y_data)
    print(f"   📈 Initial Loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("   ✅ Training step completed!")
    
    # Step 4: Test prediction
    print("\n4. Making Prediction...")
    test_input = lm.Tensor(np.random.randn(1, 10), name='test_input')
    prediction = model(test_input)
    predicted_class = np.argmax(prediction.data)
    
    print(f"   🔮 Predicted class: {predicted_class}")
    print(f"   📊 Prediction scores: {[f'{x:.3f}' for x in prediction.data[0]]}")
    
    # Step 5: Show memory status
    print("\n5. Memory Status:")
    mem_info = lm.memory_manager.get_memory_info()
    print(f"   💾 Memory used: {mem_info['allocated_mb']:.2f}MB / {mem_info['max_mb']:.2f}MB")
    print(f"   📈 Peak memory: {mem_info['peak_memory_mb']:.2f}MB")
    print(f"   🔢 Active tensors: {mem_info['tensors_count']}")
    
    # Step 6: System health
    print("\n6. System Health:")
    health_score = monitor.get_health_score()
    system_stats = monitor.get_system_stats()
    
    print(f"   🏥 Health Score: {health_score:.1f}/100")
    print(f"   🌡️ CPU Temperature: {system_stats['cpu_temp']:.1f}°C")
    print(f"   💻 CPU Usage: {system_stats['cpu_percent']:.1f}%")
    print(f"   🕐 Uptime: {system_stats['uptime']:.2f}s")
    
    print("\n" + "=" * 50)
    print("🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    
    return model

# Run immediately
if __name__ == "__main__":
    simple_demo_with_output()