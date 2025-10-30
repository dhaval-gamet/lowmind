import lowmind as lm
import numpy as np


class TrainingVisualizer:
    def __init__(self):
        self.loss_history = []
        self.accuracy_history = []
        self.memory_history = []
        
    def update(self, loss, accuracy=None):
        self.loss_history.append(loss)
        if accuracy is not None:
            self.accuracy_history.append(accuracy)
        
        mem_info = lm.memory_manager.get_memory_info()
        self.memory_history.append(mem_info['allocated_mb'])
        
        self.plot_progress()
    
    def plot_progress(self):
        # Simple text-based plotting
        if len(self.loss_history) % 5 == 0:
            current_epoch = len(self.loss_history)
            loss = self.loss_history[-1]
            
            # Progress bar for loss
            loss_bar = "█" * int(loss * 10) + "░" * (10 - int(loss * 10))
            
            if self.accuracy_history:
                acc = self.accuracy_history[-1]
                acc_bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
                print(f"Epoch {current_epoch:3d} | Loss: {loss:.3f} [{loss_bar}] | Acc: {acc:.3f} [{acc_bar}]")
            else:
                print(f"Epoch {current_epoch:3d} | Loss: {loss:.3f} [{loss_bar}]")
    
    def final_report(self):
        print("\n" + "=" * 60)
        print("📊 TRAINING REPORT")
        print("=" * 60)
        
        if self.loss_history:
            print(f"📉 Final Loss: {self.loss_history[-1]:.4f}")
            print(f"📈 Best Loss: {min(self.loss_history):.4f}")
            print(f"📊 Loss Improvement: {self.loss_history[0] - self.loss_history[-1]:.4f}")
        
        if self.accuracy_history:
            print(f"🎯 Final Accuracy: {self.accuracy_history[-1]:.3f}")
            print(f"💫 Best Accuracy: {max(self.accuracy_history):.3f}")
        
        if self.memory_history:
            print(f"💾 Peak Memory: {max(self.memory_history):.2f}MB")
            print(f"💽 Avg Memory: {np.mean(self.memory_history):.2f}MB")

def advanced_training_demo():
    print("🚀 Advanced Training with Visualization")
    print("=" * 50)
    
    # Create a more complex model
    class AdvancedNet(lm.Module):
        def __init__(self, input_size, hidden_sizes, output_size):
            super().__init__()
            self.layers = []
            
            # Create multiple layers
            prev_size = input_size
            for i, hidden_size in enumerate(hidden_sizes):
                self.layers.append(lm.Linear(prev_size, hidden_size))
                prev_size = hidden_size
            
            self.output_layer = lm.Linear(prev_size, output_size)
            
            # Register parameters
            for i, layer in enumerate(self.layers):
                self._parameters[f'layer_{i}'] = layer.weight
                if layer.bias is not None:
                    self._parameters[f'bias_{i}'] = layer.bias
            
            self._parameters['output_weight'] = self.output_layer.weight
            if self.output_layer.bias is not None:
                self._parameters['output_bias'] = self.output_layer.bias
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x).relu()
            x = self.output_layer(x)
            return x
    
    # Create model and data
    model = AdvancedNet(20, [64, 32, 16], 5)
    X = np.random.randn(500, 20)
    y = np.random.randint(0, 5, 500)
    
    optimizer = lm.SGD(model.parameters(), lr=0.01)
    visualizer = TrainingVisualizer()
    
    print("🔄 Starting Training...")
    
    for epoch in range(30):
        # Training
        outputs = model(lm.Tensor(X))
        loss = lm.cross_entropy_loss(outputs, lm.Tensor(y))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predictions = np.argmax(outputs.data, axis=1)
        accuracy = np.mean(predictions == y)
        
        # Update visualizer
        visualizer.update(loss.item(), accuracy)
        
        # Memory management
        if epoch % 10 == 0:
            lm.memory_manager.free_unused()
    
    # Final report
    visualizer.final_report()
    
    print("✅ Advanced Training Completed!")
    return model

# Run advanced training
if __name__ == "__main__":
    trained_model = advanced_training_demo()