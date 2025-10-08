import numpy as np
from lowmind import Tensor, SGD

class BinaryClassificationDataLoader:
    def __init__(self):
        pass
    
    def generate_binary_data(self, num_samples=200):
        """Generate synthetic binary classification data"""
        np.random.seed(42)
        X = np.random.randn(num_samples, 2).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
        return X, y.reshape(-1, 1)

class SimpleMLP:
    """Simple Multi-Layer Perceptron for binary classification"""
    def __init__(self, input_size, hidden_sizes, output_size, device='cpu'):
        self.layers = []
        
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            from lowmind import Linear
            layer = Linear(sizes[i], sizes[i+1], device=device)
            self.layers.append(layer)
    
    def parameters(self):
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.relu()
        return x
    
    def __call__(self, x):
        return self.forward(x)

class BinaryClassificationTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.accuracies = []
        self.losses = []
    
    def train(self, X, y, epochs=100):
        """Train binary classifier"""
        X_tensor = Tensor(X)
        y_tensor = Tensor(y)
        
        print("🎯 Training Binary Classifier")
        print("=" * 40)
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            logits = self.model(X_tensor)
            predictions = logits.sigmoid()
            
            loss = - (y_tensor * predictions.log() + (1 - y_tensor) * (1 - predictions).log()).mean()
            
            loss.backward()
            self.optimizer.step()
            
            pred_classes = (predictions.data > 0.5).astype(int).flatten()
            true_classes = y_tensor.data.astype(int).flatten()
            accuracy = np.mean(pred_classes == true_classes)
            
            self.accuracies.append(accuracy)
            self.losses.append(loss.data[0])
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d} | Loss: {loss.data[0]:.4f} | Acc: {accuracy*100:.2f}%")
        
        return self.accuracies, self.losses

def train_binary_classifier():
    """Train a binary classifier"""
    
    data_loader = BinaryClassificationDataLoader()
    X, y = data_loader.generate_binary_data(200)
    
    model = SimpleMLP(input_size=2, hidden_sizes=[4, 4], output_size=1, device='cpu')
    optimizer = SGD(model.parameters(), lr=0.01)
    
    trainer = BinaryClassificationTrainer(model, optimizer)
    accuracies, losses = trainer.train(X, y, epochs=100)
    
    print(f"\n✅ Binary Classification Training Completed!")
    print(f"Final accuracy: {accuracies[-1]*100:.2f}%")
    
    return model, accuracies, losses

if __name__ == "__main__":
    model, accuracies, losses = train_binary_classifier()
