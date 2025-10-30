import lowmind as lm
import numpy as np
import time

class MNISTClassifier(lm.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = lm.Linear(784, 256)
        self.fc2 = lm.Linear(256, 128)
        self.fc3 = lm.Linear(128, 10)
        self.dropout = lm.Dropout(0.3)
    
    def forward(self, x):
        x = x.reshape((-1, 784))  # Flatten images
        x = self.fc1(x).relu()
        x = self.dropout(x)
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x

def train_mnist_classifier():
    print("🎯 MNIST Digit Classifier Training")
    print("=" * 50)
    
    # Create sample MNIST-like data
    def create_mnist_data(num_samples=1000):
        X = np.random.randn(num_samples, 1, 28, 28).astype(np.float32)
        y = np.random.randint(0, 10, num_samples)
        return X, y
    
    model = MNISTClassifier()
    optimizer = lm.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    X_train, y_train = create_mnist_data(1000)
    X_test, y_test = create_mnist_data(200)
    
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Test samples: {len(X_test)}")
    print(f"🔢 Model parameters: {sum(np.prod(param.data.shape) for param in model.parameters()):,}")
    
    # Training loop
    best_accuracy = 0
    for epoch in range(10):
        model.train()
        total_loss = 0
        batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), 32):
            batch_x = lm.Tensor(X_train[i:i+32])
            batch_y = lm.Tensor(y_train[i:i+32])
            
            outputs = model(batch_x)
            loss = lm.cross_entropy_loss(outputs, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
        
        avg_loss = total_loss / batches
        
        # Validation
        model.eval()
        test_outputs = model(lm.Tensor(X_test))
        test_predictions = np.argmax(test_outputs.data, axis=1)
        accuracy = np.mean(test_predictions == y_test)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            status = " 💫 NEW BEST!"
        else:
            status = ""
        
        print(f"📈 Epoch {epoch+1:2d} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.3f}{status}")
    
    print("=" * 50)
    print(f"🎯 Training Completed! Best Accuracy: {best_accuracy:.3f}")
    
    return model

# Run MNIST classifier
if __name__ == "__main__":
    trained_model = train_mnist_classifier()