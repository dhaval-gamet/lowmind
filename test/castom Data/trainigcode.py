import lowmind as lm
import numpy as np

class AdvancedNN(lm.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = lm.Linear(input_size, hidden_size)
        self.fc2 = lm.Linear(hidden_size, hidden_size//2)
        self.fc3 = lm.Linear(hidden_size//2, num_classes)
        self.dropout = lm.Dropout(0.3)
    
    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.dropout(x)
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x

def create_sample_data(num_samples=1000, input_size=20, num_classes=5):
    """Sample data create करें"""
    X = np.random.randn(num_samples, input_size)
    y = np.random.randint(0, num_classes, num_samples)
    return X, y

def train_advanced_model():
    print("🚀 Advanced Training शुरू...")
    
    # डेटा तैयार करें
    X_train, y_train = create_sample_data(1000, 20, 5)
    X_val, y_val = create_sample_data(200, 20, 5)
    
    # मॉडल बनाएं
    model = AdvancedNN(20, 64, 5)
    optimizer = lm.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    print(f"📊 Training samples: {len(X_train)}")
    print(f"📊 Validation samples: {len(X_val)}")
    print(f"🔢 Model parameters: {sum(np.prod(param.data.shape) for param in model.parameters())}")
    
    # Training loop
    batch_size = 32
    best_loss = float('inf')
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            end_idx = min(i + batch_size, len(X_train))
            
            batch_x = lm.Tensor(X_train[i:end_idx])
            batch_y = lm.Tensor(y_train[i:end_idx])
            
            # Forward pass
            outputs = model(batch_x)
            loss = lm.cross_entropy_loss(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
        
        avg_train_loss = total_loss / batches
        
        # Validation
        model.eval()
        val_outputs = model(lm.Tensor(X_val))
        val_loss = lm.cross_entropy_loss(val_outputs, lm.Tensor(y_val))
        val_loss_value = val_loss.item()
        
        print(f"📈 Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss_value:.4f}")
        
        # Best model save करें
        if val_loss_value < best_loss:
            best_loss = val_loss_value
            print(f"💾 New best model! Loss: {best_loss:.4f}")
    
    return model

# Run करें
if __name__ == "__main__":
    trained_model = train_advanced_model()