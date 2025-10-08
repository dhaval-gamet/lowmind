# anomaly_detection.py
import numpy as np
from lowmind import Tensor, Linear, SGD

class Autoencoder:
    """Simple Autoencoder for anomaly detection"""
    def __init__(self, input_dim, encoding_dim=2, device='cpu'):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.device = device
        
        # Encoder
        self.encoder1 = Linear(input_dim, 8, device=device)
        self.encoder2 = Linear(8, encoding_dim, device=device)
        
        # Decoder
        self.decoder1 = Linear(encoding_dim, 8, device=device)
        self.decoder2 = Linear(8, input_dim, device=device)
    
    def parameters(self):
        """Get all parameters"""
        params = []
        params.extend(self.encoder1.parameters())
        params.extend(self.encoder2.parameters())
        params.extend(self.decoder1.parameters())
        params.extend(self.decoder2.parameters())
        return params
    
    def encode(self, x):
        """Encode input"""
        x = self.encoder1(x).relu()
        x = self.encoder2(x)
        return x
    
    def decode(self, x):
        """Decode latent representation"""
        x = self.decoder1(x).relu()
        x = self.decoder2(x).sigmoid()
        return x
    
    def forward(self, x):
        """Full autoencoder forward pass"""
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

def generate_sensor_data():
    """Generate synthetic sensor data with anomalies"""
    np.random.seed(42)
    
    # Normal sensor readings (temperature, pressure, humidity, vibration)
    num_normal = 200
    normal_data = np.random.randn(num_normal, 4) * 0.3 + np.array([25, 100, 50, 5])
    
    # Anomalous readings (different patterns)
    num_anomalies = 20
    
    # Type 1: High temperature anomalies
    anomalies1 = np.random.randn(num_anomalies//2, 4) * 0.3 + np.array([45, 100, 50, 5])
    
    # Type 2: High vibration anomalies  
    anomalies2 = np.random.randn(num_anomalies//2, 4) * 0.3 + np.array([25, 100, 50, 15])
    
    anomaly_data = np.vstack([anomalies1, anomalies2])
    
    # Combine data
    all_data = np.vstack([normal_data, anomaly_data])
    labels = np.array([0] * num_normal + [1] * num_anomalies)
    
    # Normalize data
    data_min = all_data.min(axis=0)
    data_max = all_data.max(axis=0)
    normalized_data = (all_data - data_min) / (data_max - data_min + 1e-8)
    
    return normalized_data, labels

def train_anomaly_detector():
    """Train autoencoder for anomaly detection"""
    print("🚨 Training Anomaly Detection System")
    print("=" * 50)
    
    # Generate sensor data
    data, labels = generate_sensor_data()
    print(f"Generated {len(data)} samples ({np.sum(labels==0)} normal, {np.sum(labels==1)} anomalies)")
    print("Features: [Temperature, Pressure, Humidity, Vibration]")
    
    # Split normal data for training
    normal_indices = np.where(labels == 0)[0]
    train_indices = normal_indices[:150]  # Use 150 normal samples for training
    test_indices = normal_indices[150:]   # Remaining normal + all anomalies for testing
    
    # Add anomalies to test set
    anomaly_indices = np.where(labels == 1)[0]
    test_indices = np.concatenate([test_indices, anomaly_indices])
    
    train_data = data[train_indices]
    test_data = data[test_indices]
    test_labels = labels[test_indices]
    
    # Model
    model = Autoencoder(input_dim=4, encoding_dim=2)
    
    # Training
    optimizer = SGD(model.parameters(), lr=0.01)
    epochs = 200
    batch_size = 16
    
    print("\n🔧 Training Autoencoder on Normal Data...")
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        # Shuffle training data
        indices = np.random.permutation(len(train_data))
        
        for i in range(0, len(train_data), batch_size):
            batch_indices = indices[i:min(i+batch_size, len(train_data))]
            batch_data = train_data[batch_indices]
            
            # Forward pass
            optimizer.zero_grad()
            reconstructed = model(Tensor(batch_data))
            
            # Reconstruction loss
            loss = ((reconstructed - Tensor(batch_data)) ** 2).mean()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data[0]
            num_batches += 1
        
        if epoch % 40 == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch:3d} | Reconstruction Loss: {avg_loss:.6f}")
    
    print(f"\n✅ Autoencoder Training Completed!")
    
    # Test anomaly detection
    print("\n🔍 Testing Anomaly Detection...")
    
    reconstruction_errors = []
    for i in range(len(test_data)):
        data_point = test_data[i:i+1]
        reconstructed = model(Tensor(data_point))
        error = np.mean((reconstructed.data - data_point) ** 2)
        reconstruction_errors.append(error)
    
    # Calculate threshold (mean + 2*std of normal data errors)
    normal_test_errors = [reconstruction_errors[i] for i in range(len(test_data)) if test_labels[i] == 0]
    threshold = np.mean(normal_test_errors) + 2 * np.std(normal_test_errors)
    
    print(f"\n📊 Anomaly Detection Threshold: {threshold:.6f}")
    
    # Evaluate detection performance
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    print("\n📈 Detection Results:")
    print("-" * 60)
    print(f"{'Sample':<8} {'Error':<12} {'True Label':<12} {'Predicted':<12} {'Status':<12}")
    print("-" * 60)
    
    for i, error in enumerate(reconstruction_errors):
        predicted_anomaly = error > threshold
        actual_anomaly = test_labels[i] == 1
        
        if actual_anomaly and predicted_anomaly:
            true_positives += 1
            status = "✅ DETECTED"
        elif actual_anomaly and not predicted_anomaly:
            false_negatives += 1
            status = "❌ MISSED"
        elif not actual_anomaly and predicted_anomaly:
            false_positives += 1
            status = "⚠️ FALSE ALARM"
        else:
            true_negatives += 1
            status = "✅ NORMAL"
        
        if i < 10:  # Show first 10 samples
            true_label = "ANOMALY" if actual_anomaly else "NORMAL"
            predicted = "ANOMALY" if predicted_anomaly else "NORMAL"
            print(f"{i:<8} {error:<12.6f} {true_label:<12} {predicted:<12} {status:<12}")
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(test_data)
    
    print("\n📊 Performance Metrics:")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}") 
    print(f"F1-Score:  {f1_score:.3f}")
    print(f"\nTrue Positives:  {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives:  {true_negatives}")
    print(f"False Negatives: {false_negatives}")
    
    return model, threshold

if __name__ == "__main__":
    model, threshold = train_anomaly_detector()