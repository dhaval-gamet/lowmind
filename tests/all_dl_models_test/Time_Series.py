import numpy as np
import time
import math
from lowmind import Tensor, Linear, SGD, memory_manager, RaspberryPiAdvancedMonitor

# ----------------------------
# Time Series Data Generator
# ----------------------------
class TimeSeriesDataGenerator:
    def __init__(self, sequence_length=10, prediction_horizon=1):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
    
    def generate_sine_wave(self, num_samples=1000, noise_level=0.1):
        """Generate sine wave time series data with noise"""
        t = np.linspace(0, 4 * np.pi, num_samples + self.sequence_length + self.prediction_horizon)
        sine_wave = np.sin(t) + noise_level * np.random.randn(len(t))
        
        X, y = [], []
        for i in range(len(sine_wave) - self.sequence_length - self.prediction_horizon + 1):
            X.append(sine_wave[i:i + self.sequence_length])
            y.append(sine_wave[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def generate_trend_data(self, num_samples=1000, trend=0.01, seasonality=0.5, noise_level=0.05):
        """Generate trend + seasonality time series data"""
        t = np.arange(num_samples + self.sequence_length + self.prediction_horizon)
        
        # Trend + Seasonality + Noise
        trend_component = trend * t
        seasonal_component = seasonality * np.sin(2 * np.pi * t / 50)
        noise_component = noise_level * np.random.randn(len(t))
        
        series = trend_component + seasonal_component + noise_component
        
        X, y = [], []
        for i in range(len(series) - self.sequence_length - self.prediction_horizon + 1):
            X.append(series[i:i + self.sequence_length])
            y.append(series[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def generate_multivariate_series(self, num_samples=1000):
        """Generate multivariate time series data"""
        t = np.linspace(0, 4 * np.pi, num_samples + self.sequence_length + self.prediction_horizon)
        
        # Multiple correlated time series
        series1 = np.sin(t)
        series2 = np.cos(t)
        series3 = np.sin(2 * t) * 0.5
        
        multivariate_series = np.column_stack([series1, series2, series3])
        
        X, y = [], []
        for i in range(len(multivariate_series) - self.sequence_length - self.prediction_horizon + 1):
            X.append(multivariate_series[i:i + self.sequence_length])
            y.append(multivariate_series[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon, 0])  # Predict first series
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ----------------------------
# Time Series Models
# ----------------------------
class SimpleRNN:
    """Simple RNN for time series forecasting"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device='cpu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device
        
        # RNN weights
        self.W_xh = Tensor(np.random.randn(hidden_size, input_size) * 0.01, 
                          requires_grad=True, device=device, name="W_xh")
        self.W_hh = Tensor(np.random.randn(hidden_size, hidden_size) * 0.01,
                          requires_grad=True, device=device, name="W_hh")
        self.b_h = Tensor(np.zeros(hidden_size), requires_grad=True, device=device, name="b_h")
        
        # Output layer
        self.W_hy = Tensor(np.random.randn(output_size, hidden_size) * 0.01,
                          requires_grad=True, device=device, name="W_hy")
        self.b_y = Tensor(np.zeros(output_size), requires_grad=True, device=device, name="b_y")
    
    def parameters(self):
        return [self.W_xh, self.W_hh, self.b_h, self.W_hy, self.b_y]
    
    def forward(self, x, hidden_state=None):
        """
        x shape: (batch_size, sequence_length, input_size)
        Returns: (batch_size, output_size)
        """
        batch_size, seq_len, input_size = x.shape
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            h = Tensor(np.zeros((batch_size, self.hidden_size)), device=self.device)
        else:
            h = hidden_state
        
        # Process each time step
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)
            
            # RNN update
            h = (x_t @ self.W_xh.T + h @ self.W_hh.T + self.b_h).tanh()
        
        # Final output
        output = h @ self.W_hy.T + self.b_y
        return output, h
    
    def __call__(self, x, hidden_state=None):
        return self.forward(x, hidden_state)

class LSTMModel:
    """LSTM model for time series forecasting"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, device='cpu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.device = device
        
        # LSTM gates weights
        # Input gate
        self.W_xi = Tensor(np.random.randn(hidden_size, input_size) * 0.01, requires_grad=True, device=device)
        self.W_hi = Tensor(np.random.randn(hidden_size, hidden_size) * 0.01, requires_grad=True, device=device)
        self.b_i = Tensor(np.zeros(hidden_size), requires_grad=True, device=device)
        
        # Forget gate
        self.W_xf = Tensor(np.random.randn(hidden_size, input_size) * 0.01, requires_grad=True, device=device)
        self.W_hf = Tensor(np.random.randn(hidden_size, hidden_size) * 0.01, requires_grad=True, device=device)
        self.b_f = Tensor(np.zeros(hidden_size), requires_grad=True, device=device)
        
        # Cell gate
        self.W_xc = Tensor(np.random.randn(hidden_size, input_size) * 0.01, requires_grad=True, device=device)
        self.W_hc = Tensor(np.random.randn(hidden_size, hidden_size) * 0.01, requires_grad=True, device=device)
        self.b_c = Tensor(np.zeros(hidden_size), requires_grad=True, device=device)
        
        # Output gate
        self.W_xo = Tensor(np.random.randn(hidden_size, input_size) * 0.01, requires_grad=True, device=device)
        self.W_ho = Tensor(np.random.randn(hidden_size, hidden_size) * 0.01, requires_grad=True, device=device)
        self.b_o = Tensor(np.zeros(hidden_size), requires_grad=True, device=device)
        
        # Output layer
        self.W_hy = Tensor(np.random.randn(output_size, hidden_size) * 0.01, requires_grad=True, device=device)
        self.b_y = Tensor(np.zeros(output_size), requires_grad=True, device=device)
    
    def parameters(self):
        return [
            self.W_xi, self.W_hi, self.b_i,
            self.W_xf, self.W_hf, self.b_f,
            self.W_xc, self.W_hc, self.b_c,
            self.W_xo, self.W_ho, self.b_o,
            self.W_hy, self.b_y
        ]
    
    def forward(self, x, hidden_state=None):
        """
        x shape: (batch_size, sequence_length, input_size)
        """
        batch_size, seq_len, input_size = x.shape
        
        # Initialize hidden state and cell state if not provided
        if hidden_state is None:
            h = Tensor(np.zeros((batch_size, self.hidden_size)), device=self.device)
            c = Tensor(np.zeros((batch_size, self.hidden_size)), device=self.device)
        else:
            h, c = hidden_state
        
        # Process each time step
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, input_size)
            
            # LSTM gates
            i = (x_t @ self.W_xi.T + h @ self.W_hi.T + self.b_i).sigmoid()  # Input gate
            f = (x_t @ self.W_xf.T + h @ self.W_hf.T + self.b_f).sigmoid()  # Forget gate
            g = (x_t @ self.W_xc.T + h @ self.W_hc.T + self.b_c).tanh()     # Cell gate
            o = (x_t @ self.W_xo.T + h @ self.W_ho.T + self.b_o).sigmoid()  # Output gate
            
            # Update cell state and hidden state
            c = f * c + i * g
            h = o * c.tanh()
        
        # Final output
        output = h @ self.W_hy.T + self.b_y
        return output, (h, c)
    
    def __call__(self, x, hidden_state=None):
        return self.forward(x, hidden_state)

class CNN1DTimeSeries:
    """1D CNN for time series forecasting"""
    def __init__(self, sequence_length, output_size, num_filters=32, kernel_size=3, device='cpu'):
        self.sequence_length = sequence_length
        self.output_size = output_size
        self.device = device
        
        # 1D CNN layers (implemented using 2D conv with height=1)
        self.conv1 = Conv1D(1, num_filters, kernel_size, padding=1, device=device)
        self.conv2 = Conv1D(num_filters, num_filters * 2, kernel_size, padding=1, device=device)
        
        # Calculate size after convolutions and pooling
        conv_output_size = num_filters * 2 * (sequence_length // 4)
        
        # Fully connected layers
        self.fc1 = Linear(conv_output_size, 50, device=device)
        self.fc2 = Linear(50, output_size, device=device)
        
        self.dropout = Dropout(0.2)
    
    def parameters(self):
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, 1)
        # Reshape for CNN: (batch_size, 1, sequence_length, 1)
        x = x.unsqueeze(1).unsqueeze(-1)
        
        # CNN layers
        x = self.conv1(x).relu()
        x = x.mean(axis=2, keepdims=True)  # Global average pooling along sequence
        
        x = self.conv2(x).relu()
        x = x.mean(axis=2, keepdims=True)
        
        # Flatten
        x = x.reshape(x.shape[0], -1)
        
        # Fully connected layers
        x = self.dropout(x)
        x = self.fc1(x).relu()
        x = self.fc2(x)
        
        return x
    
    def __call__(self, x):
        return self.forward(x)

# ----------------------------
# Custom 1D Convolution Layer
# ----------------------------
class Conv1D:
    """1D Convolution implemented using 2D convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, device='cpu'):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device
        
        # Initialize weights for 2D conv with height=1
        scale = np.sqrt(2.0 / (in_channels * kernel_size))
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, 1, kernel_size).astype(np.float32) * scale,
            requires_grad=True, device=device, name="conv1d_weight"
        )
        self.bias = Tensor(
            np.zeros(out_channels), 
            requires_grad=True, device=device, name="conv1d_bias"
        )
    
    def parameters(self):
        return [self.weight, self.bias]
    
    def forward(self, x):
        # x shape: (batch_size, in_channels, sequence_length, 1)
        batch_size, in_channels, seq_len, _ = x.shape
        
        # Apply 1D convolution using 2D conv
        output = np.zeros((batch_size, self.out_channels, seq_len, 1), dtype=np.float32)
        
        for i in range(seq_len):
            start_idx = max(0, i - self.padding)
            end_idx = min(seq_len, i + self.kernel_size - self.padding)
            
            x_slice = x.data[:, :, start_idx:end_idx, :]
            if x_slice.shape[2] < self.kernel_size:
                # Pad if necessary
                pad_width = ((0, 0), (0, 0), (0, self.kernel_size - x_slice.shape[2]), (0, 0))
                x_slice = np.pad(x_slice, pad_width, mode='constant')
            
            for k in range(self.out_channels):
                output[:, k, i, 0] = np.sum(x_slice * self.weight.data[k]) + self.bias.data[k]
        
        return Tensor(output, device=self.device)
    
    def __call__(self, x):
        return self.forward(x)

# ----------------------------
# Time Series Trainer
# ----------------------------
class TimeSeriesTrainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.monitor = RaspberryPiAdvancedMonitor()
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, X_train, y_train, epoch, batch_size=32):
        """Train for one epoch"""
        total_loss = 0
        num_batches = 0
        
        print(f"📈 Training Time Series Epoch {epoch}")
        print("-" * 50)
        
        # Shuffle data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        for i in range(0, len(X_train), batch_size):
            # Memory cleanup
            memory_manager.free_unused()
            
            # Get batch
            end_idx = min(i + batch_size, len(X_train))
            X_batch = Tensor(X_shuffled[i:end_idx])
            y_batch = Tensor(y_shuffled[i:end_idx])
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions, _ = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.data[0]
            num_batches += 1
            
            # Print progress
            if num_batches % 10 == 0:
                self.monitor.update_monitoring()
                health_score = self.monitor.get_health_score()
                
                print(f"Batch {num_batches:3d} | Loss: {loss.data[0]:.4f} | Health: {health_score:.1f}/100")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, X_val, y_val):
        """Validate the model"""
        total_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_val), 32):
            X_batch = Tensor(X_val[i:i+32])
            y_batch = Tensor(y_val[i:i+32])
            
            predictions, _ = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            total_loss += loss.data[0]
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def forecast(self, model, initial_sequence, steps=50):
        """Generate multi-step forecasts"""
        current_sequence = initial_sequence.copy()
        forecasts = []
        
        for step in range(steps):
            # Prepare input
            input_seq = Tensor(current_sequence.reshape(1, -1, 1))
            
            # Predict next value
            with torch.no_grad():
                prediction, _ = model(input_seq)
                next_value = prediction.data[0, 0]
            
            forecasts.append(next_value)
            
            # Update sequence (remove first, add prediction)
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_value
        
        return np.array(forecasts)

# ----------------------------
# Evaluation Metrics
# ----------------------------
def calculate_metrics(true_values, predictions):
    """Calculate time series evaluation metrics"""
    true_values = np.array(true_values).flatten()
    predictions = np.array(predictions).flatten()
    
    # Mean Absolute Error
    mae = np.mean(np.abs(true_values - predictions))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((true_values - predictions) ** 2))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((true_values - predictions) / (true_values + 1e-8))) * 100
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }

# ----------------------------
# Example 1: Sine Wave Prediction
# ----------------------------
def train_sine_wave_model():
    """Train model on sine wave data"""
    print("🌊 Sine Wave Time Series Prediction")
    print("=" * 50)
    
    # Generate data
    data_gen = TimeSeriesDataGenerator(sequence_length=20, prediction_horizon=1)
    X, y = data_gen.generate_sine_wave(1000, noise_level=0.1)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Reshape for RNN (add feature dimension)
    X_train = X_train.reshape(-1, 20, 1)
    X_val = X_val.reshape(-1, 20, 1)
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    
    # Model
    model = SimpleRNN(input_size=1, hidden_size=32, output_size=1, device='cpu')
    optimizer = SGD(model.parameters(), lr=0.01)
    
    # Trainer
    trainer = TimeSeriesTrainer(model, optimizer, mse_loss)
    
    # Training
    epochs = 20
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(X_train, y_train, epoch + 1, batch_size=32)
        val_loss = trainer.validate(X_val, y_val)
        
        print(f"📊 Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # Test forecasting
    test_sequence = X_val[0:1]  # Get first validation sequence
    forecasts = trainer.forecast(model, test_sequence[0].flatten(), steps=30)
    
    print(f"\n✅ Sine wave model training completed!")
    print(f"Final train loss: {trainer.train_losses[-1]:.4f}")
    print(f"Final val loss: {trainer.val_losses[-1]:.4f}")
    
    return model, trainer, (X_train, y_train, X_val, y_val)

# ----------------------------
# Example 2: Trend + Seasonality Prediction
# ----------------------------
def train_trend_model():
    """Train model on trend + seasonality data"""
    print("\n📈 Trend + Seasonality Time Series Prediction")
    print("=" * 50)
    
    # Generate data
    data_gen = TimeSeriesDataGenerator(sequence_length=30, prediction_horizon=1)
    X, y = data_gen.generate_trend_data(1500, trend=0.02, seasonality=0.8, noise_level=0.1)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Reshape for RNN
    X_train = X_train.reshape(-1, 30, 1)
    X_val = X_val.reshape(-1, 30, 1)
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    
    # Model - Use LSTM for better trend capture
    model = LSTMModel(input_size=1, hidden_size=64, output_size=1, device='cpu')
    optimizer = SGD(model.parameters(), lr=0.005, momentum=0.9)
    
    # Trainer
    trainer = TimeSeriesTrainer(model, optimizer, mse_loss)
    
    # Training
    epochs = 25
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(X_train, y_train, epoch + 1, batch_size=32)
        val_loss = trainer.validate(X_val, y_val)
        
        if epoch % 5 == 0:
            print(f"📊 Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    print(f"\n✅ Trend model training completed!")
    print(f"Final train loss: {trainer.train_losses[-1]:.4f}")
    print(f"Final val loss: {trainer.val_losses[-1]:.4f}")
    
    return model, trainer

# ----------------------------
# Example 3: Multivariate Time Series
# ----------------------------
def train_multivariate_model():
    """Train model on multivariate time series data"""
    print("\n🔄 Multivariate Time Series Prediction")
    print("=" * 50)
    
    # Generate multivariate data
    data_gen = TimeSeriesDataGenerator(sequence_length=15, prediction_horizon=1)
    X, y = data_gen.generate_multivariate_series(1200)
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # y is already shaped correctly (batch_size, 1)
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    
    # Model - input_size = 3 (three features)
    model = SimpleRNN(input_size=3, hidden_size=48, output_size=1, device='cpu')
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Trainer
    trainer = TimeSeriesTrainer(model, optimizer, mse_loss)
    
    # Training
    epochs = 15
    for epoch in range(epochs):
        train_loss = trainer.train_epoch(X_train, y_train, epoch + 1, batch_size=32)
        val_loss = trainer.validate(X_val, y_val)
        
        if epoch % 3 == 0:
            print(f"📊 Epoch {epoch+1:2d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    print(f"\n✅ Multivariate model training completed!")
    print(f"Final train loss: {trainer.train_losses[-1]:.4f}")
    print(f"Final val loss: {trainer.val_losses[-1]:.4f}")
    
    return model, trainer

# ----------------------------
# Real-time Prediction Demo
# ----------------------------
def real_time_prediction_demo():
    """Demo of real-time time series prediction"""
    print("\n🎯 Real-time Time Series Prediction Demo")
    print("=" * 50)
    
    # Create a simple streaming data simulator
    class DataStream:
        def __init__(self):
            self.t = 0
            self.base_frequency = 0.1
        
        def get_next_data_point(self):
            # Generate streaming data with some pattern
            value = (np.sin(self.t * self.base_frequency) + 
                   0.5 * np.sin(self.t * self.base_frequency * 3) +
                   0.1 * np.random.randn())
            self.t += 1
            return value
    
    # Initialize
    stream = DataStream()
    window_size = 20
    data_window = []
    
    # Simple model for real-time prediction
    model = SimpleRNN(input_size=1, hidden_size=16, output_size=1, device='cpu')
    
    print("Starting real-time prediction simulation...")
    print("Time | Actual | Prediction | Error")
    print("-" * 40)
    
    for step in range(50):
        # Get new data point
        new_point = stream.get_next_data_point()
        data_window.append(new_point)
        
        # Keep fixed window size
        if len(data_window) > window_size:
            data_window.pop(0)
        
        # Make prediction when we have enough data
        if len(data_window) == window_size:
            input_data = Tensor(np.array(data_window).reshape(1, window_size, 1))
            
            with torch.no_grad():
                prediction, _ = model(input_data)
                pred_value = prediction.data[0, 0]
            
            # Next actual value (we're cheating a bit for demo)
            next_actual = stream.get_next_data_point()
            error = abs(pred_value - next_actual)
            
            print(f"{step:4d} | {next_actual:6.3f} | {pred_value:9.3f} | {error:5.3f}")
            
            # Add the actual value to window for next prediction
            data_window.append(next_actual)
            if len(data_window) > window_size:
                data_window.pop(0)
    
    print("Real-time demo completed!")

# ----------------------------
# Main Time Series Training Pipeline
# ----------------------------
def main():
    """Main time series training pipeline"""
    print("🤖 Raspberry Pi Time Series Forecasting")
    print("=" * 60)
    
    # Initialize monitoring
    monitor = RaspberryPiAdvancedMonitor()
    monitor.print_detailed_status()
    
    try:
        print("\n" + "🚀 Starting Time Series Model Training".center(50, "="))
        
        # 1. Sine Wave Prediction
        sine_model, sine_trainer, sine_data = train_sine_wave_model()
        
        # 2. Trend Prediction
        trend_model, trend_trainer = train_trend_model()
        
        # 3. Multivariate Prediction
        multi_model, multi_trainer = train_multivariate_model()
        
        # 4. Real-time Demo
        real_time_prediction_demo()
        
        # Final status
        monitor.print_detailed_status()
        
        print("\n🎉 All time series models trained successfully!")
        print("\n📚 Time Series Model Summary:")
        print("  ✅ Sine Wave Prediction (Simple RNN)")
        print("  ✅ Trend + Seasonality Prediction (LSTM)") 
        print("  ✅ Multivariate Time Series Prediction")
        print("  ✅ Real-time Prediction Demo")
        print("  ✅ All models optimized for Raspberry Pi")
        
        print("\n🔮 Use Cases:")
        print("  • Stock price prediction")
        print("  • Weather forecasting")
        print("  • Sensor data analysis")
        print("  • Energy consumption forecasting")
        print("  • IoT device monitoring")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

# Add missing components
def mse_loss(output, target):
    """Mean squared error loss for time series"""
    return ((output - target) ** 2).mean()

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
    
    def __call__(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (np.random.random(x.shape) > self.p) / (1 - self.p)
        return x * Tensor(mask, device=x.device)
    
    def eval(self):
        self.training = False
    
    def train(self):
        self.training = True

class torch:
    class no_grad:
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

if __name__ == "__main__":
    main()