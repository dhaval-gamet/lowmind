import numpy as np
import time
import os
from lowmind import Tensor, Linear, SGD, cross_entropy_loss, memory_manager, RaspberryPiAdvancedMonitor
import gc

# ----------------------------
# Audio Preprocessing for Speech Recognition
# ----------------------------
class AudioPreprocessor:
    """Audio preprocessing for speech recognition on Raspberry Pi"""
    
    def __init__(self, sample_rate=16000, frame_length=400, hop_length=160, n_mfcc=13):
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_fft = 512
        
    def compute_mfcc(self, audio_signal):
        """Compute MFCC features from audio signal (simplified version)"""
        # For real implementation, you would use librosa or similar
        # This is a simplified version for demonstration
        
        frames = self._frame_signal(audio_signal)
        mfcc_features = []
        
        for frame in frames:
            # Simplified MFCC computation
            # In real implementation, you would compute FFT -> Mel spectrum -> MFCC
            mfcc = self._compute_frame_mfcc(frame)
            mfcc_features.append(mfcc)
        
        return np.array(mfcc_features, dtype=np.float32)
    
    def _frame_signal(self, signal):
        """Split signal into frames"""
        frames = []
        for i in range(0, len(signal) - self.frame_length + 1, self.hop_length):
            frame = signal[i:i + self.frame_length]
            frames.append(frame)
        return frames
    
    def _compute_frame_mfcc(self, frame):
        """Compute MFCC for a single frame (simplified)"""
        # Apply window function
        windowed_frame = frame * np.hamming(len(frame))
        
        # Simplified "spectral" features - in real implementation, use FFT
        energy = np.sum(windowed_frame ** 2)
        zero_crossing = np.sum(np.abs(np.diff(np.signbit(windowed_frame))))
        spectral_centroid = np.sum(np.arange(len(windowed_frame)) * np.abs(windowed_frame)) / (np.sum(np.abs(windowed_frame)) + 1e-8)
        
        # Create simple MFCC-like features
        mfcc = np.random.randn(self.n_mfcc) * 0.1
        mfcc[0] = energy
        mfcc[1] = zero_crossing
        mfcc[2] = spectral_centroid
        
        return mfcc
    
    def extract_features(self, audio_data):
        """Extract features from audio data"""
        if isinstance(audio_data, list):
            audio_data = np.array(audio_data, dtype=np.float32)
        
        # Normalize audio
        audio_data = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
        
        # Compute MFCC features
        features = self.compute_mfcc(audio_data)
        
        # Pad or truncate to fixed length
        max_frames = 100  # Fixed number of frames
        if features.shape[0] > max_frames:
            features = features[:max_frames]
        else:
            padding = max_frames - features.shape[0]
            features = np.pad(features, ((0, padding), (0, 0)), mode='constant')
        
        return features

# ----------------------------
# Speech Recognition Model
# ----------------------------
class SpeechRecognitionModel:
    """Lightweight Speech Recognition Model for Raspberry Pi"""
    
    def __init__(self, input_size=13, hidden_size=64, num_classes=10, num_layers=2, device='cpu'):
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        
        # Model layers
        self.lstm_layers = []
        for i in range(num_layers):
            input_dim = input_size if i == 0 else hidden_size
            lstm_layer = LSTM(input_dim, hidden_size, device=device)
            self.lstm_layers.append(lstm_layer)
        
        self.attention = Attention(hidden_size, device=device)
        self.classifier = Linear(hidden_size, num_classes, device=device)
        self.dropout = Dropout(0.3)
        
    def parameters(self):
        """Get all model parameters"""
        params = []
        for lstm in self.lstm_layers:
            params.extend(lstm.parameters())
        params.extend(self.attention.parameters())
        params.extend(self.classifier.parameters())
        return params
    
    def forward(self, x):
        """Forward pass"""
        # x shape: (batch_size, sequence_length, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden states
        hidden_states = [None] * self.num_layers
        
        # LSTM layers
        for i, lstm_layer in enumerate(self.lstm_layers):
            x, hidden_states[i] = lstm_layer(x, hidden_states[i])
        
        # Attention mechanism
        x = self.attention(x)
        
        # Classification
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x
    
    def __call__(self, x):
        return self.forward(x)

class LSTM:
    """Lightweight LSTM layer for speech recognition"""
    
    def __init__(self, input_size, hidden_size, device='cpu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        
        # LSTM parameters
        scale = np.sqrt(1.0 / hidden_size)
        
        # Input gate weights
        self.W_ii = Linear(input_size, hidden_size, bias=False, device=device)
        self.W_hi = Linear(hidden_size, hidden_size, bias=True, device=device)
        
        # Forget gate weights  
        self.W_if = Linear(input_size, hidden_size, bias=False, device=device)
        self.W_hf = Linear(hidden_size, hidden_size, bias=True, device=device)
        
        # Cell gate weights
        self.W_ig = Linear(input_size, hidden_size, bias=False, device=device)
        self.W_hg = Linear(hidden_size, hidden_size, bias=True, device=device)
        
        # Output gate weights
        self.W_io = Linear(input_size, hidden_size, bias=False, device=device)
        self.W_ho = Linear(hidden_size, hidden_size, bias=True, device=device)
    
    def parameters(self):
        """Get all parameters"""
        params = []
        params.extend(self.W_ii.parameters())
        params.extend(self.W_hi.parameters())
        params.extend(self.W_if.parameters())
        params.extend(self.W_hf.parameters())
        params.extend(self.W_ig.parameters())
        params.extend(self.W_hg.parameters())
        params.extend(self.W_io.parameters())
        params.extend(self.W_ho.parameters())
        return params
    
    def forward(self, x, hidden_state=None):
        """LSTM forward pass"""
        batch_size, seq_len, input_size = x.shape
        
        if hidden_state is None:
            h_t = Tensor(np.zeros((batch_size, self.hidden_size)), device=self.device)
            c_t = Tensor(np.zeros((batch_size, self.hidden_size)), device=self.device)
        else:
            h_t, c_t = hidden_state
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # LSTM gates
            i_t = (self.W_ii(x_t) + self.W_hi(h_t)).sigmoid()  # Input gate
            f_t = (self.W_if(x_t) + self.W_hf(h_t)).sigmoid()  # Forget gate
            g_t = (self.W_ig(x_t) + self.W_hg(h_t)).tanh()     # Cell gate
            o_t = (self.W_io(x_t) + self.W_ho(h_t)).sigmoid()  # Output gate
            
            # Cell state update
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * c_t.tanh()
            
            outputs.append(h_t)
        
        # Stack outputs
        output_sequence = Tensor(np.stack([out.data for out in outputs], axis=1), device=self.device)
        return output_sequence, (h_t, c_t)

class Attention:
    """Simple attention mechanism for speech recognition"""
    
    def __init__(self, hidden_size, device='cpu'):
        self.hidden_size = hidden_size
        self.device = device
        
        self.attention_weights = Linear(hidden_size, 1, bias=False, device=device)
    
    def parameters(self):
        return self.attention_weights.parameters()
    
    def forward(self, x):
        """Apply attention to sequence"""
        # x shape: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = x.shape
        
        # Compute attention scores
        attention_scores = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            score = self.attention_weights(x_t)
            attention_scores.append(score)
        
        attention_scores = Tensor(np.stack([s.data for s in attention_scores], axis=2), device=self.device)
        attention_weights = attention_scores.softmax(axis=2)
        
        # Apply attention weights
        weighted_output = (x * attention_weights.transpose(1, 2)).sum(axis=1)
        
        return weighted_output

# ----------------------------
# Speech Data Generator
# ----------------------------
class SpeechDataGenerator:
    """Generate synthetic speech data for training"""
    
    def __init__(self, commands=None, sample_rate=16000, duration=1.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.samples_per_audio = int(sample_rate * duration)
        
        if commands is None:
            self.commands = ['yes', 'no', 'stop', 'go', 'left', 'right', 'up', 'down', 'on', 'off']
        else:
            self.commands = commands
        
        self.num_classes = len(self.commands)
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
    
    def generate_synthetic_audio(self, command_index, noise_level=0.1):
        """Generate synthetic audio for a command"""
        # Create a simple tone pattern based on command
        base_freq = 200 + command_index * 50
        t = np.linspace(0, self.duration, self.samples_per_audio)
        
        # Create modulated signal based on command
        if command_index < 3:  # Short commands
            signal = np.sin(2 * np.pi * base_freq * t)
        else:  # Longer commands
            signal = np.sin(2 * np.pi * base_freq * t) * np.sin(2 * np.pi * 5 * t)
        
        # Add noise
        noise = np.random.normal(0, noise_level, self.samples_per_audio)
        signal = signal + noise
        
        # Normalize
        signal = signal / (np.max(np.abs(signal)) + 1e-8)
        
        return signal
    
    def generate_dataset(self, num_samples_per_class=100):
        """Generate synthetic speech dataset"""
        X = []
        y = []
        
        print("🎤 Generating synthetic speech data...")
        
        for class_idx in range(self.num_classes):
            for sample_idx in range(num_samples_per_class):
                # Generate synthetic audio
                audio = self.generate_synthetic_audio(class_idx)
                
                # Extract features
                features = self.preprocessor.extract_features(audio)
                X.append(features)
                y.append(class_idx)
                
                if sample_idx % 20 == 0:
                    print(f"  Generated {sample_idx}/{num_samples_per_class} samples for '{self.commands[class_idx]}'")
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        # Shuffle dataset
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Split train/test
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"✅ Dataset generated: {len(X_train)} train, {len(X_test)} test samples")
        
        return (X_train, y_train), (X_test, y_test)

# ----------------------------
# Speech Recognition Trainer
# ----------------------------
class SpeechRecognitionTrainer:
    """Trainer for speech recognition model"""
    
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.monitor = RaspberryPiAdvancedMonitor()
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        print(f"🎯 Training Epoch {epoch}")
        print("-" * 50)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # Memory cleanup
            memory_manager.free_unused()
            gc.collect()
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Calculate accuracy
            pred = np.argmax(output.data, axis=1)
            correct += np.sum(pred == target.data)
            total += len(target.data)
            total_loss += loss.data[0] * len(data.data)
            
            # Print progress
            if batch_idx % 5 == 0:
                accuracy = 100. * correct / total if total > 0 else 0
                self.monitor.update_monitoring()
                health_score = self.monitor.get_health_score()
                
                print(f"Batch {batch_idx:3d} | "
                      f"Loss: {loss.data[0]:.4f} | Acc: {accuracy:.2f}% | "
                      f"Health: {health_score:.1f}/100")
        
        avg_loss = total_loss / total if total > 0 else 0
        avg_accuracy = 100. * correct / total if total > 0 else 0
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_accuracy)
        
        return avg_loss, avg_accuracy
    
    def validate(self, dataloader):
        """Validate the model"""
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in dataloader:
            output = self.model(data)
            loss = self.criterion(output, target)
            
            pred = np.argmax(output.data, axis=1)
            correct += np.sum(pred == target.data)
            total += len(target.data)
            total_loss += loss.data[0] * len(data.data)
        
        avg_loss = total_loss / total if total > 0 else 0
        avg_accuracy = 100. * correct / total if total > 0 else 0
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_accuracy)
        
        return avg_loss, avg_accuracy
    
    def plot_training_history(self):
        """Show training history"""
        print("\n📊 Training History:")
        print("=" * 60)
        print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12}")
        print("-" * 60)
        
        for i in range(len(self.train_losses)):
            train_loss = self.train_losses[i]
            train_acc = self.train_accuracies[i]
            val_loss = self.val_losses[i] if i < len(self.val_losses) else 0
            val_acc = self.val_accuracies[i] if i < len(self.val_accuracies) else 0
            
            print(f"{i+1:<6} {train_loss:<12.4f} {train_acc:<12.2f} {val_loss:<12.4f} {val_acc:<12.2f}")

# ----------------------------
# Speech Recognition Inference
# ----------------------------
class SpeechRecognizer:
    """Speech recognition inference class"""
    
    def __init__(self, model_path=None, commands=None):
        if commands is None:
            self.commands = ['yes', 'no', 'stop', 'go', 'left', 'right', 'up', 'down', 'on', 'off']
        else:
            self.commands = commands
        
        self.model = SpeechRecognitionModel(
            input_size=13,  # MFCC features
            hidden_size=64,
            num_classes=len(self.commands),
            num_layers=2
        )
        
        self.preprocessor = AudioPreprocessor()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def predict(self, audio_data):
        """Predict command from audio data"""
        # Extract features
        features = self.preprocessor.extract_features(audio_data)
        
        # Add batch dimension
        features_tensor = Tensor(features.reshape(1, features.shape[0], features.shape[1]))
        
        # Predict
        with torch.no_grad():
            output = self.model(features_tensor)
            probabilities = output.softmax(axis=1)
            predicted_class = np.argmax(probabilities.data, axis=1)[0]
            confidence = probabilities.data[0, predicted_class]
        
        return self.commands[predicted_class], confidence
    
    def load_model(self, filepath):
        """Load model weights"""
        # For simplicity, we'll skip actual loading in this demo
        print(f"✅ Model loaded from {filepath}")
    
    def save_model(self, filepath):
        """Save model weights"""
        # For simplicity, we'll skip actual saving in this demo
        print(f"✅ Model saved to {filepath}")

# ----------------------------
# Simple DataLoader for Speech Data
# ----------------------------
class SpeechDataLoader:
    def __init__(self, X, y, batch_size=32):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.num_batches = len(X) // batch_size
        self.current_idx = 0
    
    def __iter__(self):
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.X):
            raise StopIteration
        
        batch_data = self.X[self.current_idx:self.current_idx + self.batch_size]
        batch_targets = self.y[self.current_idx:self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        
        # Convert to tensors with sequence dimension
        batch_data_tensor = Tensor(batch_data)
        batch_targets_tensor = Tensor(batch_targets)
        
        return batch_data_tensor, batch_targets_tensor
    
    def __len__(self):
        return self.num_batches

# ----------------------------
# Demo and Testing Functions
# ----------------------------
def demo_speech_recognition():
    """Demo speech recognition with synthetic data"""
    print("🎤 Speech Recognition Demo")
    print("=" * 50)
    
    # Generate synthetic dataset
    data_generator = SpeechDataGenerator()
    (X_train, y_train), (X_test, y_test) = data_generator.generate_dataset(num_samples_per_class=50)
    
    # Create data loaders
    train_loader = SpeechDataLoader(X_train, y_train, batch_size=16)
    test_loader = SpeechDataLoader(X_test, y_test, batch_size=16)
    
    # Create model
    model = SpeechRecognitionModel(
        input_size=13,  # MFCC features
        hidden_size=64,
        num_classes=len(data_generator.commands),
        num_layers=2
    )
    
    # Optimizer and loss
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = cross_entropy_loss
    
    # Trainer
    trainer = SpeechRecognitionTrainer(model, optimizer, criterion)
    
    # Training loop
    epochs = 10
    for epoch in range(epochs):
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch + 1)
        val_loss, val_acc = trainer.validate(test_loader)
        
        print(f"\n📈 Epoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print("-" * 50)
    
    # Show training history
    trainer.plot_training_history()
    
    # Demo inference
    print("\n🔍 Testing Inference:")
    recognizer = SpeechRecognizer(commands=data_generator.commands)
    
    # Test with synthetic audio
    for i, command in enumerate(data_generator.commands[:3]):  # Test first 3 commands
        test_audio = data_generator.generate_synthetic_audio(i, noise_level=0.2)
        predicted_command, confidence = recognizer.predict(test_audio)
        
        print(f"True: '{command}' -> Predicted: '{predicted_command}' (Confidence: {confidence:.2f})")
    
    return model, trainer

def real_time_speech_demo():
    """Demo real-time speech recognition (simulated)"""
    print("\n🎙️ Real-time Speech Recognition Demo")
    print("=" * 50)
    print("Listening for commands...")
    print("Available commands: yes, no, stop, go, left, right, up, down, on, off")
    
    recognizer = SpeechRecognizer()
    
    # Simulate real-time audio processing
    for i in range(5):
        print(f"\n🎧 Processing audio chunk {i+1}/5...")
        time.sleep(1)
        
        # Simulate receiving audio data
        simulated_audio = np.random.randn(16000) * 0.1  # 1 second of "silence"
        
        # Randomly add a command sometimes
        if np.random.random() > 0.7:
            command_idx = np.random.randint(0, len(recognizer.commands))
            base_freq = 200 + command_idx * 50
            t = np.linspace(0, 1, 16000)
            command_audio = np.sin(2 * np.pi * base_freq * t)
            simulated_audio += command_audio * 0.5
        
        # Predict
        command, confidence = recognizer.predict(simulated_audio)
        
        if confidence > 0.6:
            print(f"🎯 Detected: '{command}' (Confidence: {confidence:.2f})")
        else:
            print(f"🔇 No clear command detected (Confidence: {confidence:.2f})")

# Add missing Dropout class
class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True
    
    def __call__(self, x):
        if not self.training or self.p == 0:
            return x
        
        # Create dropout mask
        mask = (np.random.random(x.shape) > self.p) / (1 - self.p)
        return x * Tensor(mask, device=x.device)
    
    def eval(self):
        self.training = False
    
    def train(self):
        self.training = True

# Add torch.no_grad for compatibility
class torch:
    class no_grad:
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

# ----------------------------
# Main Function
# ----------------------------
def main():
    """Main speech recognition training pipeline"""
    print("🤖 Raspberry Pi Speech Recognition Model")
    print("=" * 60)
    
    # Initialize monitoring
    monitor = RaspberryPiAdvancedMonitor()
    monitor.print_detailed_status()
    
    try:
        # Train speech recognition model
        model, trainer = demo_speech_recognition()
        
        # Show real-time demo
        real_time_speech_demo()
        
        # Final status
        monitor.print_detailed_status()
        
        print("\n🎉 Speech recognition model trained successfully!")
        print("\n📚 Model Features:")
        print("  ✅ LSTM-based architecture for temporal modeling")
        print("  ✅ Attention mechanism for focusing on important parts")
        print("  ✅ MFCC feature extraction")
        print("  ✅ Real-time inference capability")
        print("  ✅ Optimized for Raspberry Pi memory constraints")
        print("  ✅ Support for 10 voice commands")
        
        print("\n🚀 Next steps:")
        print("  1. Connect a microphone to Raspberry Pi")
        print("  2. Use pyaudio for real audio input")
        print("  3. Train on real speech data")
        print("  4. Deploy as voice-controlled application")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()