import lowmind as lm
import numpy as np

# सरल न्यूरल नेटवर्क बनाएं
class SimpleNN(lm.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = lm.Linear(784, 128)  # 28x28 पिक्सेल इनपुट
        self.fc2 = lm.Linear(128, 64)
        self.fc3 = lm.Linear(64, 10)    # 10 क्लासेस (0-9 डिजिट)
        self.dropout = lm.Dropout(0.2)
    
    def forward(self, x):
        x = x.reshape(-1, 784)  # फ्लैटन करें
        x = self.fc1(x).relu()
        x = self.dropout(x)
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x

# ट्रेनिंग फंक्शन
def train_simple_model():
    print("🚀 LowMind के साथ सरल मॉडल ट्रेनिंग शुरू...")
    
    # मॉडल और ऑप्टिमाइज़र बनाएं
    model = SimpleNN()
    optimizer = lm.SGD(model.parameters(), lr=0.01)
    
    # सिंपल डेटा जेनरेट करें (डमी डेटा)
    batch_size = 32
    x_train = lm.Tensor(np.random.randn(batch_size, 1, 28, 28))
    y_train = lm.Tensor(np.random.randint(0, 10, (batch_size,)))
    
    print(f"📊 डेटा शेप: {x_train.shape}")
    print(f"🎯 लेबल शेप: {y_train.shape}")
    
    # ट्रेनिंग लूप
    epochs = 3
    for epoch in range(epochs):
        # फॉरवर्ड पास
        outputs = model(x_train)
        loss = lm.cross_entropy_loss(outputs, y_train)
        
        # बैकवर्ड पास
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"📈 Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

# प्रेडिक्शन फंक्शन
def predict_digit(model, image_data):
    """नए डिजिट की प्रेडिक्शन करें"""
    model.eval()  # एवल्यूएशन मोड
    
    # इनपुट तैयार करें
    if isinstance(image_data, np.ndarray):
        input_tensor = lm.Tensor(image_data)
    else:
        input_tensor = lm.Tensor(np.array(image_data))
    
    # प्रेडिक्शन करें
    with lm.memory_trace("Prediction"):
        output = model(input_tensor)
        probabilities = lm.softmax(output)
        predicted_class = np.argmax(probabilities.data)
    
    return predicted_class, probabilities.data

# सॉफ्टमैक्स फंक्शन (अतिरिक्त)
def softmax(x):
    """सॉफ्टमैक्स फंक्शन"""
    exp_x = np.exp(x.data - np.max(x.data))
    return lm.Tensor(exp_x / np.sum(exp_x, axis=-1, keepdims=True))

# मुख्य प्रोग्राम
if __name__ == "__main__":
    print("=" * 50)
    print("          LowMind डेमो प्रोग्राम")
    print("=" * 50)
    
    # सिस्टम स्टेटस चेक करें
    monitor = lm.RaspberryPiAdvancedMonitor()
    monitor.print_detailed_status()
    
    # मॉडल ट्रेन करें
    trained_model = train_simple_model()
    
    print("\n" + "=" * 50)
    print("           प्रेडिक्शन टेस्ट")
    print("=" * 50)
    
    # टेस्ट प्रेडिक्शन करें
    test_image = np.random.randn(1, 28, 28)  # डमी टेस्ट इमेज
    predicted_digit, probabilities = predict_digit(trained_model, test_image)
    
    print(f"🔮 प्रेडिक्टेड डिजिट: {predicted_digit}")
    print(f"📊 प्रोबेबिलिटीज: {probabilities}")
    
    # फाइनल मेमोरी स्टेटस
    print("\n" + "=" * 50)
    print("           फाइनल स्टेटस")
    print("=" * 50)
    monitor.update_monitoring()
    monitor.print_detailed_status()
    
    health_score = monitor.get_health_score()
    print(f"🏥 फाइनल हेल्थ स्कोर: {health_score:.1f}/100")
    
    print("\n🎉 प्रोग्राम सफलतापूर्वक पूरा हुआ!")