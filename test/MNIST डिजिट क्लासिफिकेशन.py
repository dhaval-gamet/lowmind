import lowmind as lm
import numpy as np

# सरल न्यूरल नेटवर्क बनाएं - FIXED VERSION
class SimpleNN(lm.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = lm.Linear(784, 128)  # 28x28 पिक्सेल इनपुट
        self.fc2 = lm.Linear(128, 64)
        self.fc3 = lm.Linear(64, 10)    # 10 क्लासेस (0-9 डिजिट)
        self.dropout = lm.Dropout(0.2)
    
    def forward(self, x):
        # FIXED: reshape को सही तरीके से use करें
        if hasattr(x, 'data'):
            # Tensor object है
            x_reshaped = x.reshape((-1, 784))  # tuple में pass करें
        else:
            # numpy array है
            x_reshaped = lm.Tensor(x.reshape(-1, 784))
        
        x = self.fc1(x_reshaped).relu()
        x = self.dropout(x)
        x = self.fc2(x).relu()
        x = self.fc3(x)
        return x

# सॉफ्टमैक्स फंक्शन
def softmax(x):
    """सॉफ्टमैक्स फंक्शन"""
    if hasattr(x, 'data'):
        # Tensor object
        exp_x = np.exp(x.data - np.max(x.data))
        softmax_val = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return lm.Tensor(softmax_val)
    else:
        # numpy array
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# ट्रेनिंग फंक्शन - IMPROVED VERSION
def train_simple_model():
    print("🚀 LowMind के साथ सरल मॉडल ट्रेनिंग शुरू...")
    
    # मॉडल और ऑप्टिमाइज़र बनाएं
    model = SimpleNN()
    optimizer = lm.SGD(model.parameters(), lr=0.01)
    
    # बेहतर डेटा जेनरेट करें (सही shape के साथ)
    batch_size = 32
    x_train = lm.Tensor(np.random.randn(batch_size, 784))  # सीधे flattened
    y_train = lm.Tensor(np.random.randint(0, 10, (batch_size,)))
    
    print(f"📊 डेटा शेप: {x_train.shape}")
    print(f"🎯 लेबल शेप: {y_train.shape}")
    print(f"🔢 मॉडल पैरामीटर्स: {sum(np.prod(param.data.shape) for param in model.parameters())}")
    
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
        if image_data.ndim == 3:  # (1, 28, 28) जैसा shape
            image_data = image_data.reshape(1, -1)  # flatten करें
        input_tensor = lm.Tensor(image_data)
    else:
        input_tensor = lm.Tensor(np.array(image_data).reshape(1, -1))
    
    # प्रेडिक्शन करें
    with lm.memory_trace("Prediction"):
        output = model(input_tensor)
        probabilities = softmax(output)
        predicted_class = np.argmax(probabilities.data if hasattr(probabilities, 'data') else probabilities)
    
    return predicted_class, probabilities

# टेस्ट फंक्शन
def test_model(model):
    """मॉडल को टेस्ट करें"""
    print("\n🧪 मॉडल टेस्टिंग...")
    
    # टेस्ट डेटा
    test_images = np.random.randn(5, 784)  # 5 टेस्ट इमेजेस
    test_labels = np.random.randint(0, 10, (5,))
    
    correct = 0
    for i, (img, label) in enumerate(zip(test_images, test_labels)):
        pred, probs = predict_digit(model, img)
        
        # सबसे high probability वाला class
        max_prob = np.max(probs.data if hasattr(probs, 'data') else probs)
        
        status = "✅" if pred == label else "❌"
        print(f"{status} टेस्ट {i+1}: प्रेडिक्टेड {pred}, एक्चुअल {label}, कॉन्फिडेंस: {max_prob:.3f}")
        
        if pred == label:
            correct += 1
    
    accuracy = correct / len(test_images) * 100
    print(f"📊 एक्यूरेसी: {accuracy:.1f}%")

# मुख्य प्रोग्राम
if __name__ == "__main__":
    print("=" * 50)
    print("          LowMind डेमो प्रोग्राम - FIXED VERSION")
    print("=" * 50)
    
    # सिस्टम स्टेटस चेक करें
    monitor = lm.RaspberryPiAdvancedMonitor()
    monitor.print_detailed_status()
    
    try:
        # मॉडल ट्रेन करें
        trained_model = train_simple_model()
        
        print("\n" + "=" * 50)
        print("           प्रेडिक्शन टेस्ट")
        print("=" * 50)
        
        # टेस्ट प्रेडिक्शन करें
        test_image = np.random.randn(1, 784)  # पहले से flattened टेस्ट इमेज
        predicted_digit, probabilities = predict_digit(trained_model, test_image)
        
        print(f"🔮 प्रेडिक्टेड डिजिट: {predicted_digit}")
        
        # प्रोबेबिलिटीज display करें
        prob_array = probabilities.data if hasattr(probabilities, 'data') else probabilities
        print(f"📊 टॉप 3 प्रेडिक्शन:")
        top_indices = np.argsort(prob_array[0])[-3:][::-1]
        for idx in top_indices:
            print(f"   डिजिट {idx}: {prob_array[0][idx]:.3f}")
        
        # मॉडल टेस्टिंग
        test_model(trained_model)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔧 Debugging info:")
        import traceback
        traceback.print_exc()
    
    # फाइनल मेमोरी स्टेटस
    print("\n" + "=" * 50)
    print("           फाइनल स्टेटस")
    print("=" * 50)
    monitor.update_monitoring()
    monitor.print_detailed_status()
    
    health_score = monitor.get_health_score()
    print(f"🏥 फाइनल हेल्थ स्कोर: {health_score:.1f}/100")
    
    print("\n🎉 प्रोग्राम सफलतापूर्वक पूरा हुआ!")