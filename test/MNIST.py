import lowmind as lm
import numpy as np

# बहुत ही सरल रेग्रेशन उदाहरण
def simple_regression_demo():
    print("📈 सरल रेग्रेशन डेमो")
    
    # सरल डेटा: y = 2x + 1
    x_data = np.array([[1], [2], [3], [4]], dtype=np.float32)
    y_data = np.array([[3], [5], [7], [9]], dtype=np.float32)
    
    # सरल लीनियर मॉडल
    model = lm.Linear(1, 1)  # 1 इनपुट, 1 आउटपुट
    optimizer = lm.SGD(model.parameters(), lr=0.01)
    
    # ट्रेनिंग
    for epoch in range(100):
        total_loss = 0
        for x, y in zip(x_data, y_data):
            x_tensor = lm.Tensor([x])
            y_tensor = lm.Tensor([y])
            
            # फॉरवर्ड पास
            pred = model(x_tensor)
            loss = lm.mse_loss(pred, y_tensor)
            
            # बैकवर्ड पास
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(x_data):.4f}")
    
    # टेस्ट प्रेडिक्शन
    test_x = lm.Tensor([[5.0]])
    prediction = model(test_x)
    print(f"📊 इनपुट: 5, प्रेडिक्शन: {prediction.item():.2f}, एक्स्पेक्टेड: 11")
    
    # वेट प्रिंट करें
    print(f"🔍 ट्रेंड वेट: {model.weight.data}, बायस: {model.bias.data}")

# रन करें
if __name__ == "__main__":
    simple_regression_demo()