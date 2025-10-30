import lowmind as lm
import numpy as np


class CropHealthPredictor:
    def __init__(self):
        self.model = lm.Linear(5, 3)  # 5 inputs -> 3 health states
        self.health_states = ["HEALTHY", "NEEDS_CARE", "CRITICAL"]
    
    def softmax(self, x):
        """Manual softmax implementation"""
        if hasattr(x, 'data'):
            # Tensor object
            exp_x = np.exp(x.data - np.max(x.data))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        else:
            # numpy array
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        
    def predict_crop_health(self, soil_moisture, temperature, humidity, light, ph_level):
        crop_data = [soil_moisture, temperature, humidity, light, ph_level]
        input_tensor = lm.Tensor([crop_data])
        
        output = self.model(input_tensor)
        probabilities = self.softmax(output)
        health_idx = np.argmax(probabilities[0])
        confidence = probabilities[0][health_idx]
        
        return self.health_states[health_idx], confidence

def smart_agriculture_demo():
    print("🌱 Smart Agriculture System")
    print("=" * 45)
    
    predictor = CropHealthPredictor()
    
    # Different crop scenarios
    crops = {
        "🍅 Tomatoes": [0.7, 25.0, 60, 800, 6.5],    # Good conditions
        "🥬 Lettuce": [0.3, 30.0, 40, 1200, 7.2],   # Needs water
        "🌽 Corn": [0.8, 35.0, 80, 600, 5.8],       # Critical - too hot
        "🥕 Carrots": [0.6, 22.0, 65, 900, 6.8],    # Healthy
    }
    
    for crop_name, crop_data in crops.items():
        health, confidence = predictor.predict_crop_health(*crop_data)
        
        print(f"\n{crop_name}:")
        print(f"  🌡️ Conditions -> Moisture: {crop_data[0]}, Temp: {crop_data[1]}°C, "
              f"Humidity: {crop_data[2]}%, Light: {crop_data[3]}lux, pH: {crop_data[4]}")
        print(f"  💚 Health: {health} (confidence: {confidence:.2f})")
        
        # Smart recommendations
        if health == "NEEDS_CARE":
            if crop_data[0] < 0.5:
                print("  💧 ACTION: Increase watering")
            if crop_data[1] > 28:
                print("  🌬️ ACTION: Provide shade or ventilation")
            if crop_data[4] < 6.0 or crop_data[4] > 7.5:
                print("  🧪 ACTION: Adjust soil pH")
                
        elif health == "CRITICAL":
            print("  🚨 IMMEDIATE ACTION REQUIRED!")
            if crop_data[4] < 6.0 or crop_data[4] > 7.5:
                print("  🧪 Check soil pH levels immediately")
            if crop_data[1] > 32:
                print("  ❄️ Cool down environment immediately")
            if crop_data[0] < 0.3:
                print("  💦 Emergency watering needed")
        
        else:
            print("  ✅ All parameters optimal")
    
    print("\n" + "=" * 45)
    print("✅ Agriculture Analysis Completed!")

# Run agriculture demo
if __name__ == "__main__":
    smart_agriculture_demo()