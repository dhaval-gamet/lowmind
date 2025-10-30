import lowmind as lm
import numpy as np


def smart_home_demo_visible():
    print("🏠 Smart Home AI Demo")
    print("=" * 40)
    
    class SmartHomeAI(lm.Module):
        def __init__(self):
            super().__init__()
            self.decision_layer = lm.Linear(4, 3)  # 4 sensors -> 3 actions
        
        def forward(self, x):
            return self.decision_layer(x).sigmoid()
    
    # Create AI system
    home_ai = SmartHomeAI()
    
    # Simulate different scenarios
    scenarios = {
        "🌞 Sunny Day": [30.0, 800, 0.1, 40],   # hot, bright, no motion, dry
        "🌙 Night Time": [22.0, 50, 0.9, 65],    # cool, dark, motion, humid
        "🌧️ Rainy Day": [18.0, 200, 0.2, 85]    # cold, dim, little motion, very humid
    }
    
    for scenario_name, sensor_data in scenarios.items():
        print(f"\n{scenario_name}:")
        print(f"  Sensors -> Temp: {sensor_data[0]}°C, Light: {sensor_data[1]}lux, "
              f"Motion: {sensor_data[2]}, Humidity: {sensor_data[3]}%")
        
        # Get AI decision
        input_tensor = lm.Tensor([sensor_data])
        decisions = home_ai(input_tensor)
        ac, lights, security = decisions.data[0]
        
        print("  🤖 AI Decisions:")
        print(f"     ❄️  AC: {'ON' if ac > 0.5 else 'OFF'} (confidence: {ac:.2f})")
        print(f"     💡 Lights: {'ON' if lights > 0.5 else 'OFF'} (confidence: {lights:.2f})")
        print(f"     🔒 Security: {'ARMED' if security > 0.5 else 'DISARMED'} (confidence: {security:.2f})")
    
    print("\n" + "=" * 40)
    print("✅ Smart Home Demo Completed!")

# Run this demo
if __name__ == "__main__":
    smart_home_demo_visible()