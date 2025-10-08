from image_classifier import train_image_classifier
from regression_model import train_regression_model
from binary_classifier import train_binary_classifier
from model_inference import run_all_inference
from lowmind import RaspberryPiAdvancedMonitor

def main():
    """Main training pipeline"""
    print("🤖 Raspberry Pi Deep Learning Model Training")
    print("=" * 60)
    
    monitor = RaspberryPiAdvancedMonitor()
    monitor.print_detailed_status()
    
    try:
        print("\n" + "🚀 Starting Model Training".center(50, "="))
        
        # Train all models separately
        image_model, image_trainer = train_image_classifier()
        regression_model, regression_losses = train_regression_model()
        binary_model, binary_accuracies, binary_losses = train_binary_classifier()
        
        # Show inference examples
        run_all_inference()
        
        # Final status
        monitor.print_detailed_status()
        
        print("\n🎉 All models trained successfully!")
        print("\n📚 Model Summary:")
        print("  ✅ Image Classification Model (CIFAR-10 like)")
        print("  ✅ Linear Regression Model") 
        print("  ✅ Binary Classification Model")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()