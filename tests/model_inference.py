import numpy as np
from lowmind import Tensor
from image_classifier import EnhancedMicroCNN, CIFAR10ImageLoader
from regression_model import Linear
from binary_classifier import SimpleMLP

class ModelInference:
def __init__(self):
self.dataloader = CIFAR10ImageLoader()

def image_classification_inference(self):
"""Image classification inference example"""
print("1. Image Classification Inference:")
model = EnhancedMicroCNN(num_classes=10, device='cpu')

dummy_images = Tensor(np.random.randn(4, 3, 32, 32))

model.eval()
with torch.no_grad():
outputs = model(dummy_images)
predictions = np.argmax(outputs.data, axis=1)

print(f" Input shape: {dummy_images.shape}")
print(f" Output shape: {outputs.shape}")
print(f" Predictions: {predictions}")
print(f" Predicted classes: {[self.dataloader.class_names[p] for p in predictions]}")

def regression_inference(self):
"""Regression inference example"""
print("\n2. Regression Inference:")
model = Linear(1, 1, device='cpu')

test_x = Tensor(np.array([[0.5], [1.0], [1.5]]))
predictions = model(test_x)

print(f" Input: {test_x.data.flatten()}")
print(f" Predictions: {predictions.data.flatten()}")

def binary_classification_inference(self):
"""Binary classification inference example"""
print("\n3. Binary Classification Inference:")
model = SimpleMLP(2, [4, 4], 1, device='cpu')

test_points = Tensor(np.array([[1.0, 1.0], [-1.0, -1.0], [2.0, -1.0]]))
logits = model(test_points)
probabilities = logits.sigmoid()
predictions = (probabilities.data > 0.5).astype(int)

print(f" Test points: {test_points.data}")
print(f" Probabilities: {probabilities.data.flatten()}")
print(f" Predictions: {predictions.flatten()}")

def run_all_inference():
"""Run all inference examples"""
print("🔍 Model Inference Examples")
print("=" * 50)

inference = ModelInference()
inference.image_classification_inference()
inference.regression_inference()
inference.binary_classification_inference()

if __name__ == "__main__":
run_all_inference()