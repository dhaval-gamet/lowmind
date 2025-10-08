
import numpy as np
import time
import os
from lowmind import Tensor, MicroCNN, Linear, Conv2d, SGD, cross_entropy_loss, memory_manager, RaspberryPiAdvancedMonitor

class CIFAR10ImageLoader:
def __init__(self, batch_size=32):
self.batch_size = batch_size
self.num_classes = 10
self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
'dog', 'frog', 'horse', 'ship', 'truck']

def generate_dummy_data(self, num_samples=1000):
"""Generate dummy CIFAR-10 like data for testing"""
X_train = np.random.randn(num_samples, 3, 32, 32).astype(np.float32) * 0.5 + 0.5
y_train = np.random.randint(0, self.num_classes, num_samples)
X_test = np.random.randn(200, 3, 32, 32).astype(np.float32) * 0.5 + 0.5
y_test = np.random.randint(0, self.num_classes, 200)

return (X_train, y_train), (X_test, y_test)

def get_batches(self, X, y):
"""Generate batches from data"""
indices = np.random.permutation(len(X))
for start_idx in range(0, len(X) - self.batch_size + 1, self.batch_size):
batch_indices = indices[start_idx:start_idx + self.batch_size]
yield Tensor(X[batch_indices]), Tensor(y[batch_indices])

class EnhancedMicroCNN(MicroCNN):
"""Enhanced version of MicroCNN with better architecture"""
def __init__(self, num_classes=10, device='cpu'):
super().__init__(num_classes, device)

self.conv1 = Conv2d(3, 16, 3, padding=1, device=device)
self.conv2 = Conv2d(16, 32, 3, padding=1, device=device)
self.conv3 = Conv2d(32, 64, 3, padding=1, device=device)
self.fc = Linear(64, num_classes, device=device)
self.dropout = Dropout(0.2)

def forward(self, x):
x = self.conv1(x).relu()
x = self.conv2(x).relu()
x = self.conv3(x).relu()

if x.ndim == 4:
x = x.mean(axis=(2, 3))

x = self.dropout(x)
x = self.fc(x)
return x

class ImageModelTrainer:
def __init__(self, model, optimizer, criterion):
self.model = model
self.optimizer = optimizer
self.criterion = criterion
self.monitor = RaspberryPiAdvancedMonitor()
self.train_losses = []
self.val_losses = []
self.train_accuracies = []
self.val_accuracies = []

def train_epoch(self, dataloader, epoch):
"""Train for one epoch"""
self.model.train()
total_loss = 0
correct = 0
total = 0

print(f"🎯 Training Image Classification Epoch {epoch}")
print("-" * 50)

for batch_idx, (data, target) in enumerate(dataloader.get_batches(*dataloader.train_data)):
memory_manager.free_unused()

self.optimizer.zero_grad()
output = self.model(data)
loss = self.criterion(output, target)

loss.backward()
self.optimizer.step()

pred = np.argmax(output.data, axis=1)
correct += np.sum(pred == target.data)
total += len(target.data)
total_loss += loss.data[0] * len(data.data)

if batch_idx % 10 == 0:
accuracy = 100. * correct / total
self.monitor.update_monitoring()
health_score = self.monitor.get_health_score()

print(f"Batch {batch_idx:3d} | "
f"Loss: {loss.data[0]:.4f} | Acc: {accuracy:.2f}% | "
f"Health: {health_score:.1f}/100")

avg_loss = total_loss / total
avg_accuracy = 100. * correct / total
self.train_losses.append(avg_loss)
self.train_accuracies.append(avg_accuracy)

return avg_loss, avg_accuracy

def validate(self, dataloader):
"""Validate the model"""
self.model.eval()
total_loss = 0
correct = 0
total = 0

with torch.no_grad():
for data, target in dataloader.get_batches(*dataloader.test_data):
output = self.model(data)
loss = self.criterion(output, target)

pred = np.argmax(output.data, axis=1)
correct += np.sum(pred == target.data)
total += len(target.data)
total_loss += loss.data[0] * len(data.data)

avg_loss = total_loss / total
avg_accuracy = 100. * correct / total
self.val_losses.append(avg_loss)
self.val_accuracies.append(avg_accuracy)

return avg_loss, avg_accuracy

def train_image_classifier():
"""Train an image classification model on CIFAR-10 like data"""
print("🖼️ Training Image Classification Model")
print("=" * 50)

batch_size = 32
epochs = 5
learning_rate = 0.001

dataloader = CIFAR10ImageLoader(batch_size=batch_size)
(X_train, y_train), (X_test, y_test) = dataloader.generate_dummy_data(1000)
dataloader.train_data = (X_train, y_train)
dataloader.test_data = (X_test, y_test)

model = EnhancedMicroCNN(num_classes=10, device='cpu')
optimizer = SGD(model.parameters(), lr=learning_rate, momentum=0.9)
criterion = cross_entropy_loss

trainer = ImageModelTrainer(model, optimizer, criterion)

for epoch in range(epochs):
train_loss, train_acc = trainer.train_epoch(dataloader, epoch + 1)
val_loss, val_acc = trainer.validate(dataloader)

print(f"\n📈 Epoch {epoch+1} Summary:")
print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
print("-" * 50)

return model, trainer

if __name__ == "__main__":
model, trainer = train_image_classifier()
print("✅ Image Classification Model Training Completed!")