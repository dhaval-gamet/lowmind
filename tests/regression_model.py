import numpy as np
from lowmind import Tensor, Linear, SGD

class RegressionDataLoader:
def __init__(self):
pass

def generate_regression_data(self, num_samples=100):
"""Generate synthetic regression data: y = 2x + 1 + noise"""
X = np.random.randn(num_samples, 1).astype(np.float32)
y = 2 * X + 1 + 0.1 * np.random.randn(num_samples, 1).astype(np.float32)
return X, y

class RegressionTrainer:
def __init__(self, model, optimizer):
self.model = model
self.optimizer = optimizer
self.losses = []

def train(self, X, y, epochs=100):
"""Train regression model"""
X_tensor = Tensor(X)
y_tensor = Tensor(y)

print("📈 Training Regression Model")
print("=" * 40)

for epoch in range(epochs):
self.optimizer.zero_grad()

predictions = self.model(X_tensor)
loss = ((predictions - y_tensor) ** 2).mean()

loss.backward()
self.optimizer.step()

self.losses.append(loss.data[0])

if epoch % 20 == 0:
print(f"Epoch {epoch:3d} | Loss: {loss.data[0]:.4f}")

return self.losses

def train_regression_model():
"""Train a simple regression model"""

data_loader = RegressionDataLoader()
X, y = data_loader.generate_regression_data(100)

model = Linear(1, 1, device='cpu')
optimizer = SGD(model.parameters(), lr=0.01)

trainer = RegressionTrainer(model, optimizer)
losses = trainer.train(X, y, epochs=100)

print(f"\n✅ Regression Training Completed!")
print(f"Final weights: {model.weight.data[0][0]:.4f}")
print(f"Final bias: {model.bias.data[0]:.4f}")
print(f"True function: y = 2x + 1")

return model, losses

if __name__ == "__main__":
model, losses = train_regression_model()