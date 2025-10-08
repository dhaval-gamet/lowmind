# gan_model.py
import numpy as np
from lowmind import Tensor, Linear, SGD

class Generator:
    """Generator for GAN"""
    def __init__(self, latent_dim=2, output_dim=2, device='cpu'):
        self.latent_dim = latent_dim
        self.device = device
        
        self.fc1 = Linear(latent_dim, 16, device=device)
        self.fc2 = Linear(16, 16, device=device)
        self.fc3 = Linear(16, output_dim, device=device)
    
    def parameters(self):
        params = []
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.fc3.parameters())
        return params
    
    def forward(self, z):
        x = self.fc1(z).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x).tanh()
        return x

class Discriminator:
    """Discriminator for GAN"""
    def __init__(self, input_dim=2, device='cpu'):
        self.input_dim = input_dim
        self.device = device
        
        self.fc1 = Linear(input_dim, 16, device=device)
        self.fc2 = Linear(16, 16, device=device)
        self.fc3 = Linear(16, 1, device=device)
    
    def parameters(self):
        params = []
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.fc3.parameters())
        return params
    
    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x).sigmoid()
        return x

def generate_real_data():
    """Generate real data (8-shaped distribution)"""
    np.random.seed(42)
    num_samples = 400
    
    # Create 8-shaped distribution
    t = np.linspace(0, 2*np.pi, num_samples)
    
    # First circle
    x1 = 0.5 * np.cos(t) - 0.5
    y1 = 0.5 * np.sin(t)
    
    # Second circle
    x2 = 0.5 * np.cos(t) + 0.5
    y2 = 0.5 * np.sin(t)
    
    # Combine with some noise
    real_data = np.column_stack([
        np.concatenate([x1, x2]) + np.random.normal(0, 0.05, 2*num_samples),
        np.concatenate([y1, y2]) + np.random.normal(0, 0.05, 2*num_samples)
    ])
    
    return real_data.astype(np.float32)

def train_gan():
    """Train GAN to generate 2D data"""
    print("🎨 Training Generative Adversarial Network (GAN)")
    print("=" * 50)
    
    # Generate real data
    real_data = generate_real_data()
    print(f"Generated {len(real_data)} real data samples (8-shaped distribution)")
    
    # Models
    generator = Generator(latent_dim=2, output_dim=2)
    discriminator = Discriminator(input_dim=2)
    
    # Optimizers
    g_optimizer = SGD(generator.parameters(), lr=0.01)
    d_optimizer = SGD(discriminator.parameters(), lr=0.01)
    
    # Training
    epochs = 300
    batch_size = 64
    
    print("\n🔧 Training GAN...")
    print("Epoch | Discriminator Loss | Generator Loss")
    print("-" * 45)
    
    for epoch in range(epochs):
        # Train Discriminator
        d_losses = []
        for i in range(0, len(real_data), batch_size):
            # Real data batch
            real_batch = real_data[i:min(i+batch_size, len(real_data))]
            
            # Generate fake data
            z = Tensor(np.random.randn(len(real_batch), 2).astype(np.float32))
            fake_batch = generator.forward(z)
            
            # Discriminator predictions
            d_real = discriminator.forward(Tensor(real_batch))
            d_fake = discriminator.forward(fake_batch)
            
            # Discriminator loss
            d_loss = - (d_real.log().mean() + (1 - d_fake).log().mean())
            
            # Update discriminator
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            
            d_losses.append(d_loss.data[0])
        
        # Train Generator
        g_losses = []
        for i in range(0, len(real_data), batch_size):
            # Generate fake data
            z = Tensor(np.random.randn(batch_size, 2).astype(np.float32))
            fake_batch = generator.forward(z)
            
            # Generator loss
            d_fake = discriminator.forward(fake_batch)
            g_loss = - d_fake.log().mean()
            
            # Update generator
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
            g_losses.append(g_loss.data[0])
        
        if epoch % 50 == 0:
            avg_d_loss = np.mean(d_losses)
            avg_g_loss = np.mean(g_losses)
            print(f"{epoch:5d} | {avg_d_loss:17.4f} | {avg_g_loss:14.4f}")
    
    print(f"\n✅ GAN Training Completed!")
    
    # Generate and display samples
    print("\n🎲 Generated Samples from Trained Generator:")
    print("-" * 45)
    
    # Generate multiple sets of samples
    for sample_set in range(3):
        z = Tensor(np.random.randn(5, 2).astype(np.float32))
        generated_samples = generator.forward(z)
        
        print(f"\nSample Set {sample_set + 1}:")
        for i, sample in enumerate(generated_samples.data):
            print(f"  Sample {i}: ({sample[0]:7.3f}, {sample[1]:7.3f})")
    
    # Analyze generated data distribution
    print("\n📊 Generated Data Analysis:")
    z_test = Tensor(np.random.randn(1000, 2).astype(np.float32))
    generated_data = generator.forward(z_test).data
    
    mean_x, mean_y = np.mean(generated_data, axis=0)
    std_x, std_y = np.std(generated_data, axis=0)
    
    print(f"Mean: ({mean_x:.3f}, {mean_y:.3f})")
    print(f"Std:  ({std_x:.3f}, {std_y:.3f})")
    print(f"Data range: X[{generated_data[:,0].min():.3f}, {generated_data[:,0].max():.3f}] "
          f"Y[{generated_data[:,1].min():.3f}, {generated_data[:,1].max():.3f}]")
    
    # Test discriminator on real vs generated data
    print("\n🔍 Discriminator Performance:")
    
    # Test on real data
    real_test = real_data[:100]
    d_real_pred = discriminator.forward(Tensor(real_test)).data
    real_accuracy = np.mean(d_real_pred > 0.5)
    
    # Test on generated data
    z_test = Tensor(np.random.randn(100, 2).astype(np.float32))
    fake_test = generator.forward(z_test).data
    d_fake_pred = discriminator.forward(Tensor(fake_test)).data
    fake_accuracy = np.mean(d_fake_pred < 0.5)
    
    print(f"Real data detection accuracy:    {real_accuracy:.3f}")
    print(f"Generated data detection accuracy: {fake_accuracy:.3f}")
    print(f"Overall discriminator accuracy:   {(real_accuracy + fake_accuracy)/2:.3f}")
    
    return generator, discriminator

if __name__ == "__main__":
    generator, discriminator = train_gan()
