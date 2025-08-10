# GAN Playground Examples

This document provides practical examples and tutorials for using GAN Playground.

## Quick Start

### Basic Training Example

```python
from gan_playground.gan_module import GANModule
from gan_playground.data import get_mnist_dataloader
import pytorch_lightning as pl

# Create GAN module with DCGAN architecture
gan = GANModule(architecture="dcgan")

# Create trainer
trainer = pl.Trainer(
    max_epochs=20,
    accelerator="auto",
    devices="auto"
)

# Get data loader
dataloader = get_mnist_dataloader(batch_size=128)

# Train the model
trainer.fit(gan, dataloader)
```

### Generating Images After Training

```python
import torch
import matplotlib.pyplot as plt

# Generate fake images
noise = torch.randn(16, 100)  # 16 images, 100-dim noise
fake_images = gan(noise)

# Convert to numpy and denormalize
fake_images = fake_images.detach().cpu().numpy()
fake_images = (fake_images + 1) / 2  # Convert from [-1, 1] to [0, 1]

# Display images
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(fake_images[i, 0], cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

## Architecture Examples

### Classic GAN

```python
# Simple MLP-based GAN
gan = GANModule(
    architecture="classic",
    latent_dim=100,
    image_size=64,
    lr=0.0002
)
```

### DCGAN (Recommended)

```python
# Deep Convolutional GAN
gan = GANModule(
    architecture="dcgan",
    latent_dim=100,
    image_size=64,
    lr=0.0002,
    beta1=0.5,
    beta2=0.999
)
```

### WGAN-GP

```python
# Wasserstein GAN with Gradient Penalty
gan = GANModule(
    architecture="wgan-gp",
    latent_dim=100,
    image_size=64,
    lr=0.0001,
    n_critic=5,
    gp_lambda=10.0
)
```

### LSGAN

```python
# Least Squares GAN
gan = GANModule(
    architecture="lsgan",
    latent_dim=100,
    image_size=64,
    lr=0.0002
)
```

### Conditional GAN

```python
# Conditional GAN
gan = GANModule(
    architecture="cgan",
    latent_dim=100,
    image_size=64,
    lr=0.0002
)
```

## Advanced Examples

### Custom Training Loop

```python
import torch
from gan_playground.models import Generator, Discriminator
from gan_playground.data import get_mnist_dataloader

# Create models
generator = Generator()
discriminator = Discriminator()

# Create optimizers
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
criterion = torch.nn.BCELoss()

# Data loader
dataloader = get_mnist_dataloader(batch_size=128)

# Training loop
for epoch in range(20):
    for batch_idx, (real_images, _) in enumerate(dataloader):
        batch_size = real_images.size(0)

        # Train Discriminator
        d_optimizer.zero_grad()

        # Real images
        real_labels = torch.ones(batch_size, 1)
        real_outputs = discriminator(real_images)
        d_real_loss = criterion(real_outputs, real_labels)

        # Fake images
        noise = torch.randn(batch_size, 100)
        fake_images = generator(noise)
        fake_labels = torch.zeros(batch_size, 1)
        fake_outputs = discriminator(fake_images.detach())
        d_fake_loss = criterion(fake_outputs, fake_labels)

        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # Train Generator
        g_optimizer.zero_grad()

        fake_outputs = discriminator(fake_images)
        g_loss = criterion(fake_outputs, real_labels)
        g_loss.backward()
        g_optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch}/{20}] Batch [{batch_idx}/{len(dataloader)}] "
                  f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
```

### Saving and Loading Models

```python
import torch

# Save models
torch.save(gan.generator.state_dict(), "generator.pth")
torch.save(gan.discriminator.state_dict(), "discriminator.pth")

# Load models
generator = Generator()
discriminator = Discriminator()

generator.load_state_dict(torch.load("generator.pth"))
discriminator.load_state_dict(torch.load("discriminator.pth"))

# Set to evaluation mode
generator.eval()
discriminator.eval()
```

### Monitoring Training Progress

```python
import matplotlib.pyplot as plt
from gan_playground.gan_module import GANModule
import pytorch_lightning as pl

class GANModuleWithLogging(GANModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.g_losses = []
        self.d_losses = []

    def training_step(self, batch, batch_idx):
        g_loss, d_loss = super().training_step(batch, batch_idx)

        # Log losses
        self.g_losses.append(g_loss.item())
        self.d_losses.append(d_loss.item())

        return g_loss, d_loss

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.g_losses)
        plt.title("Generator Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")

        plt.subplot(1, 2, 2)
        plt.plot(self.d_losses)
        plt.title("Discriminator Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")

        plt.tight_layout()
        plt.savefig(f"losses_epoch_{self.current_epoch}.png")
        plt.close()

# Use the enhanced module
gan = GANModuleWithLogging(architecture="dcgan")
trainer = pl.Trainer(max_epochs=20)
trainer.fit(gan, get_mnist_dataloader())
```

## Command Line Examples

### Basic Training

```bash
# Train DCGAN for 20 epochs
python gan_playground/main.py --arch dcgan --epochs 20

# Train WGAN-GP with custom parameters
python gan_playground/main.py --arch wgan-gp --epochs 50 --n_critic 5 --gp_lambda 10.0

# Train with smaller batch size for memory-constrained systems
python gan_playground/main.py --arch dcgan --batch_size 64
```

### Quick Testing

```bash
# Quick test run (1 epoch)
python gan_playground/main.py --arch classic --epochs 1

# Test different architectures
python gan_playground/main.py --arch lsgan --epochs 5
python gan_playground/main.py --arch cgan --epochs 5
```

## Troubleshooting Examples

### Memory Issues

```python
# Reduce batch size
dataloader = get_mnist_dataloader(batch_size=32)

# Use gradient checkpointing
gan = GANModule(architecture="dcgan")
gan.generator = torch.utils.checkpoint.checkpoint_wrapper(gan.generator)
```

### Training Stability

```python
# Use WGAN-GP for more stable training
gan = GANModule(
    architecture="wgan-gp",
    lr=0.0001,
    n_critic=5,
    gp_lambda=10.0
)

# Or use LSGAN
gan = GANModule(architecture="lsgan", lr=0.0002)
```

### Custom Data

```python
# For custom datasets, modify the data loading function
def get_custom_dataloader(batch_size=128):
    # Your custom dataset loading logic here
    pass
```
