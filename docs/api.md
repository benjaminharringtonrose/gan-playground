# GAN Playground API Documentation

## Overview

GAN Playground is a comprehensive library for experimenting with different Generative Adversarial Network (GAN) architectures. This document provides detailed API documentation for all modules and classes.

## Core Modules

### gan_playground.models

Contains the core GAN model architectures.

#### Generator

The Generator class creates fake images from random noise.

```python
class Generator(nn.Module):
    def __init__(self, latent_dim=100, image_size=64, channels=1):
        """
        Initialize the Generator.

        Args:
            latent_dim (int): Dimension of the latent space
            image_size (int): Size of generated images (assumed square)
            channels (int): Number of image channels
        """
```

**Methods:**

- `forward(noise)`: Generate fake images from noise

#### Discriminator

The Discriminator class distinguishes between real and fake images.

```python
class Discriminator(nn.Module):
    def __init__(self, image_size=64, channels=1):
        """
        Initialize the Discriminator.

        Args:
            image_size (int): Size of input images (assumed square)
            channels (int): Number of image channels
        """
```

**Methods:**

- `forward(images)`: Classify images as real or fake

### gan_playground.data

Handles data loading and preprocessing.

#### get_mnist_dataloader

```python
def get_mnist_dataloader(batch_size=128, num_workers=4):
    """
    Create a DataLoader for the MNIST dataset.

    Args:
        batch_size (int): Number of samples per batch
        num_workers (int): Number of worker processes for data loading

    Returns:
        DataLoader: PyTorch DataLoader for MNIST
    """
```

### gan_playground.gan_module

The main GAN training module using PyTorch Lightning.

#### GANModule

```python
class GANModule(pl.LightningModule):
    def __init__(self, architecture="dcgan", latent_dim=100,
                 image_size=64, channels=1, lr=0.0002,
                 beta1=0.5, beta2=0.999, n_critic=5, gp_lambda=10.0):
        """
        Initialize the GAN module.

        Args:
            architecture (str): GAN architecture type
            latent_dim (int): Dimension of latent space
            image_size (int): Size of generated images
            channels (int): Number of image channels
            lr (float): Learning rate
            beta1 (float): Beta1 for Adam optimizer
            beta2 (float): Beta2 for Adam optimizer
            n_critic (int): Number of critic iterations (WGAN-GP)
            gp_lambda (float): Gradient penalty lambda (WGAN-GP)
        """
```

**Methods:**

- `forward(noise)`: Generate fake images
- `training_step(batch, batch_idx)`: Training logic
- `configure_optimizers()`: Configure optimizers
- `on_train_epoch_end()`: End of epoch logic

### gan_playground.config

Configuration constants and hyperparameters.

```python
# Training parameters
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 0.0002
BETA1 = 0.5
BETA2 = 0.999

# Model parameters
LATENT_DIM = 100
IMAGE_SIZE = 64
CHANNELS = 1

# WGAN-GP specific parameters
N_CRITIC = 5
GP_LAMBDA = 10.0
```

## Supported Architectures

### Classic GAN

- Basic MLP-based generator and discriminator
- Binary cross-entropy loss

### DCGAN

- Deep Convolutional GAN
- Uses convolutional layers with batch normalization
- Binary cross-entropy loss

### WGAN-GP

- Wasserstein GAN with Gradient Penalty
- Wasserstein loss with gradient penalty regularization
- Improved training stability

### LSGAN

- Least Squares GAN
- Uses least squares loss instead of binary cross-entropy
- More stable training

### Conditional GAN

- Conditional GAN with class labels
- Generates images conditioned on class information

## Usage Examples

### Basic Training

```python
from gan_playground.gan_module import GANModule
from gan_playground.data import get_mnist_dataloader
import pytorch_lightning as pl

# Create GAN module
gan = GANModule(architecture="dcgan")

# Create trainer
trainer = pl.Trainer(max_epochs=20)

# Train
trainer.fit(gan, get_mnist_dataloader())
```

### Custom Configuration

```python
# Custom GAN with specific parameters
gan = GANModule(
    architecture="wgan-gp",
    latent_dim=128,
    image_size=32,
    lr=0.0001,
    n_critic=5,
    gp_lambda=10.0
)
```

### Generating Images

```python
import torch

# Generate fake images
noise = torch.randn(16, 100)  # 16 images, 100-dim noise
fake_images = gan(noise)
```

## Command Line Interface

The main script provides a command-line interface for training:

```bash
python gan_playground/main.py --arch dcgan --epochs 50 --batch_size 64
```

Available options:

- `--arch`: GAN architecture (classic, dcgan, wgan-gp, lsgan, cgan)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--n_critic`: Number of critic iterations (WGAN-GP)
- `--gp_lambda`: Gradient penalty lambda (WGAN-GP)
