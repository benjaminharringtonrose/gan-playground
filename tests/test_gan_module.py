"""
Tests for the GAN module.
"""

import pytest
import torch
from gan_playground.gan_module import GANModule


class TestGANModule:
    """Test cases for the GANModule class."""

    def test_gan_module_initialization(self):
        """Test that GANModule can be initialized."""
        gan_module = GANModule(architecture="dcgan")
        
        assert gan_module is not None
        assert hasattr(gan_module, 'generator')
        assert hasattr(gan_module, 'discriminator')

    def test_gan_module_forward(self):
        """Test that GANModule can perform forward pass."""
        gan_module = GANModule(architecture="dcgan")
        batch_size = 4
        latent_dim = 100
        noise = torch.randn(batch_size, latent_dim)
        
        fake_images = gan_module(noise)
        
        assert fake_images.shape[0] == batch_size
        assert fake_images.shape[1] == 1  # channels
        assert fake_images.shape[2] == 64  # height
        assert fake_images.shape[3] == 64  # width

    def test_different_architectures(self):
        """Test that GANModule works with different architectures."""
        architectures = ["classic", "dcgan", "wgan-gp", "lsgan", "cgan"]
        
        for arch in architectures:
            gan_module = GANModule(architecture=arch)
            assert gan_module is not None
            
            # Test forward pass
            noise = torch.randn(2, 100)
            fake_images = gan_module(noise)
            assert fake_images.shape == (2, 1, 64, 64)

    def test_gan_module_parameters(self):
        """Test that GANModule has trainable parameters."""
        gan_module = GANModule(architecture="dcgan")
        
        # Check that both generator and discriminator have parameters
        generator_params = list(gan_module.generator.parameters())
        discriminator_params = list(gan_module.discriminator.parameters())
        
        assert len(generator_params) > 0
        assert len(discriminator_params) > 0
        
        # Check that parameters are trainable
        for param in generator_params:
            assert param.requires_grad
        
        for param in discriminator_params:
            assert param.requires_grad
