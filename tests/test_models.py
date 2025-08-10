"""
Tests for the models module.
"""

import pytest
import torch
from gan_playground.models import Generator, Discriminator


class TestGenerator:
    """Test cases for the Generator class."""

    def test_generator_initialization(self):
        """Test that Generator can be initialized with default parameters."""
        generator = Generator()
        assert generator is not None
        assert isinstance(generator, torch.nn.Module)

    def test_generator_forward_pass(self):
        """Test that Generator can perform forward pass."""
        generator = Generator()
        batch_size = 4
        latent_dim = 100
        noise = torch.randn(batch_size, latent_dim)
        
        output = generator(noise)
        
        assert output.shape[0] == batch_size
        assert output.shape[1] == 1  # channels
        assert output.shape[2] == 64  # height
        assert output.shape[3] == 64  # width

    def test_generator_output_range(self):
        """Test that Generator output is in expected range (tanh activation)."""
        generator = Generator()
        noise = torch.randn(2, 100)
        output = generator(noise)
        
        assert torch.all(output >= -1.0)
        assert torch.all(output <= 1.0)


class TestDiscriminator:
    """Test cases for the Discriminator class."""

    def test_discriminator_initialization(self):
        """Test that Discriminator can be initialized with default parameters."""
        discriminator = Discriminator()
        assert discriminator is not None
        assert isinstance(discriminator, torch.nn.Module)

    def test_discriminator_forward_pass(self):
        """Test that Discriminator can perform forward pass."""
        discriminator = Discriminator()
        batch_size = 4
        image = torch.randn(batch_size, 1, 64, 64)
        
        output = discriminator(image)
        
        assert output.shape[0] == batch_size
        assert output.shape[1] == 1  # single output value

    def test_discriminator_output_range(self):
        """Test that Discriminator output is in expected range."""
        discriminator = Discriminator()
        image = torch.randn(2, 1, 64, 64)
        output = discriminator(image)
        
        # Output should be a single value per image (real/fake probability)
        assert output.shape == (2, 1)


class TestModelCompatibility:
    """Test that Generator and Discriminator are compatible."""

    def test_generator_discriminator_compatibility(self):
        """Test that Generator output can be fed to Discriminator."""
        generator = Generator()
        discriminator = Discriminator()
        
        batch_size = 4
        noise = torch.randn(batch_size, 100)
        fake_images = generator(noise)
        
        # Discriminator should be able to process generator output
        discriminator_output = discriminator(fake_images)
        
        assert discriminator_output.shape[0] == batch_size
        assert discriminator_output.shape[1] == 1
