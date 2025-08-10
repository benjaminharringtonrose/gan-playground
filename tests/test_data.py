"""
Tests for the data module.
"""

import pytest
import torch
from torch.utils.data import DataLoader
from gan_playground.data import get_mnist_dataloader


class TestDataLoading:
    """Test cases for data loading functionality."""

    def test_mnist_dataloader_creation(self):
        """Test that MNIST dataloader can be created."""
        dataloader = get_mnist_dataloader(batch_size=32)
        
        assert isinstance(dataloader, DataLoader)
        assert dataloader.batch_size == 32

    def test_mnist_data_shape(self):
        """Test that MNIST data has correct shape."""
        dataloader = get_mnist_dataloader(batch_size=4)
        
        for batch in dataloader:
            images, labels = batch
            
            # Check image shape
            assert images.shape[0] == 4  # batch size
            assert images.shape[1] == 1  # channels (grayscale)
            assert images.shape[2] == 64  # height
            assert images.shape[3] == 64  # width
            
            # Check label shape
            assert labels.shape[0] == 4  # batch size
            
            # Check data types
            assert images.dtype == torch.float32
            assert labels.dtype == torch.long
            
            # Only test first batch
            break

    def test_mnist_data_range(self):
        """Test that MNIST data is normalized to [-1, 1] range."""
        dataloader = get_mnist_dataloader(batch_size=4)
        
        for batch in dataloader:
            images, _ = batch
            
            # Check that images are in [-1, 1] range
            assert torch.all(images >= -1.0)
            assert torch.all(images <= 1.0)
            
            # Only test first batch
            break

    def test_different_batch_sizes(self):
        """Test that dataloader works with different batch sizes."""
        batch_sizes = [1, 16, 32, 64]
        
        for batch_size in batch_sizes:
            dataloader = get_mnist_dataloader(batch_size=batch_size)
            
            for batch in dataloader:
                images, _ = batch
                assert images.shape[0] == batch_size
                break
