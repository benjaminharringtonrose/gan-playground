# Data loading and preprocessing
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import IMG_SIZE, IMG_CH, BATCH_SIZE

def make_loader(batch_size=BATCH_SIZE):
    """Create MNIST data loader with preprocessing"""
    tfm = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*IMG_CH, [0.5]*IMG_CH),
    ])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        persistent_workers=True
    )

def get_mnist_dataloader(batch_size=BATCH_SIZE, num_workers=4):
    """Create MNIST data loader with preprocessing (alias for make_loader)"""
    tfm = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*IMG_CH, [0.5]*IMG_CH),
    ])
    ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    return DataLoader(
        ds, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True,
        persistent_workers=True
    )
