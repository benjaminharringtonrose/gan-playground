# Configuration constants for GAN Playground
import os

# Image configuration
IMG_SIZE = 64
IMG_CH = 1          # MNIST
Z_DIM = 100
G_FEAT = 64
D_FEAT = 64

# Training configuration
LR = 2e-4
BETAS_BCE = (0.5, 0.999)    # classic/DCGAN/LSGAN
BETAS_WGAN = (0.0, 0.9)     # WGAN-GP common choice
BATCH_SIZE = 128
EPOCHS = 20
NUM_CLASSES = 10

# Directories
SAMPLES_DIR = "samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)
