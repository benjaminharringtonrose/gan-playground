# GAN Playground

A comprehensive playground for experimenting with different Generative Adversarial Network (GAN) architectures using PyTorch and PyTorch Lightning.

## Features

- **Multiple GAN Architectures**: Classic GAN, DCGAN, WGAN-GP, LSGAN, and Conditional GAN
- **MNIST Dataset**: Pre-configured for training on handwritten digits
- **Automatic Sample Generation**: Saves generated images after each epoch
- **Flexible Configuration**: Command-line arguments for easy experimentation

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)

## Installation

1. **Clone or download this repository**

   ```bash
   cd gan-playground
   ```

2. **Install dependencies**

   ```bash
   # Install all required packages
   pip install -r requirements.txt
   ```

   Or if you encounter externally-managed-environment errors:

   ```bash
   # Create virtual environment
   python3.10 -m venv venv

   # Activate virtual environment
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows

   # Install dependencies
   pip install -r requirements.txt
   ```

## Usage

### Method 1: Using the Main Script (Recommended)

Run the GAN training with default settings (DCGAN on MNIST):

```bash
# Using virtual environment Python directly
./venv/bin/python gan_playground/main.py

# Or activate virtual environment first
source venv/bin/activate
python gan_playground/main.py
```

### Method 2: Using the Training Script

Use the enhanced training script with more options:

```bash
# Basic training
./venv/bin/python scripts/train.py

# With custom parameters
./venv/bin/python scripts/train.py --arch wgan-gp --epochs 50 --batch_size 64
```

### Method 3: Install as Package

Install the package in development mode:

```bash
# Install in development mode
pip install -e .

# Run using the console script
gan-playground --arch dcgan --epochs 20
```

### Available Architectures

```bash
# Classic GAN (MLP-based)
./venv/bin/python gan_playground/main.py --arch classic

# DCGAN (Deep Convolutional GAN) - Default
./venv/bin/python gan_playground/main.py --arch dcgan

# WGAN-GP (Wasserstein GAN with Gradient Penalty)
./venv/bin/python gan_playground/main.py --arch wgan-gp

# LSGAN (Least Squares GAN)
./venv/bin/python gan_playground/main.py --arch lsgan

# Conditional GAN
./venv/bin/python gan_playground/main.py --arch cgan
```

### Command Line Options

```bash
./venv/bin/python gan_playground/main.py [OPTIONS]

Options:
  --arch {classic,dcgan,wgan-gp,lsgan,cgan}
                        GAN architecture to use (default: dcgan)
  --epochs INT          Number of training epochs (default: 20)
  --batch_size INT      Batch size for training (default: 128)
  --n_critic INT        Number of critic iterations for WGAN-GP (default: 5)
  --gp_lambda FLOAT     Gradient penalty lambda for WGAN-GP (default: 10.0)
```

### Examples

```bash
# Quick test run (1 epoch)
./venv/bin/python gan_playground/main.py --arch classic --epochs 1

# Train WGAN-GP with custom parameters
./venv/bin/python gan_playground/main.py --arch wgan-gp --epochs 50 --n_critic 5 --gp_lambda 15.0

# Train with smaller batch size for memory-constrained systems
./venv/bin/python gan_playground/main.py --arch dcgan --batch_size 64

# Train conditional GAN for longer
./venv/bin/python gan_playground/main.py --arch cgan --epochs 100
```

## Output

- **Generated Samples**: Images are saved to `samples/` directory after each epoch
- **Training Progress**: Real-time loss metrics displayed during training
- **Model Summary**: Architecture and parameter count shown at startup

## File Structure

```
gan_playground/
├── gan_playground/             # Main package directory
│   ├── __init__.py             # Package initialization
│   ├── config.py               # Configuration constants and hyperparameters
│   ├── models.py               # GAN model architectures (Generator/Discriminator)
│   ├── data.py                 # Data loading and preprocessing
│   ├── gan_module.py           # Main GAN Lightning module with training logic
│   └── main.py                 # Main script entry point
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── scripts/                    # Utility scripts
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation script
├── pyproject.toml              # Modern Python packaging configuration
├── samples/                    # Generated images (created automatically)
└── data/                       # MNIST dataset (downloaded automatically)
```

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=gan_playground
```

### Code Quality

```bash
# Format code
black gan_playground/ tests/

# Lint code
flake8 gan_playground/ tests/

# Type checking
mypy gan_playground/
```

## Troubleshooting

### Import Errors in Editor

If your code editor shows import errors but the code runs fine:

1. Configure your editor to use the virtual environment Python: `./venv/bin/python`
2. For VS Code: Press `Cmd+Shift+P` → "Python: Select Interpreter" → Choose `./venv/bin/python`

### Externally Managed Environment Error

If you get "externally-managed-environment" errors:

```bash
# Use virtual environment instead
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Memory Issues

If you run out of memory:

```bash
# Reduce batch size
./venv/bin/python gan_playground/main.py --batch_size 32
```

## Dependencies

- **torch** (2.2.2): PyTorch deep learning framework
- **torchvision** (0.17.2): Computer vision utilities
- **pytorch-lightning** (2.5.2): High-level training framework
- **numpy** (1.26.4): Numerical computing
- **pillow**: Image processing
- **tqdm**: Progress bars

## License

This project is open source and available under the MIT License.
