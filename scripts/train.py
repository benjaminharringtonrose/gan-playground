#!/usr/bin/env python3
"""
Training script for GAN Playground.

This script provides a convenient way to train GAN models with various configurations.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytorch_lightning as pl
from gan_playground.gan_module import GANModule
from gan_playground.data import get_mnist_dataloader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train GAN models")
    
    parser.add_argument(
        "--arch",
        type=str,
        default="dcgan",
        choices=["classic", "dcgan", "wgan-gp", "lsgan", "cgan"],
        help="GAN architecture to use"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0002,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=100,
        help="Dimension of latent space"
    )
    
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Size of generated images"
    )
    
    parser.add_argument(
        "--n_critic",
        type=int,
        default=5,
        help="Number of critic iterations (WGAN-GP)"
    )
    
    parser.add_argument(
        "--gp_lambda",
        type=float,
        default=10.0,
        help="Gradient penalty lambda (WGAN-GP)"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading"
    )
    
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "tpu"],
        help="Accelerator to use"
    )
    
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="Number of devices to use"
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints"
    )
    
    parser.add_argument(
        "--samples_dir",
        type=str,
        default="samples",
        help="Directory to save generated samples"
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.samples_dir, exist_ok=True)
    
    print(f"Training {args.arch.upper()} GAN")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Image size: {args.image_size}")
    
    if args.arch == "wgan-gp":
        print(f"Critic iterations: {args.n_critic}")
        print(f"Gradient penalty lambda: {args.gp_lambda}")
    
    # Create GAN module
    gan = GANModule(
        architecture=args.arch,
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        lr=args.lr,
        n_critic=args.n_critic,
        gp_lambda=args.gp_lambda
    )
    
    # Create data loader
    dataloader = get_mnist_dataloader(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        default_root_dir=args.checkpoint_dir,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=args.checkpoint_dir,
                filename=f"{args.arch}-{{epoch:02d}}-{{val_loss:.2f}}",
                save_top_k=3,
                monitor="val_loss",
                mode="min"
            ),
            pl.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                mode="min"
            )
        ]
    )
    
    # Train the model
    trainer.fit(gan, dataloader)
    
    print("Training completed!")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    print(f"Samples saved to: {args.samples_dir}")


if __name__ == "__main__":
    main()
