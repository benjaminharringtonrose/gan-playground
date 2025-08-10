# Main GAN Playground Script
import argparse
import pytorch_lightning as pl

from config import EPOCHS, BATCH_SIZE
from data import make_loader
from gan_module import GANPlay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["classic","dcgan","wgan-gp","lsgan","cgan"], default="dcgan")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--n_critic", type=int, default=5, help="for wgan-gp")
    parser.add_argument("--gp_lambda", type=float, default=10.0, help="for wgan-gp")
    args = parser.parse_args()

    loader = make_loader(args.batch_size)
    model = GANPlay(arch=args.arch, gp_lambda=args.gp_lambda, n_critic=args.n_critic)

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="auto", devices="auto", log_every_n_steps=50)
    trainer.fit(model, loader)

if __name__ == "__main__":
    main()