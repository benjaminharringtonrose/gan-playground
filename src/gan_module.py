# Main GAN Lightning Module
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import utils as vutils

from config import Z_DIM, IMG_CH, IMG_SIZE, LR, BETAS_BCE, BETAS_WGAN, NUM_CLASSES, SAMPLES_DIR
from models import DCGAN_G, DCGAN_D, MLP_G, MLP_D, onehot

class GANPlay(pl.LightningModule):
    def __init__(self, arch="dcgan", z_dim=Z_DIM, img_ch=IMG_CH, img_size=IMG_SIZE,
                 gp_lambda=10.0, n_critic=5):
        super().__init__()
        self.save_hyperparameters()
        self.arch = arch
        self.is_dc = arch in ["dcgan", "wgan-gp", "lsgan", "cgan"]
        self.is_wgan = arch == "wgan-gp"
        self.is_lsgan = arch == "lsgan"
        self.is_cgan = arch == "cgan"

        if arch == "classic":
            self.G = MLP_G(z_dim=z_dim, img_size=img_size, img_ch=img_ch)
            self.D = MLP_D(img_size=img_size, img_ch=img_ch)
            self.fixed_z = torch.randn(64, z_dim)
        elif arch == "cgan":
            self.G = DCGAN_G(in_ch=z_dim + NUM_CLASSES, out_ch=img_ch)
            self.D = DCGAN_D(in_ch=img_ch + 1)  # concat one label plane
            self.fixed_labels = torch.tensor([i%NUM_CLASSES for i in range(64)])
            self.fixed_z = torch.randn(64, z_dim, 1, 1)
        else:
            self.G = DCGAN_G(in_ch=z_dim, out_ch=img_ch)
            self.D = DCGAN_D(in_ch=img_ch)
            self.fixed_z = torch.randn(64, z_dim, 1, 1)

        # Losses
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

        # Manual optimization for all architectures to support multiple optimizers
        self.automatic_optimization = False
        self.example_input_array = torch.randn(2, z_dim, 1, 1) if self.is_dc else torch.randn(2, z_dim)
    
    def forward(self, z):
        """Dummy forward method for Lightning compatibility"""
        return self.G(z)

    # --------- Optimizers ----------
    def configure_optimizers(self):
        betas = BETAS_WGAN if self.is_wgan else BETAS_BCE
        opt_D = torch.optim.Adam(self.D.parameters(), lr=LR, betas=betas)
        opt_G = torch.optim.Adam(self.G.parameters(), lr=LR, betas=betas)
        return [opt_D, opt_G]

    # --------- Training ----------
    def _make_noise(self, bsz):
        if self.arch in ["classic"]:
            return torch.randn(bsz, self.hparams.z_dim, device=self.device)
        elif self.arch == "cgan":
            return torch.randn(bsz, self.hparams.z_dim, 1, 1, device=self.device), \
                   torch.randint(0, NUM_CLASSES, (bsz,), device=self.device)
        else:
            return torch.randn(bsz, self.hparams.z_dim, 1, 1, device=self.device)

    def _gen(self, z, y=None):
        if self.arch == "classic":
            return self.G(z)
        if self.arch == "cgan":
            y_onehot = onehot(y).unsqueeze(-1).unsqueeze(-1)  # (B,C,1,1)
            zc = torch.cat([z, y_onehot], dim=1)
            return self.G(zc)
        return self.G(z)

    def _disc(self, x, y=None):
        if self.arch == "classic":
            return self.D(x)
        if self.arch == "cgan":
            B, _, H, W = x.shape
            y_plane = (y.float()/9.0).view(-1,1,1,1).expand(-1,1,H,W)  # simple conditioning
            return self.D(torch.cat([x, y_plane], dim=1))
        return self.D(x)

    # ---- WGAN-GP helpers ----
    def _gradient_penalty(self, real, fake):
        bsz = real.size(0)
        eps = torch.rand(bsz, 1, 1, 1, device=self.device)
        x_hat = eps*real + (1-eps)*fake
        x_hat.requires_grad_(True)
        d_hat = self.D(x_hat)
        grad = torch.autograd.grad(outputs=d_hat, inputs=x_hat,
                                   grad_outputs=torch.ones_like(d_hat),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        gp = ((grad.view(bsz, -1).norm(2, dim=1) - 1.0) ** 2).mean()
        return gp

    def training_step(self, batch, batch_idx):
        real, labels = batch
        bsz = real.size(0)
        opt_D, opt_G = self.optimizers()

        # --------- WGAN-GP ----------
        if self.is_wgan:
            for _ in range(self.hparams.n_critic):
                z = self._make_noise(bsz)
                fake = self.G(z) if self.arch != "cgan" else self._gen(*z)
                if self.arch == "cgan":
                    y = z[1]
                    d_real = self._disc(real, labels)
                    d_fake = self._disc(fake.detach(), y)
                else:
                    d_real = self.D(real)
                    d_fake = self.D(fake.detach())

                gp = self._gradient_penalty(real, fake.detach())
                d_loss = -(d_real.mean() - d_fake.mean()) + self.hparams.gp_lambda*gp

                opt_D.zero_grad(); self.manual_backward(d_loss); opt_D.step()

            # G step
            z = self._make_noise(bsz)
            fake = self.G(z) if self.arch != "cgan" else self._gen(*z)
            if self.arch == "cgan":
                g_loss = -self._disc(fake, z[1]).mean()
            else:
                g_loss = -self.D(fake).mean()

            opt_G.zero_grad(); self.manual_backward(g_loss); opt_G.step()

            self.log_dict({"loss_D": d_loss.detach(), "loss_G": g_loss.detach()},
                          prog_bar=True, batch_size=bsz)
            return

        # --------- BCE / LSGAN ----------
        # D step
        if self.arch == "cgan":
            z, y_fake = self._make_noise(bsz)
            fake = self._gen(z, y_fake).detach()
            d_real = self._disc(real, labels)
            d_fake = self._disc(fake, y_fake)
        else:
            z = self._make_noise(bsz)
            fake = self._gen(z).detach()
            d_real = self._disc(real)
            d_fake = self._disc(fake)

        if self.is_lsgan:
            loss_D = self.mse(d_real, torch.ones_like(d_real)) + \
                     self.mse(d_fake, torch.zeros_like(d_fake))
        else:
            loss_D = self.bce(d_real, torch.ones_like(d_real)) + \
                     self.bce(d_fake, torch.zeros_like(d_fake))
        
        opt_D.zero_grad(); self.manual_backward(loss_D); opt_D.step()
        self.log("loss_D", loss_D, prog_bar=True, batch_size=bsz)

        # G step
        if self.arch == "cgan":
            z, y_fake = self._make_noise(bsz)
            d_fake = self._disc(self._gen(z, y_fake), y_fake)
        else:
            z = self._make_noise(bsz)
            d_fake = self._disc(self._gen(z))

        if self.is_lsgan:
            loss_G = self.mse(d_fake, torch.ones_like(d_fake))
        else:
            loss_G = self.bce(d_fake, torch.ones_like(d_fake))
        
        opt_G.zero_grad(); self.manual_backward(loss_G); opt_G.step()
        self.log("loss_G", loss_G, prog_bar=True, batch_size=bsz)

    # --------- Epoch samples ----------
    def on_train_epoch_end(self):
        self.G.eval()
        with torch.no_grad():
            if self.arch == "classic":
                z = self.fixed_z.to(self.device)
                fake = self.G(z)
            elif self.arch == "cgan":
                z = self.fixed_z.to(self.device)
                y = self.fixed_labels.to(self.device)
                fake = self._gen(z, y)
            else:
                z = self.fixed_z.to(self.device)
                fake = self.G(z)
            grid = vutils.make_grid(fake, nrow=8, normalize=True, value_range=(-1,1))
            out = os.path.join(SAMPLES_DIR, f"{self.arch}_epoch_{self.current_epoch+1:03d}.png")
            vutils.save_image(grid, out)
        self.print(f"Saved {out}")
