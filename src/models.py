# GAN Model Architectures
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Z_DIM, G_FEAT, D_FEAT, IMG_SIZE, IMG_CH, NUM_CLASSES

# -------------------- Generator Models --------------------
class DCGAN_G(nn.Module):
    def __init__(self, in_ch=Z_DIM, g=G_FEAT, out_ch=IMG_CH):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, g*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g*8), nn.ReLU(True),
            nn.ConvTranspose2d(g*8, g*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g*4), nn.ReLU(True),
            nn.ConvTranspose2d(g*4, g*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g*2), nn.ReLU(True),
            nn.ConvTranspose2d(g*2, g,   4, 2, 1, bias=False),
            nn.BatchNorm2d(g),   nn.ReLU(True),
            nn.ConvTranspose2d(g, out_ch, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class MLP_G(nn.Module):
    def __init__(self, z_dim=Z_DIM, img_size=IMG_SIZE, img_ch=IMG_CH):
        super().__init__()
        self.img_size = img_size; self.img_ch = img_ch
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(True),
            nn.Linear(256, 512), nn.ReLU(True),
            nn.Linear(512, 1024), nn.ReLU(True),
            nn.Linear(1024, img_size*img_size*img_ch), nn.Tanh()
        )
    def forward(self, z):
        x = self.net(z)
        return x.view(-1, self.img_ch, self.img_size, self.img_size)

# -------------------- Discriminator Models --------------------
class DCGAN_D(nn.Module):
    def __init__(self, in_ch=IMG_CH, d=D_FEAT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, d, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(d, d*2, 4, 2, 1, bias=False), nn.BatchNorm2d(d*2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(d*2, d*4, 4, 2, 1, bias=False), nn.BatchNorm2d(d*4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(d*4, d*8, 4, 2, 1, bias=False), nn.BatchNorm2d(d*8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(d*8, 1, 4, 1, 0, bias=False)
        )
    def forward(self, x): return self.net(x).view(-1)  # logits / critic score

class MLP_D(nn.Module):
    def __init__(self, img_size=IMG_SIZE, img_ch=IMG_CH):
        super().__init__()
        self.img_size = img_size; self.img_ch = img_ch
        self.net = nn.Sequential(
            nn.Linear(img_size*img_size*img_ch, 512), nn.LeakyReLU(0.2, True),
            nn.Linear(512, 256), nn.LeakyReLU(0.2, True),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        x = x.view(-1, self.img_size*self.img_size*self.img_ch)
        return self.net(x).view(-1)

# -------------------- cGAN Helpers --------------------
def onehot(labels, num_classes=NUM_CLASSES):
    return F.one_hot(labels, num_classes=num_classes).float()

def label_to_zch(labels):       # (B,num_classes,1,1) for G input
    oh = onehot(labels).unsqueeze(-1).unsqueeze(-1)
    return oh

def label_to_img_channel(labels, H, W):  # (B,1,H,W) label map for D input
    return onehot(labels).argmax(dim=1, keepdim=True).float().fill_(0.0).add(0.0)  # placeholder

def label_plane(labels, H, W):
    # Single-channel constant map holding class index / 9.0 (simple conditioning)
    return (labels.float()/9.0).view(-1,1,1,1).expand(-1,1,H,W)
