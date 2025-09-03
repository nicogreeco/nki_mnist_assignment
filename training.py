import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from .network import SmallBackbone, ClassifierHead, SmallCNN

# ---- initialize
emb_dim = 256
encoder = SmallBackbone(num_channels_1=16, num_channels_2=32, emb_dim=emb_dim)
decoder = ClassifierHead(emb_dim=emb_dim, num_classes=10, p=0.25)
model = SmallCNN(encoder, decoder, lr=1e-3)

# setup data
dataset = datasets.MNIST('./data/', download=False, train=True, transform=ToTensor())
train_loader = torch.utils.data.DataLoader(dataset)