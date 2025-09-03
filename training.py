import torch
from torch import nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from .network import SmallBackbone, ClassifierHead, SmallCNN

# initialize model
emb_dim = 256
encoder = SmallBackbone(num_channels_1=16, num_channels_2=32, emb_dim=emb_dim, p=0.25)
decoder = ClassifierHead(emb_dim=emb_dim, num_classes=10, p=0.25)
smallCNN = SmallCNN(encoder, decoder, lr=1e-3)

# setup data
full_dataset = datasets.MNIST('./data/', download=False, train=True, transform=ToTensor())
train_dataset, val_dataset = random_split(full_dataset, [0.80, 0.20])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# training
logger = TensorBoardLogger(
    save_dir='./models/', 
    version=1,
    name="lightning_logs"
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=1,
    mode="min",
    filename="best-{epoch:02d}-{val_loss:.4f}"
)

every_epoch_callback = ModelCheckpoint(
    every_n_epochs=2,
    save_top_k=-1,
    filename="epoch-{epoch:02d}"
)

early_stopping_callback = EarlyStopping(monitor="val_loss")

trainer = L.Trainer( 
            callbacks=[checkpoint_callback, every_epoch_callback], 
            max_epochs=100, 
            logger=logger)
trainer.fit(model=smallCNN, train_dataloaders=train_loader, val_dataloaders=val_loader)