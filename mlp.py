from argparse import ArgumentParser
from omegaconf import OmegaConf
import torch
from torch import nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader

class MLP(L.LightningModule):
    def __init__(self, hidden_dim1=256, hidden_dim2=128, num_classes=10, lr=1e-3, p=0.25):
        """
        A simple 3-layer MLP for MNIST (28x28 images).
        Input:  (B, 1, 28, 28)
        Output: (B, num_classes)
        """
        super().__init__()
        self.save_hyperparameters()

        self.flatten = nn.Flatten()
        
        self.backbone = nn.Sequential(
            nn.Linear(28 * 28, hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden_dim1, hidden_dim2),
        )
        
        self.head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(hidden_dim2, num_classes),
        )

        self.lr = lr

    # get embeddings only
    def encode(self, x):
        x = self.flatten(x)
        return self.backbone(x)

    def forward(self, x):
        z = self.encode(x)
        logits = self.head(z)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=2)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}
    
    
def main(config):
       
    mlp = MLP()

    # setup data
    train_transform = transforms.Compose([
        transforms.RandomCrop((28, 28), padding=config.data.padding),
        transforms.RandomRotation(degrees=(-config.data.rotation, config.data.rotation)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_train_aug = datasets.MNIST("./data/", download=True, train=True, transform=train_transform)
    full_train_val = datasets.MNIST("./data/", download=True, train=True, transform=val_transform)

    split = torch.load("data/MNIST/train_val_split.pt")
    train_idx, val_idx = split["train_idx"], split["val_idx"]

    train_dataset = Subset(full_train_aug, train_idx)
    val_dataset = Subset(full_train_val, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.data.batch_size, 
        shuffle=True, 
        num_workers=config.data.num_workers, 
        pin_memory=True, 
        persistent_workers=True)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.data.batch_size, 
        num_workers=config.data.num_workers, 
        pin_memory=True, 
        persistent_workers=True)

    # training
    logger = TensorBoardLogger(
        save_dir='./log/', 
        version=1,
        name="lightning_logs_mlp"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        mode="min",
        filename="best-{epoch:02d}-{val_loss:.4f}"
    )

    every_epoch_callback = ModelCheckpoint(
        every_n_epochs=config.training.every_n_epochs,
        save_top_k=-1,
        filename="epoch-{epoch:02d}-{val_loss:.4f}"
    )

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience = config.training.patience)

    # lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = L.Trainer( 
                callbacks=[checkpoint_callback, every_epoch_callback, early_stopping_callback], 
                max_epochs=config.training.max_epochs,
                logger=logger, 
                limit_train_batches=0.10)
    
    trainer.fit(model=mlp, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='./config.yaml',
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)
