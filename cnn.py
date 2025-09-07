import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class SmallBackbone(nn.Module):
    """
    Input:  (B, 1, 28, 28)
    Output: (B, emb_dim)  # vector embedding
    """
    def __init__(self, num_channels_1: int=16, num_channels_2: int=32, emb_dim: int = 128, p: float = 0.25):
        super().__init__()
        self.CNN = nn.Sequential(
            nn.Conv2d(1, num_channels_1, 3, padding=1),   # num_channels_1x28x28
            nn.BatchNorm2d(num_channels_1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                  # num_channels_1x14x14
            nn.Conv2d(num_channels_1, num_channels_2, 3, padding=1),  # num_channels_2x14x14
            nn.BatchNorm2d(num_channels_2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                  # num_channels_2x7x7
            nn.Dropout(p),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),                     # num_channels_2*7*7
            nn.Linear(num_channels_2 * 7 * 7, emb_dim),
        )

    def forward(self, x):
        x = self.CNN(x)
        z = self.proj(x)      # embedding vector
        return z

class ClassifierHead(nn.Module):
    """
    Input:  (B, emb_dim)
    Output: (B, num_classes)
    """
    def __init__(self, emb_dim: int, num_classes: int = 10, p: float = 0.5):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(emb_dim, num_classes)
        )

    def forward(self, z):
        return self.classifier(z)

class SmallCNN(L.LightningModule):
    def __init__(self, backbone: nn.Module, head: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.lr = lr

    # get embeddings only
    def encode(self, x):
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
