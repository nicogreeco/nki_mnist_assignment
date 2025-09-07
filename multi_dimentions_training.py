from argparse import ArgumentParser
from omegaconf import OmegaConf
import torch
from torch import nn
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader, random_split

from cnn import SmallBackbone, ClassifierHead, SmallCNN

def main(config, emb_dims):
    
    emb_dims = [int(emb_dim.strip()) for emb_dim in emb_dims.split(',')]
    for emb_dim in emb_dims:
        print(f'Train model with emb_dim {emb_dim}')
        
        # initialize model
        backbone = SmallBackbone(
            num_channels_1=config.model.num_channels_1, 
            num_channels_2=config.model.num_channels_2, 
            emb_dim=emb_dim, 
            p=config.model.dropout)
        
        head = ClassifierHead(
            emb_dim=emb_dim, 
            num_classes=10, 
            p=config.model.dropout)
        
        smallCNN = SmallCNN(
            backbone, 
            head, 
            lr=config.model.lr)

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
            name=f"lightning_logs_cnn_emb_dim_{emb_dim}"
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
                    logger=logger)
        
        trainer.fit(model=smallCNN, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default='./config.yaml',
    )
    
    parser.add_argument(
        "--emb_dims",
        type=str,
        default='128,64,32,16,8,4,2',
    )
    
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config, args.emb_dims)