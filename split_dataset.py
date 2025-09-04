import torch, os
from torchvision import datasets

seed = 42
path = 'data/MNIST/train_val_split.pt'
full_train = datasets.MNIST("./data/", download=True, train=True)

if not os.path.exists(path):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(full_train), generator=g)
    train_idx = perm[:50000]
    val_idx   = perm[50000:]
    torch.save({"seed": seed, "train_idx": train_idx, "val_idx": val_idx}, path)
    print(f"Saved split to {path}")

