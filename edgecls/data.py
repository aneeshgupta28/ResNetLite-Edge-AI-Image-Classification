from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def load_cache(data_dir, size, cache_dir="cache"):
    cache = Path(cache_dir) / f"trashnet_{size}.pt"
    if cache.exists():
        return torch.load(cache)
    images, labels = [], []
    for label, cls in enumerate(CLASSES):
        for path in sorted((Path(data_dir) / cls).glob("*.jpg")):
            img = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
            images.append(torch.from_numpy(np.array(img)).permute(2, 0, 1).clone())
            labels.append(label)
    data = {"x": torch.stack(images), "y": torch.tensor(labels)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, cache)
    return data


def split_indices(labels, seed=0, val=0.15, test=0.15):
    idx = np.arange(len(labels))
    labels = np.asarray(labels)
    train_val, test_idx = train_test_split(idx, test_size=test, stratify=labels, random_state=seed)
    train_idx, val_idx = train_test_split(
        train_val, test_size=val / (1 - test), stratify=labels[train_val], random_state=seed
    )
    return train_idx, val_idx, test_idx


def train_transform(size):
    return v2.Compose([
        v2.RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(12),
        v2.ColorJitter(0.2, 0.2, 0.2, 0.05),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD),
    ])


def eval_transform():
    return v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])


class TensorImages(Dataset):
    def __init__(self, x, y, transform):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.transform(self.x[i]), self.y[i]


def make_loaders(data, splits, size, batch_size=32, workers=0, seed=0):
    x, y = data["x"], data["y"]
    train_idx, val_idx, test_idx = splits
    train_ds = TensorImages(x[train_idx], y[train_idx], train_transform(size))
    counts = np.bincount(y[train_idx].numpy(), minlength=len(CLASSES))
    weights = (1.0 / counts)[y[train_idx].numpy()]
    sampler = WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
    loaders = {
        "train": DataLoader(train_ds, batch_size, sampler=sampler, num_workers=workers,
                            persistent_workers=workers > 0),
    }
    for name, idx in (("val", val_idx), ("test", test_idx)):
        ds = TensorImages(x[idx], y[idx], eval_transform())
        loaders[name] = DataLoader(ds, 64, shuffle=False)
    return loaders
