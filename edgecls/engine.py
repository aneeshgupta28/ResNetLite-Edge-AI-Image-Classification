import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0))
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam


def train_one_epoch(model, loader, optimizer, scheduler=None, mixup_alpha=0.0, bf16=False):
    criterion = nn.CrossEntropyLoss()
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x = x.contiguous(memory_format=torch.channels_last)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
            if mixup_alpha > 0:
                x, ya, yb, lam = mixup(x, y, mixup_alpha)
                out = model(x)
                loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
            else:
                loss = criterion(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


@torch.no_grad()
def predict(model, loader, dtype=None):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        if dtype is not None:
            x = x.to(dtype)
        preds.append(model(x).float().argmax(1))
        labels.append(y)
    return torch.cat(labels).numpy(), torch.cat(preds).numpy()


def metrics(y_true, y_pred, num_classes=6):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "per_class_acc": (cm.diagonal() / cm.sum(1)).tolist(),
        "confusion": cm.tolist(),
    }


def fit(model, loaders, epochs, lr=1e-3, weight_decay=1e-4, mixup_alpha=0.0, bf16=False, ckpt=None,
        log=print):
    model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(loaders["train"])
    )
    best, best_epoch = -1.0, 0
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loaders["train"], optimizer, scheduler, mixup_alpha, bf16)
        val = metrics(*predict(model, loaders["val"]))
        log(f"epoch {epoch:02d} loss {loss:.4f} val_acc {val['accuracy']:.4f} val_f1 {val['macro_f1']:.4f}")
        if val["macro_f1"] > best:
            best, best_epoch = val["macro_f1"], epoch
            if ckpt:
                torch.save(model.state_dict(), ckpt)
    return best, best_epoch
