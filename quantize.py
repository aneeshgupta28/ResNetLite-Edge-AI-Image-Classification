import argparse
import copy
import json
from pathlib import Path

import torch
import torch.ao.quantization as tq
from torch.utils.data import DataLoader

from edgecls.bench import state_dict_mb, torch_latency
from edgecls.data import TensorImages, eval_transform, load_cache, make_loaders, split_indices
from edgecls.engine import metrics, predict, seed_everything, train_one_epoch
from edgecls.models import ResNetLite

torch.backends.quantized.engine = "x86"


def calibration_loader(data, train_idx, n=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    idx = train_idx[torch.randperm(len(train_idx), generator=g)[:n].numpy()]
    return DataLoader(TensorImages(data["x"][idx], data["y"][idx], eval_transform()), 64)


def ptq(fp32, calib):
    model = copy.deepcopy(fp32).eval().fuse()
    model.qconfig = tq.get_default_qconfig("x86")
    tq.prepare(model, inplace=True)
    predict(model, calib)
    return tq.convert(model)


def qat(fp32, loaders, epochs, lr, log):
    model = copy.deepcopy(fp32).train().fuse(qat=True)
    model.qconfig = tq.get_default_qat_qconfig("x86")
    tq.prepare_qat(model, inplace=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(loaders["train"]))
    best, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        if epochs >= 3 and epoch == epochs - 1:
            model.apply(tq.disable_observer)
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        loss = train_one_epoch(model, loaders["train"], optimizer, scheduler, mixup_alpha=0.0)
        converted = tq.convert(copy.deepcopy(model).eval())
        val = metrics(*predict(converted, loaders["val"]))
        log(f"qat epoch {epoch} loss {loss:.4f} val_f1 {val['macro_f1']:.4f}")
        if val["macro_f1"] > best:
            best, best_state = val["macro_f1"], converted
    return best_state


def load_int8_qat(path, width):
    model = ResNetLite(base_ch=width).train().fuse(qat=True)
    model.qconfig = tq.get_default_qat_qconfig("x86")
    tq.prepare_qat(model, inplace=True)
    model = tq.convert(model.eval())
    model.load_state_dict(torch.load(path))
    return model


def finetune_fp32(fp32, loaders, epochs, lr):
    model = copy.deepcopy(fp32).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(loaders["train"]))
    best, best_state = -1.0, None
    for _ in range(epochs):
        train_one_epoch(model, loaders["train"], optimizer, scheduler, mixup_alpha=0.0)
        val = metrics(*predict(model, loaders["val"]))
        if val["macro_f1"] > best:
            best, best_state = val["macro_f1"], copy.deepcopy(model).eval()
    return best_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--data", default="dataset-resized")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--qat-epochs", type=int, default=6)
    p.add_argument("--qat-lr", type=float, default=1e-4)
    args = p.parse_args()

    run = Path(args.run)
    seed = json.loads((run / "result.json").read_text())["seed"]
    seed_everything(seed)
    data = load_cache(args.data, args.size)
    splits = split_indices(data["y"].numpy(), seed=0)
    loaders = make_loaders(data, splits, args.size, seed=seed)

    fp32 = ResNetLite(base_ch=args.width)
    fp32.load_state_dict(torch.load(run / "best.pt"))
    fp32.eval()

    fp16 = copy.deepcopy(fp32).half()
    int8_ptq = ptq(fp32, calibration_loader(data, splits[0], seed=seed))
    int8_qat = qat(fp32, loaders, args.qat_epochs, args.qat_lr, print)
    fp32_ft = finetune_fp32(fp32, loaders, args.qat_epochs, args.qat_lr)

    variants = {
        "fp32": (fp32, None),
        "fp32_ft": (fp32_ft, None),
        "fp16": (fp16, torch.float16),
        "int8_ptq": (int8_ptq, None),
        "int8_qat": (int8_qat, None),
    }
    out = {}
    for name, (model, dtype) in variants.items():
        test = metrics(*predict(model, loaders["test"], dtype=dtype))
        out[name] = {
            "test": test,
            "size_mb": state_dict_mb(model),
            "latency": torch_latency(model, args.size, dtype=dtype or torch.float32),
        }
        print(f"{name:9s} acc {test['accuracy']:.4f} f1 {test['macro_f1']:.4f} "
              f"size {out[name]['size_mb']:.2f}MB p50 {out[name]['latency']['median_ms']:.2f}ms")
    torch.save(int8_qat.state_dict(), run / "int8_qat.pt")
    (run / "quant.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
