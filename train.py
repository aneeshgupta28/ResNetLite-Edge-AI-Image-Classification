import argparse
import json
import time
from pathlib import Path

import torch

from edgecls.data import load_cache, make_loaders, split_indices
from edgecls.engine import fit, metrics, predict, seed_everything
from edgecls.models import build, count_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnetlite_w32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data", default="dataset-resized")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--out", default="runs")
    args = p.parse_args()

    run_dir = Path(args.out) / f"{args.model}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / "train.log", "w")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    data = load_cache(args.data, args.size)
    splits = split_indices(data["y"].numpy(), seed=0)
    seed_everything(args.seed)
    loaders = make_loaders(data, splits, args.size, args.batch_size, args.workers, args.seed)
    model = build(args.model)

    t0 = time.time()
    ckpt = run_dir / "best.pt"
    best_val, best_epoch = fit(model, loaders, args.epochs, lr=args.lr, mixup_alpha=args.mixup,
                                 bf16=args.bf16, ckpt=ckpt, log=log)
    model.load_state_dict(torch.load(ckpt))
    test = metrics(*predict(model, loaders["test"]))
    result = {
        "model": args.model,
        "seed": args.seed,
        "size": args.size,
        "epochs": args.epochs,
        "bf16": args.bf16,
        "mixup": args.mixup,
        "params": count_params(model),
        "best_epoch": best_epoch,
        "val_macro_f1": best_val,
        "train_minutes": (time.time() - t0) / 60,
        "test": test,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    log(f"test acc {test['accuracy']:.4f} macro_f1 {test['macro_f1']:.4f} (best epoch {best_epoch})")


if __name__ == "__main__":
    main()
