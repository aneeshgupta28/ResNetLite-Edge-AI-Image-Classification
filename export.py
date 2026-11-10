import argparse

import torch
import torch.nn as nn

from edgecls.data import CLASSES, MEAN, STD
from edgecls.models import ResNetLite


class Normalized(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--out", default="resnetlite.mlpackage")
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    args = p.parse_args()

    import coremltools as ct

    model = ResNetLite(base_ch=args.width)
    model.load_state_dict(torch.load(args.ckpt))
    model = Normalized(model.eval().fuse()).eval()
    traced = torch.jit.trace(model, torch.rand(1, 3, args.size, args.size))
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, args.size, args.size), scale=1 / 255.0)],
        classifier_config=ct.ClassifierConfig(CLASSES),
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32,
    )
    mlmodel.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
