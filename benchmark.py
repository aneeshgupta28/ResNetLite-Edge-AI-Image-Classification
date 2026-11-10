import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn

from edgecls.bench import ort_latency, state_dict_mb, torch_latency
from edgecls.models import build, count_params


def count_macs(model, size):
    macs = 0

    def conv_hook(m, inp, out):
        nonlocal macs
        macs += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def linear_hook(m, inp, out):
        nonlocal macs
        macs += m.in_features * m.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
    with torch.no_grad():
        model.eval()(torch.randn(1, 3, size, size))
    for h in hooks:
        h.remove()
    return macs


def quant_latency(run, width, size):
    from edgecls.data import load_cache, split_indices
    from quantize import calibration_loader, load_int8_qat, ptq
    from edgecls.models import ResNetLite

    fp32 = ResNetLite(base_ch=width)
    fp32.load_state_dict(torch.load(Path(run) / "best.pt"))
    fp32.eval()
    data = load_cache("dataset-resized", size)
    calib = calibration_loader(data, split_indices(data["y"].numpy(), seed=0)[0])
    variants = {
        "fp32": (fp32, torch.float32),
        "fp16": (copy.deepcopy(fp32).half(), torch.float16),
        "int8_ptq": (ptq(fp32, calib), torch.float32),
        "int8_qat": (load_int8_qat(Path(run) / "int8_qat.pt", width), torch.float32),
    }
    out = {}
    for name, (model, dtype) in variants.items():
        out[name] = torch_latency(model, size, dtype=dtype, runs=500)
        print(f"{name:9s} p50 {out[name]['median_ms']:.2f}ms p99 {out[name]['p99_ms']:.2f}ms", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=[])
    p.add_argument("--quant-run")
    p.add_argument("--quant-width", type=int, default=32)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--out", default="runs/latency.json")
    args = p.parse_args()

    if args.quant_run:
        Path("runs/quant_latency.json").write_text(
            json.dumps(quant_latency(args.quant_run, args.quant_width, args.size), indent=2))

    onnx_dir = Path("runs/onnx")
    onnx_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in args.models:
        model = build(name).eval()
        if name.startswith("resnetlite"):
            model.fuse()
        path = onnx_dir / f"{name}.onnx"
        torch.onnx.export(model, torch.randn(1, 3, args.size, args.size), path,
                          input_names=["input"], output_names=["logits"], opset_version=17, dynamo=False)
        results[name] = {
            "params": count_params(build(name)),
            "macs": count_macs(build(name), args.size),
            "size_mb": state_dict_mb(model),
            "onnx_mb": path.stat().st_size / 2**20,
            "ort_1t": ort_latency(path, args.size),
            "torch_1t": torch_latency(model, args.size),
        }
        r = results[name]
        print(f"{name:15s} params {r['params']/1e6:.2f}M macs {r['macs']/1e6:.0f}M "
              f"ort p50 {r['ort_1t']['median_ms']:.2f}ms p99 {r['ort_1t']['p99_ms']:.2f}ms "
              f"torch p50 {r['torch_1t']['median_ms']:.2f}ms", flush=True)
    if results:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
