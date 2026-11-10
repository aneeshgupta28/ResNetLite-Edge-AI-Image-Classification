import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from edgecls.data import CLASSES

RUNS = Path("runs")
OUT = Path("results")
MODELS = ["resnetlite_w16", "resnetlite_w24", "resnetlite_w32", "resnetlite_w48",
          "mobilenetv3s", "resnet18"]
LABELS = {"resnetlite_w16": "ResNetLite-16", "resnetlite_w24": "ResNetLite-24",
          "resnetlite_w32": "ResNetLite-32", "resnetlite_w48": "ResNetLite-48",
          "mobilenetv3s": "MobileNetV3-S", "resnet18": "ResNet-18"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def pm(values, scale=100, digits=1):
    v = np.asarray(values) * scale
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}" if len(v) > 1 else f"{v.mean():.{digits}f}"


def load_runs(model):
    return [json.loads(p.read_text()) for p in sorted(RUNS.glob(f"{model}_s*/result.json"))]


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main_table(latency):
    rows = ["| Model | Params | MACs | Test acc (%) | Macro-F1 (%) | Trash acc (%) | ORT 1-thread p50 / p99 (ms) | Seeds |",
            "|---|---|---|---|---|---|---|---|"]
    points = []
    for m in MODELS:
        runs = load_runs(m)
        if not runs:
            continue
        acc = [r["test"]["accuracy"] for r in runs]
        f1 = [r["test"]["macro_f1"] for r in runs]
        trash = [r["test"]["per_class_acc"][CLASSES.index("trash")] for r in runs]
        lat = latency[m]["ort_1t"]
        rows.append(f"| {LABELS[m]} | {latency[m]['params']/1e6:.2f}M | {latency[m]['macs']/1e6:.0f}M | "
                    f"{pm(acc)} | {pm(f1)} | {pm(trash)} | {lat['median_ms']:.2f} / {lat['p99_ms']:.2f} | {len(runs)} |")
        points.append((m, lat["median_ms"], np.mean(acc) * 100,
                       np.std(acc, ddof=1) * 100 if len(acc) > 1 else 0, len(acc)))
    return "\n".join(rows), points


def per_class_table(model):
    runs = load_runs(model)
    support = np.asarray(runs[0]["test"]["confusion"]).sum(1)
    rows = ["| Class | Test images | Accuracy (%) |", "|---|---|---|"]
    for i, c in enumerate(CLASSES):
        rows.append(f"| {c} | {support[i]} | {pm([r['test']['per_class_acc'][i] for r in runs])} |")
    return "\n".join(rows)


def quant_table(model):
    clean_latency = json.loads((RUNS / "quant_latency.json").read_text())
    runs = [json.loads(p.read_text()) for p in sorted(RUNS.glob(f"{model}_s*/quant.json"))]
    names = {"fp32": "FP32", "fp32_ft": "FP32 + same extra fine-tuning (control)", "fp16": "FP16",
             "int8_ptq": "INT8 PTQ (no QAT)", "int8_qat": "INT8 QAT"}
    rows = ["| Variant | Test acc (%) | Macro-F1 (%) | Trash acc (%) | Size (MB) | PyTorch 1-thread p50 / p99 (ms) |",
            "|---|---|---|---|---|---|"]
    for k, label in names.items():
        acc = [r[k]["test"]["accuracy"] for r in runs]
        f1 = [r[k]["test"]["macro_f1"] for r in runs]
        trash = [r[k]["test"]["per_class_acc"][CLASSES.index("trash")] for r in runs]
        size = np.mean([r[k]["size_mb"] for r in runs])
        lat = clean_latency.get(k.replace("_ft", ""))
        p50, p99 = lat["median_ms"], lat["p99_ms"]
        rows.append(f"| {label} | {pm(acc)} | {pm(f1)} | {pm(trash)} | {size:.2f} | {p50:.2f} / {p99:.2f} |")
    return "\n".join(rows), len(runs)


def pareto_plot(points, path):
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=160)
    style(ax)
    lite = [p for p in points if p[0].startswith("resnetlite")]
    base = [p for p in points if not p[0].startswith("resnetlite")]
    ax.plot([p[1] for p in lite], [p[2] for p in lite], color=BLUE, linewidth=2, zorder=2)
    for group, color, label in ((lite, BLUE, "ResNetLite (width sweep)"), (base, ORANGE, "Baselines")):
        ax.errorbar([p[1] for p in group], [p[2] for p in group], yerr=[p[3] for p in group],
                    fmt="o", color=color, markersize=7, markeredgecolor="white", markeredgewidth=1.5,
                    elinewidth=1.2, capsize=3, label=label, zorder=3)
    for m, x, y, _, n in points:
        label = LABELS[m] + (f" (n={n})" if n > 1 else "")
        dy = -13 if m == "resnetlite_w24" else 4
        ax.annotate(label, (x, y), xytext=(7, dy), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlabel("CPU latency, batch 1, single thread, ONNX Runtime (ms, log scale)", color=INK2, fontsize=9)
    ax.set_ylabel("Test accuracy (%)", color=INK2, fontsize=9)
    ax.set_title("Accuracy vs latency on TrashNet", color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def confusion_plot(model, path):
    runs = load_runs(model)
    cm = np.sum([np.asarray(r["test"]["confusion"]) for r in runs], axis=0)
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=160)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{norm[i, j]*100:.0f}", ha="center", va="center", fontsize=8.5,
                    color="white" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha="right", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8.5, color=INK2)
    ax.set_xlabel("Predicted", color=INK2, fontsize=9)
    ax.set_ylabel("True", color=INK2, fontsize=9)
    ax.set_title(f"{LABELS[model]} confusion matrix, row-normalised (%)\nsummed over {len(runs)} seeds",
                 color=INK, fontsize=9.5, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    latency = json.loads((RUNS / "latency.json").read_text())
    table, points = main_table(latency)
    qtable, qseeds = quant_table("resnetlite_w32")
    pareto_plot(points, OUT / "pareto.png")
    confusion_plot("resnetlite_w32", OUT / "confusion_resnetlite_w32.png")
    text = "\n\n".join([
        "## Model comparison\n\n" + table,
        "## Per-class test accuracy, ResNetLite-32\n\n" + per_class_table("resnetlite_w32"),
        f"## Quantization ablation, ResNetLite-32 ({qseeds} seeds)\n\n" + qtable,
    ])
    (OUT / "results.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
