# ResNetLite: Edge Image Classification on TrashNet

Small ResNet for 6-class waste classification on TrashNet (2,527 images). Includes INT8/FP16 quantization experiments and a latency comparison with ResNet-18 and MobileNetV3-Small.

## Setup

```
pip install torch torchvision scikit-learn numpy matplotlib onnx onnxruntime
pip install coremltools   # only for export.py
```

Get `dataset-resized.zip` from [garythung/trashnet](https://github.com/garythung/trashnet) and unzip it into the repo root, so the images are in `dataset-resized/<class>/`.

## Experimental setup

Stratified 70/15/15 train/val/test split (1768/379/380 images, 21 of the test images are trash). The checkpoint is picked on val macro-F1 and the test set is only evaluated once at the end.

All models are trained from scratch with the same settings: 128x128 input, 30 epochs, AdamW with OneCycle (max lr 2e-3), class-balanced sampling, standard augmentation. ResNets use bf16 autocast, MobileNetV3 uses FP32. ResNetLite-32 was run with 3 seeds (mean ± std reported), the other models with 1 seed.

Latency is batch size 1, single thread, median and p99 over 500 runs, on a cloud VM (Intel Xeon, 2 vCPUs). The model comparison uses ONNX Runtime FP32. The quantization table uses PyTorch since that is where the INT8 kernels run.

## Usage

```
python train.py --model resnetlite_w32 --seed 0 --epochs 30 --bf16
python quantize.py --run runs/resnetlite_w32_s0 --width 32
python benchmark.py --models resnetlite_w16 resnetlite_w24 resnetlite_w32 resnetlite_w48 mobilenetv3s resnet18 \
    --quant-run runs/resnetlite_w32_s0
python report.py
python export.py --ckpt runs/resnetlite_w32_s0/best.pt --out resnetlite_w32.mlpackage
```

`report.py` writes the tables to `results/results.md` and saves the plots in `results/`. Per-run metrics and confusion matrices are in `runs/*/result.json` and `runs/*/quant.json`.

## Results

### Model comparison

| Model | Params | MACs | Test acc (%) | Macro-F1 (%) | Trash acc (%) | ORT 1-thread p50 / p99 (ms) | Seeds |
|---|---|---|---|---|---|---|---|
| ResNetLite-16 | 0.70M | 44M | 77.4 | 75.2 | 76.2 | 0.91 / 1.08 | 1 |
| ResNetLite-24 | 1.58M | 92M | 80.0 | 76.9 | 57.1 | 2.27 / 3.48 | 1 |
| ResNetLite-32 | 2.80M | 158M | 80.1 ± 0.4 | 77.7 ± 0.2 | 73.0 ± 12.0 | 2.35 / 2.87 | 3 |
| ResNetLite-48 | 6.29M | 340M | 82.1 | 79.7 | 71.4 | 5.87 / 9.04 | 1 |
| MobileNetV3-S | 1.52M | 19M | 71.1 | 68.5 | 57.1 | 1.15 / 1.95 | 1 |
| ResNet-18 | 11.18M | 592M | 81.1 | 78.5 | 71.4 | 8.58 / 15.86 | 1 |

### Per-class test accuracy, ResNetLite-32

| Class | Test images | Accuracy (%) |
|---|---|---|
| cardboard | 61 | 84.7 ± 2.5 |
| glass | 75 | 77.3 ± 5.8 |
| metal | 62 | 77.4 ± 2.8 |
| paper | 89 | 86.5 ± 1.1 |
| plastic | 72 | 75.5 ± 4.5 |
| trash | 21 | 73.0 ± 12.0 |

### Quantization, ResNetLite-32 (3 seeds)

| Variant | Test acc (%) | Macro-F1 (%) | Trash acc (%) | Size (MB) | PyTorch 1-thread p50 / p99 (ms) |
|---|---|---|---|---|---|
| FP32 | 80.1 ± 0.4 | 77.7 ± 0.2 | 73.0 ± 12.0 | 10.74 | 7.27 / 11.06 |
| FP32 + same extra fine-tuning | 81.8 ± 0.7 | 79.4 ± 0.9 | 69.8 ± 7.3 | 10.74 | 7.27 / 11.06 |
| FP16 | 80.1 ± 0.4 | 77.7 ± 0.2 | 73.0 ± 12.0 | 5.39 | 9.33 / 14.47 |
| INT8 PTQ (no QAT) | 79.2 ± 0.3 | 76.8 ± 0.3 | 71.4 ± 12.6 | 2.75 | 2.15 / 2.98 |
| INT8 QAT | 79.5 ± 0.0 | 77.2 ± 0.6 | 71.4 ± 4.8 | 2.75 | 2.22 / 3.64 |

### Observations

ResNetLite-48 gets slightly higher accuracy than ResNet-18 (82.1 vs 81.1) with about half the parameters and lower latency. Going from width 16 to 48 adds roughly 5 points for about 6x the latency.

Width 24 has fewer MACs than width 32 but runs at the same speed, probably because 24/48/96 channels don't line up well with the CPU vector width. So MACs aren't a great proxy for latency here.

MobileNetV3-Small has by far the fewest MACs but isn't the fastest on CPU, and trained from scratch on this little data it only reaches 71%. It normally relies on ImageNet pretraining.

Post-training INT8 only loses about 1 point and is 3.9x smaller and 3.4x faster than FP32. QAT gets back 0.3 points, which is within seed noise. Fine-tuning the FP32 model for the same number of extra epochs gives +1.7, so the QAT gain is mostly just the extra training. QAT did make the trash accuracy more stable across seeds.

FP16 keeps the same accuracy at half the size, but it's slower on this CPU since there's no native FP16 support.

Trash is the weakest class and gets confused with glass most often. Metal and glass also get mixed up.
