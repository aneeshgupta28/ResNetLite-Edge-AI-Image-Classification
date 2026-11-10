## Model comparison

| Model | Params | MACs | Test acc (%) | Macro-F1 (%) | Trash acc (%) | ORT 1-thread p50 / p99 (ms) | Seeds |
|---|---|---|---|---|---|---|---|
| ResNetLite-16 | 0.70M | 44M | 77.4 | 75.2 | 76.2 | 0.91 / 1.08 | 1 |
| ResNetLite-24 | 1.58M | 92M | 80.0 | 76.9 | 57.1 | 2.27 / 3.48 | 1 |
| ResNetLite-32 | 2.80M | 158M | 80.1 ± 0.4 | 77.7 ± 0.2 | 73.0 ± 12.0 | 2.35 / 2.87 | 3 |
| ResNetLite-48 | 6.29M | 340M | 82.1 | 79.7 | 71.4 | 5.87 / 9.04 | 1 |
| MobileNetV3-S | 1.52M | 19M | 71.1 | 68.5 | 57.1 | 1.15 / 1.95 | 1 |
| ResNet-18 | 11.18M | 592M | 81.1 | 78.5 | 71.4 | 8.58 / 15.86 | 1 |

## Per-class test accuracy, ResNetLite-32

| Class | Test images | Accuracy (%) |
|---|---|---|
| cardboard | 61 | 84.7 ± 2.5 |
| glass | 75 | 77.3 ± 5.8 |
| metal | 62 | 77.4 ± 2.8 |
| paper | 89 | 86.5 ± 1.1 |
| plastic | 72 | 75.5 ± 4.5 |
| trash | 21 | 73.0 ± 12.0 |

## Quantization ablation, ResNetLite-32 (3 seeds)

| Variant | Test acc (%) | Macro-F1 (%) | Trash acc (%) | Size (MB) | PyTorch 1-thread p50 / p99 (ms) |
|---|---|---|---|---|---|
| FP32 | 80.1 ± 0.4 | 77.7 ± 0.2 | 73.0 ± 12.0 | 10.74 | 7.27 / 11.06 |
| FP32 + same extra fine-tuning (control) | 81.8 ± 0.7 | 79.4 ± 0.9 | 69.8 ± 7.3 | 10.74 | 7.27 / 11.06 |
| FP16 | 80.1 ± 0.4 | 77.7 ± 0.2 | 73.0 ± 12.0 | 5.39 | 9.33 / 14.47 |
| INT8 PTQ (no QAT) | 79.2 ± 0.3 | 76.8 ± 0.3 | 71.4 ± 12.6 | 2.75 | 2.15 / 2.98 |
| INT8 QAT | 79.5 ± 0.0 | 77.2 ± 0.6 | 71.4 ± 4.8 | 2.75 | 2.22 / 3.64 |
