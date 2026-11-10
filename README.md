# ResNetLite: Edge Image Classification on TrashNet

A small ResNet variant for 6-class waste classification (TrashNet, 2,527 images), with a quantization study and an accuracy/latency comparison against standard efficient baselines.

## Setup

```
pip install torch torchvision scikit-learn numpy matplotlib onnx onnxruntime
pip install coremltools   # only for export.py
```

Download `dataset-resized.zip` from [garythung/trashnet](https://github.com/garythung/trashnet) and unzip it so that `dataset-resized/<class>/*.jpg` sits in the repo root.

## Protocol

- **Split:** stratified 70 / 15 / 15 (train / val / test), fixed split seed. 1,768 / 379 / 380 images; the test set has 21 `trash` images.
- **Model selection:** checkpoint chosen by validation macro-F1. The test set is evaluated once, on that checkpoint.
- **Seeds:** ResNetLite-32 and its quantized variants are reported as mean ± std over 3 training seeds. The width sweep and baselines use seed 0 (see Limitations).
- **Metrics:** accuracy, macro-F1, per-class accuracy, confusion matrix.
- **Training:** 128×128 input, 30 epochs, AdamW + OneCycle (max lr 2e-3), class-balanced sampling, standard augmentation. Same recipe for every model, all trained from scratch with no per-model tuning. ResNets train with bf16 autocast; MobileNetV3 trains in FP32.
- **Latency:** batch 1, single thread, median and p99 over 500 runs after warmup, on an idle Intel Xeon (Sapphire Rapids, 2.1 GHz). Cross-model numbers use ONNX Runtime FP32. The quantization table uses PyTorch eager (x86 INT8 backend), because that is where the INT8 kernels run.

## Usage

```
python train.py --model resnetlite_w32 --seed 0 --epochs 30 --bf16
python quantize.py --run runs/resnetlite_w32_s0 --width 32
python benchmark.py --models resnetlite_w16 resnetlite_w24 resnetlite_w32 resnetlite_w48 mobilenetv3s resnet18 \
    --quant-run runs/resnetlite_w32_s0
python report.py
python export.py --ckpt runs/resnetlite_w32_s0/best.pt --out resnetlite_w32.mlpackage
```

`report.py` writes the result tables to `results/results.md` and the accuracy/latency and confusion-matrix plots to `results/`. Per-run metrics, including confusion matrices, are in `runs/*/result.json` and `runs/*/quant.json`.

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

### Quantization ablation, ResNetLite-32 (3 seeds)

| Variant | Test acc (%) | Macro-F1 (%) | Trash acc (%) | Size (MB) | PyTorch 1-thread p50 / p99 (ms) |
|---|---|---|---|---|---|
| FP32 | 80.1 ± 0.4 | 77.7 ± 0.2 | 73.0 ± 12.0 | 10.74 | 7.27 / 11.06 |
| FP32 + same extra fine-tuning (control) | 81.8 ± 0.7 | 79.4 ± 0.9 | 69.8 ± 7.3 | 10.74 | 7.27 / 11.06 |
| FP16 | 80.1 ± 0.4 | 77.7 ± 0.2 | 73.0 ± 12.0 | 5.39 | 9.33 / 14.47 |
| INT8 PTQ (no QAT) | 79.2 ± 0.3 | 76.8 ± 0.3 | 71.4 ± 12.6 | 2.75 | 2.15 / 2.98 |
| INT8 QAT | 79.5 ± 0.0 | 77.2 ± 0.6 | 71.4 ± 4.8 | 2.75 | 2.22 / 3.64 |

### Findings

- **Width trade-off.** Going from width 16 to 48 buys about 4.7 points of accuracy for 6.5x the latency. ResNetLite-48 matches or beats ResNet-18 (82.1% vs 81.1%) at 0.56x the parameters and 0.68x the latency.
- **Width 24 is a bad operating point.** It costs 92M MACs against 158M for width 32, but runs at the same speed (2.27 vs 2.35 ms), presumably because channel counts that are not multiples of 16/32 map poorly onto the SIMD kernels. MACs alone do not predict latency.
- **MobileNetV3-Small has the lowest MAC count (19M) but is not the fastest on CPU,** and from scratch on 1.8k images it reaches only 71.1%. Its usual advantage depends on ImageNet pretraining and long schedules. All models here are trained from scratch with one shared recipe.
- **INT8 is close to free.** With per-channel weight quantization, post-training INT8 loses 0.9 points of accuracy for a 3.9x size reduction and a 3.4x CPU speed-up. QAT recovers 0.3 of those points, which is within seed noise. The FP32 control trained for the same extra epochs gains 1.7 points, so most of what looks like a QAT gain at fixed budget is just extra training. QAT does reduce the seed-to-seed variance on `trash`.
- **FP16 is accuracy-neutral and halves the size,** but it is slower on this CPU, which has no native FP16 path.
- **Errors:** `trash` is the hardest class and is most often confused with `glass`. `metal` and `glass` confuse each other in both directions.


## Changes from the original pipeline

- The single 80/20 split was used both to pick the best epoch and to report accuracy, so the reported number was optimistic. Replaced with a train/val/test protocol.
- QAT could not actually convert: the model had no quant/dequant stubs, used a float `+=` for the residual add, and shared one ReLU module across two call sites. The model now has `QuantStub`/`DeQuantStub`, `FloatFunctional` adds, separate ReLUs and a `fuse()` method. Existing checkpoints still load, since the parameter names are unchanged.
- The pipeline ran QAT and then exported FP16 Core ML, so the INT8 work never reached the deployed model. The FP16/INT8 trade-off is now measured explicitly.
- The Core ML export scaled inputs by 1/255 but skipped the ImageNet mean/std normalization the model was trained with. `export.py` folds the normalization into the exported graph.

## Limitations

- Latency is measured on a server CPU, not on a phone, Raspberry Pi or Neural Engine. Relative ordering between models is meaningful; absolute numbers will differ on edge hardware. The exported `.mlpackage` can be profiled with Xcode's Core ML performance report.
- FP16 has no native compute path on this CPU, so its latency here is worse than FP32. Its benefit is size, plus speed on GPU/ANE targets.
- The width sweep and baselines are single-seed. ResNetLite-32's seed-to-seed std (about 0.4 points accuracy) gives a sense of the noise.
- With only 21 `trash` test images, per-class accuracy for that class has wide variance across seeds.
