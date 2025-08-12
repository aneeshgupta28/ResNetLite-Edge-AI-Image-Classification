# ResNetLite Edge AI Image Classification Pipeline

Lightweight ResNet-based image classification pipeline optimized for edge devices.  
Includes training, quantization-aware training (QAT), model optimization, and export to Core ML format for Apple Silicon and other platforms.

---

## Features

- Custom **ResNetLite** CNN designed for efficient edge deployment  
- Full pipeline: training → QAT fine-tuning → batch norm fusion → TorchScript & Core ML export  
- Quantization-aware training for improved low-precision accuracy  
- Core ML `.mlpackage` export with FP16 quantization for iOS/macOS deployment  
- Data augmentation (random resized crop, color jitter, mixup) and weighted sampling for robust training  
- Reproducible with fixed seeds, stratified splits, and learning rate scheduling  
- Supports CPU, CUDA, and Apple Metal Performance Shaders (MPS)

---

## Requirements

- Python 3.8+  
- PyTorch 1.13+  
- torchvision  
- scikit-learn  
- coremltools (for Core ML export)  
- numpy

Install dependencies with:

    pip install torch torchvision scikit-learn numpy coremltools

---

## Usage

1. Prepare your dataset in `dataset-resized/` folder, organized by class subfolders.  

2. Configure options in `model.py`:  
   - `DO_TRAIN = True` to train model from scratch  
   - `DO_QAT = True` to enable quantization-aware training  
   - `DO_EXPORT = True` to export model to Core ML format  

3. Run the pipeline:

    python model.py

---

## Outputs

- `best_resnet_lite.pth`: Trained baseline weights  
- `qat_resnetlite.pth`: Quantization-aware trained weights  
- `resnetlite_traced.pt`: TorchScript model  
- `adcnn_trashnet_fp16.mlpackage`: Core ML FP16 quantized model for edge deployment

---

## Results

Baseline validation accuracy: 81%

Model size reduction: 5.6MB

Target deployment: iOS/macOS devices using Core ML with FP16 precision.

---

## Dataset

https://github.com/garythung/trashnet - (Reworked resize.py)

---

