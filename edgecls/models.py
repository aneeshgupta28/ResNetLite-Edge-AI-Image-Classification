import torch
import torch.nn as nn
import torch.ao.quantization as tq
from torchvision import models as tvm


def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU()
        self.downsample = downsample
        self.add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(self.add.add(out, identity))


class ResNetLite(nn.Module):
    def __init__(self, num_classes=6, base_ch=32, layers=(2, 2, 2, 2)):
        super().__init__()
        self.quant = tq.QuantStub()
        self.dequant = tq.DeQuantStub()
        self.in_ch = base_ch
        self.conv1 = nn.Conv2d(3, base_ch, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_ch)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(base_ch, layers[0], 1)
        self.layer2 = self._make_layer(base_ch * 2, layers[1], 2)
        self.layer3 = self._make_layer(base_ch * 4, layers[2], 2)
        self.layer4 = self._make_layer(base_ch * 8, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_ch * 8, num_classes)

    def _make_layer(self, out_ch, blocks, stride):
        downsample = None
        if stride != 1 or self.in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        layers = [BasicBlock(self.in_ch, out_ch, stride, downsample)]
        self.in_ch = out_ch
        layers += [BasicBlock(out_ch, out_ch) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        return self.dequant(self.fc(x))

    def fuse(self, qat=False):
        fuse = tq.fuse_modules_qat if qat else tq.fuse_modules
        fuse(self, [["conv1", "bn1", "relu"]], inplace=True)
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            for block in layer:
                fuse(block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]], inplace=True)
                if block.downsample is not None:
                    fuse(block.downsample, [["0", "1"]], inplace=True)
        return self


def build(name, num_classes=6):
    if name.startswith("resnetlite"):
        return ResNetLite(num_classes, base_ch=int(name.split("_w")[1]))
    if name == "resnet18":
        return tvm.resnet18(num_classes=num_classes)
    if name == "mobilenetv3s":
        return tvm.mobilenet_v3_small(num_classes=num_classes)
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
