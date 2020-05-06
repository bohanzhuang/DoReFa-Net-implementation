import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
from dorefa import QConv2d, QReLU
import torch.nn.init as init
import torch.nn.functional as F
from model_serialization import load_state_dict



def conv3x3(in_planes, out_planes, bitW, stride=1):
    "3x3 convolution with padding"
    return QConv2d(in_planes, out_planes, bitW, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, bitW=32, stride=1):
    "3x3 convolution with padding"
    return QConv2d(in_planes, out_planes, bitW=32, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, bitW, bitA, stride=1, downsample=None, is_last=False):
        super(Bottleneck, self).__init__()
        self.bitW = bitW
        self.bitA = bitA
        self.conv1 = QConv2d(inplanes, planes, bitW, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QConv2d(planes, planes, bitW, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = QConv2d(planes, planes * 4, bitW, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.downsample = downsample
        self.QReLU = QReLU(k=bitA)
        self.stride = stride
        self.is_last = is_last

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.QReLU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.QReLU(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if not self.is_last:
            out = self.QReLU(out)
        else:
            out = F.relu(out)

        return out


#class downsample_layer(nn.Module):
#    def __init__(self, inplanes, planes, bitW, kernel_size=1, stride=1, bias=False):
#        super(downsample_layer, self).__init__()
 #       self.conv = QConv2d(inplanes, planes, bitW, kernel_size, stride=stride, bias=False)
 #       self.batch_norm = nn.BatchNorm2d(planes)

#    def forward(self, x):
#        x = self.conv(x)
#        x = self.batch_norm(x)
 #       return x


class ResNet(nn.Module):

    def __init__(self, block, layers, bitW, bitA, num_classes=1000):
        self.inplanes = 64
        self.bitW = bitW
        self.bitA = bitA
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.QReLU = QReLU(k=self.bitA)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, is_last=True)  #don't quantize the last layer
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, QConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, is_last=False):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:

            downsample = nn.Sequential(
                QConv2d(self.inplanes, planes * block.expansion, self.bitW,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.bitW, self.bitA, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks-1):
            layers.append(block(self.inplanes, planes, self.bitW, self.bitA))

        layers.append(block(self.inplanes, planes, self.bitW, self.bitA, is_last=is_last))

        return nn.Sequential(*layers)


    def forward(self, x):
 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.QReLU(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(bitW, bitA, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], bitW, bitA, **kwargs)
    if pretrained == True:
        load_dict = torch.load('./full_precision_records/weights/model_best.pth.tar')['state_dict']
        model_dict = model.state_dict()
        model_keys = model_dict.keys()
        for name, param in load_dict.items():
            if name.replace('module.', '') in model_keys:
                model_dict[name.replace('module.', '')] = param    
        model.load_state_dict(model_dict)  
    return model


def resnet34(bitW, bitA, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], bitW, bitA, **kwargs)
    if pretrained == True:
        load_dict = torch.load('./full_precision_records/weights/model_best.pth.tar')['state_dict']
        model_dict = model.state_dict()
        model_keys = model_dict.keys()
        for name, param in load_dict.items():
            if name.replace('module.', '') in model_keys:
                model_dict[name.replace('module.', '')] = param    
        model.load_state_dict(model_dict)  
    return model


def resnet50(bitW, bitA, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], bitW, bitA, **kwargs)
    if pretrained:
        load_dict = torch.load('./resnet50.pth')
        load_state_dict(model, load_dict)
    else:
        load_dict = torch.load('./full_precision_records/weights/model_best.pth.tar')['state_dict']
    return model
