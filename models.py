import torch.nn as nn
import torchvision.models as models
import torchvision.ops as ops

class ResNet(nn.Module):
    def __init__(self) -> None:
        super(ResNet, self).__init__()
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.out_channels = [64, 128, 256, 512]
    
    def forward(self, x) -> dict:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        # return [c2, c3, c4, c5]
        return {"0": c2, "1": c3, "2": c4, "3": c5}


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels) -> None:
        super(FPN, self).__init__()
        self.fpn = ops.FeaturePyramidNetwork(in_channels_list, out_channels)

    def forward(self, x) -> list:
        features = self.fpn(x)
        return [features[key] for key in sorted(features.keys())]
    
class Subnet(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes) -> None:
        super(Subnet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.cls_conv = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
        self.reg_conv = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x) -> tuple:
        features = self.conv(x)
        cls_out = self.cls_conv(features)
        reg_out = self.reg_conv(features)
        return cls_out, reg_out
    
class RetinaNet(nn.Module):
    def __init__(self, num_classes) -> None:
        super(RetinaNet, self).__init__()
        self.resnet = ResNet()
        self.fpn = FPN(self.resnet.out_channels, 64)
        self.num_anchors = 9
        self.num_classes = num_classes
        self.classification_subnet = Subnet(64, self.num_anchors, num_classes)
        self.regression_subnet = Subnet(64, self.num_anchors, 4)

    def forward(self, x) -> tuple:
        resnet_features = self.resnet(x)
        fpn_feature_map = self.fpn(resnet_features)
        cls_outputs = []
        reg_outputs = []
        for feature in fpn_feature_map:
            cls_out, reg_out = self.classification_subnet(feature)
            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
        return cls_outputs, reg_outputs
