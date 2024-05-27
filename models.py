import torch.nn as nn
import torchvision.models as models
import torchvision.ops as ops

class ResNet(nn.Module):
    def __init__(self) -> None:
        super(ResNet, self).__init__()
        net = models.resnet18(pretrained=True)
        self.net = nn.Sequential(*list(net.children())[:-2])
        self.out_channels = [64, 128, 256, 512]
    
    def forward(self, x:int) -> list:       
        c2:int = self.net[4](x)
        c3:int = self.net[5](c2)
        c4:int = self.net[6](c3)
        c5:int = self.net[7](c4)
        return [c2, c3, c4, c5]
    
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels) -> None:
        super(FPN, self).__init__()
        self.fpn = ops.FeaturePyramidNetwork(in_channels_list, out_channels)

    def forward(self, x:int) -> list:
        features = self.fpn(x)
        return [features[key] for key in sorted(features.keys())]
    
class Subnet(nn.Module):
    def __init__(self, in_channels:int, num_anchors:int, num_classes:int) -> None:
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
    def __init__(self, num_classes:int) -> None:
        super(RetinaNet, self).__init__()
        self.resnet = ResNet()
        self.fpn = FPN(self.resnet.out_channels, 256)
        self.num_anchors = 9
        self.num_classes = num_classes
        self.classification_subnet = Subnet(256, self.num_anchors, num_classes)
        self.regression_subnet = Subnet(256, self.num_anchors, 4)

    def forward(self, x) -> tuple:
        """
        TODO: 
        cls_out = self.classification_subnet(feature)
        reg_out = self.regression_subnet(feature)
        """
        resnet_features = self.resnet(x)
        fpn_feature_map = self.fpn(resnet_features)
        cls_outputs = []
        reg_outputs = []
        for feature in fpn_feature_map:
            cls_out, reg_out = self.classification_subnet(feature)
            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
        return cls_outputs, reg_outputs



