import torch.nn as nn
import torch


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class ResNet_Decoder(nn.Module):
    def __init__(self, inplanes, midplanes, outplanes):
        super(ResNet_Decoder, self).__init__()

        self.mse = nn.MSELoss(reduction='mean')

        self.layer1 = self._make_layer(inplanes[0], midplanes[0], outplanes[0])
        self.layer2 = self._make_layer(inplanes[1], midplanes[1], outplanes[1])
        self.layer3 = self._make_layer(inplanes[2], midplanes[2], outplanes[2])
        self.layer4 = self._make_layer(inplanes[3], midplanes[3], outplanes[3])

        self.finallayer = conv3x3(outplanes[3], 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, inplane, midplane, outplane):
        layers = []
        layers.append(conv3x3(inplane, midplane))
        layers.append(nn.BatchNorm2d(midplane))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv3x3(midplane, midplane))
        layers.append(nn.BatchNorm2d(midplane))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.ConvTranspose2d(midplane, outplane, 2, stride=2))
        layers.append(nn.BatchNorm2d(outplane))

        return nn.Sequential(*layers)


    def train_forward(self, directs, gt):

        x = self.layer1(directs[2])
        x = torch.cat((x, directs[1]), 1)
        x = self.layer2(x)
        x = torch.cat((x, directs[0]), 1)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.finallayer(x)

        return x, self.mse(x, gt)


    def test_forward(self, directs):
        x = self.layer1(directs[2])
        x = torch.cat((x, directs[1]), 1)
        x = self.layer2(x)
        x = torch.cat((x, directs[0]), 1)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.finallayer(x)

        return x

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        # normal
        self.load_state_dict(checkpoint, strict=False)