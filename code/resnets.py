import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Resnet_backbone(nn.Module):
    def __init__(self, block, num_blocks):
        super(Resnet_backbone, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


class InPainting_ResNet(nn.Module):
    def __init__(self, block, num_blocks, mask_size=10, in_chanels=3):
        super(InPainting_ResNet, self).__init__()
        self.backbone = Resnet_backbone(block, num_blocks)
        self.in_chanels = in_chanels
        self.mask_size = mask_size
        self.sigmoid = nn.Sigmoid()
        self.InPainting_head = nn.Linear(512*block.expansion,
                                         self.in_chanels*(mask_size**2))

    def forward(self, x):
        pen = self.backbone(x)
        inPainting = self.sigmoid(self.InPainting_head(pen))
        return pen, inPainting.view(pen.size(0), self.in_chanels,
                                    self.mask_size, self.mask_size)

    def penultimate_forward(self, x):
        pen = self.backbone(x)
        return pen


class Contrastive_ResNet(nn.Module):
    def __init__(self, block, num_blocks, projection_size=128):
        super(Contrastive_ResNet, self).__init__()
        self.backbone = Resnet_backbone(block, num_blocks)
        self.normalize = F.normalize
        self.projection_layer = nn.Linear(512*block.expansion,
                                          projection_size)
        self.mlp_head = nn.Linear(projection_size, projection_size)

    def forward(self, x):
        pen = self.backbone(x)
        projection = self.projection_layer(pen)
        projection = self.mlp_head(F.relu(projection))
        return pen, self.normalize(projection, dim=1)

    def penultimate_forward(self, x):
        pen = self.backbone(x)
        return pen


class SimSiam_ResNet(nn.Module):
    def __init__(self, block, num_blocks, projection_size=128,
                 projection_hiden_size=2048, predictor_hiden_size=512):
        super(SimSiam_ResNet, self).__init__()
        self.backbone = Resnet_backbone(block, num_blocks)
        self.normalize = F.normalize
        self.projector = nn.Sequential(nn.Linear(512*block.expansion,
                                                 projection_hiden_size,
                                                 bias=False),
                                       nn.BatchNorm1d(projection_hiden_size),
                                       nn.ReLU(),
                                       nn.Linear(projection_hiden_size,
                                                 projection_hiden_size,
                                                 bias=False),
                                       nn.BatchNorm1d(projection_hiden_size),
                                       nn.ReLU(),
                                       nn.Linear(projection_hiden_size,
                                                 projection_size, bias=False),
                                       nn.BatchNorm1d(projection_size))
        self.predictor = nn.Sequential(nn.Linear(projection_size,
                                                 predictor_hiden_size,
                                                 bias=False),
                                       nn.BatchNorm1d(predictor_hiden_size),
                                       nn.ReLU(),
                                       nn.Linear(predictor_hiden_size,
                                                 projection_size))

    def forward(self, x):
        pen = self.backbone(x)
        projection = self.projector(pen)
        prediction = self.predictor(projection)

        projection = self.normalize(projection.detach(), dim=1)
        prediction = self.normalize(prediction, dim=1)
        return pen, projection, prediction

    def penultimate_forward(self, x):
        pen = self.backbone(x)
        return pen


def InPainting_Resnet34(args):
    return InPainting_ResNet(BasicBlock, [3, 4, 6, 3],
                             mask_size=args.mask_size)


def InPainting_Resnet18(args):
    return InPainting_ResNet(BasicBlock, [2, 2, 2, 2],
                             mask_size=args.mask_size)


def Contrastive_Resnet18(args):
    return Contrastive_ResNet(BasicBlock, [2, 2, 2, 2],
                              projection_size=args.projection_size)


def SimSiam_Resnet18(args):
    return SimSiam_ResNet(BasicBlock, [2, 2, 2, 2],
                          projection_size=args.projection_size)


def build_net(args):
    name = args.type+'_'+args.arch
    nets = {'InPainting_Resnet34': InPainting_Resnet34,
            'InPainting_Resnet18': InPainting_Resnet18,
            'SimCLR_Resnet18': Contrastive_Resnet18,
            'VICReg_Resnet18': Contrastive_Resnet18,
            'SimSiam_Resnet18': SimSiam_Resnet18,
            }
    return nets[name](args)
