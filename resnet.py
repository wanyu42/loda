from numpy.random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet_9

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class Net(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, random=False):
        """__init__.
        Parameters
        ----------
        block :
            block
        num_blocks :
            num_blocks
        num_classes :
            num_classes
        """
        super(Net, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.dropout = nn.Dropout(p=0.8)
        # self.retrain= retrain
        self.random = random

        self.penultimate = None
        self.previous = None
        #self.penultimate = list()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, mask_update=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        # if self.random == True:
        #     out = self.dropout(out)
        out = self.layer2(out)
        # if self.random == True:
        #     out = self.dropout(out)
        out = self.layer3(out)
        if self.random == True:
            out = self.dropout(out)
        out = self.layer4(out)
        # self.previous = out.clone()
        self.previous = out
        # import ipdb; ipdb.set_trace()
        #self.penultimate.append(out.clone())

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # self.penultimate = out.clone()
        self.penultimate = out

        #self.penultimate.append(out.clone())
        if self.random == True:
            out = self.dropout(out)

        out = self.linear(out)
        return out

def ResNet9(random=False):
    return resnet_9.Net(random=random)

def ResNet18(num_classes=10, random=False):
    return Net(BasicBlock, [2,2,2,2], num_classes=num_classes, random=random)

def ResNet34(random=False):
    return Net(BasicBlock, [3,4,6,3], random=random)

def ResNet50(random=False):
    return Net(Bottleneck, [3,4,6,3], random=random)

def ResNet101():
    return Net(Bottleneck, [3,4,23,3])

def ResNet152():
    return Net(Bottleneck, [3,8,36,3])


def test(model, device, test_loader):
    model.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            loss = F.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss += loss.item()

    test_loss /= len(test_loader)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return 100. * correct / len(test_loader.dataset)

def train(model, device, train_loader, optimizer, epoch, defense=None):
    model.train()

    # lr = util.adjust_learning_rate(optimizer, epoch, args) # don't need it if we use Adam
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = torch.tensor(data).to(device), torch.tensor(target).to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
        loss.backward()

        if defense is not None: 
            model = defense.model_apply(model)

        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.shape[0]

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f} || Acc: {:.3f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss/(batch_idx + 1), 100.*correct/total))