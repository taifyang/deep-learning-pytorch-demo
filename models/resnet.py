import torch
    
    
class Residual(torch.nn.Module): 
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = torch.nn.functional.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return torch.nn.functional.relu(Y + X)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return torch.nn.Sequential(*blk)


class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64), 
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block1 = resnet_block(64, 64, 2, first_block=True)
        self.block2 = resnet_block(64, 128, 2)
        self.block3 = resnet_block(128, 256, 2)
        self.block4 = resnet_block(256, 512, 2)
        self.fc = torch.nn.Linear(512, 10)                 
        
    def forward(self, img):
        out0 = self.net(img)
        out1 = self.block1(out0)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        pool = torch.nn.functional.avg_pool2d(out4, kernel_size=out4.size()[2:])
        flat = pool.view(pool.shape[0], -1)
        out = self.fc(flat)
        return out


net = ResNet()
print(net)
print('parameters:', sum(param.numel() for param in net.parameters()))