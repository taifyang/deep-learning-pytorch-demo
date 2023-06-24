import torch


class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(2, 2), # kernel_size, stride
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.Sigmoid(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(16*4*4, 120),
            torch.nn.Sigmoid(),
            torch.nn.Linear(120, 84),
            torch.nn.Sigmoid(),
            torch.nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        flat = feature.view(img.shape[0], -1)
        output = self.fc(flat)
        return output
 
 
net = LeNet()   
print(net)
print('parameters:', sum(param.numel() for param in net.parameters()))