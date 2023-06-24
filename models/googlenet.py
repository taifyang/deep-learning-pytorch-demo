import torch


class Inception(torch.nn.Module):
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = torch.nn.Conv2d(in_c, c1, kernel_size=1)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = torch.nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = torch.nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = torch.nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = torch.nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = torch.nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = torch.nn.functional.relu(self.p1_1(x))
        p2 = torch.nn.functional.relu(self.p2_2(torch.nn.functional.relu(self.p2_1(x))))
        p3 = torch.nn.functional.relu(self.p3_2(torch.nn.functional.relu(self.p3_1(x))))
        p4 = torch.nn.functional.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出
    

class GoogleNet(torch.nn.Module):
    def __init__(self):
        super(GoogleNet, self).__init__()
        self.b1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   torch.nn.ReLU(),
                   torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b2 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, kernel_size=1),
                        torch.nn.Conv2d(64, 192, kernel_size=3, padding=1),
                        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b3 = torch.nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                        Inception(256, 128, (128, 192), (32, 96), 64),
                        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b4 = torch.nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                        Inception(512, 160, (112, 224), (24, 64), 64),
                        Inception(512, 128, (128, 256), (24, 64), 64),
                        Inception(512, 112, (144, 288), (32, 64), 64),
                        Inception(528, 256, (160, 320), (32, 128), 128),
                        torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.b5 = torch.nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                        Inception(832, 384, (192, 384), (48, 128), 128))       
        self.fc = torch.nn.Linear(1024, 10)    
        
    def forward(self, img): 
        out1 = self.b1(img)
        out2 = self.b2(out1)
        out3 = self.b3(out2)
        out4 = self.b4(out3)
        out5 = self.b5(out4)
        out6 = torch.nn.functional.avg_pool2d(out5, kernel_size=out5.size()[2:])
        flat = out6.view(out6.shape[0], -1)
        out = self.fc(flat)
        return out
  

net = GoogleNet()
print(net)
print('parameters:', sum(param.numel() for param in net.parameters()))      