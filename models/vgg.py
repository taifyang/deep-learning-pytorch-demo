import torch


conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
fc_features = 512 * 7 * 7 # c * w * h 经过5个vgg_block, 宽高会减半5次, 变成 224/32 = 7
fc_hidden_units = 4096 # 任意
ratio = 8
small_conv_arch = [(1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio), (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio)]


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(torch.nn.ReLU())
    blk.append(torch.nn.MaxPool2d(kernel_size=2, stride=2)) # 这里会使宽高减半
    return torch.nn.Sequential(*blk)


class vgg(torch.nn.Module):
    def __init__(self, conv_arch, fc_features, fc_hidden_units=4096):
        super(vgg, self).__init__()
        self.conv = torch.nn.Sequential()
        # 卷积层部分
        for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
            self.conv.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
        # 全连接层部分
        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc", torch.nn.Sequential(torch.nn.Linear(fc_features, fc_hidden_units),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(0.5),
                                    torch.nn.Linear(fc_hidden_units, fc_hidden_units),
                                    torch.nn.ReLU(),
                                    torch.nn.Dropout(0.5),
                                    torch.nn.Linear(fc_hidden_units, 10)
                                    ))
    
    def forward(self, img):       
        feature = self.conv(img)
        flat = feature.view(feature.shape[0], -1)
        output = self.fc(flat)
        return output


net = vgg(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)
print(net)
print('parameters:', sum(param.numel() for param in net.parameters())) 