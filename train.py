import torch
import torchvision
import time
import argparse
import importlib


def parse_args():
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size in training')
    parser.add_argument('--num_epochs', default=5, type=int, help='number of epoch in training')
    parser.add_argument('--model',  default='mlp', help='model name [default: mlp]')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    model = importlib.import_module('models.'+args.model) 
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model.net.to(device)

    loss = torch.nn.CrossEntropyLoss()
    if args.model == 'mlp':
        optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
          
    train_path = r'./Datasets/mnist_png/training'
    test_path = r'./Datasets/mnist_png/testing'
    transform_list = [torchvision.transforms.Grayscale(num_output_channels=1), torchvision.transforms.ToTensor()]
    if args.model == 'alexnet' or args.model == 'vgg':
        transform_list.append(torchvision.transforms.Resize(size=224))
    if args.model == 'googlenet' or args.model == 'resnet':
        transform_list.append(torchvision.transforms.Resize(size=96))
    transform = torchvision.transforms.Compose(transform_list)

    train_dataset = torchvision.datasets.ImageFolder(train_path, transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(test_path, transform=transform)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for epoch in range(num_epochs):
            train_l, train_acc, test_acc, m, n, batch_count, start = 0.0, 0.0, 0.0, 0, 0, 0, time.time()
            for X, y in train_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                train_l += l.cpu().item()
                train_acc += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                m += y.shape[0]
                batch_count += 1
            with torch.no_grad():
                for X, y in test_iter:
                    net.eval() # 评估模式
                    test_acc += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                    net.train() # 改回训练模式
                    n += y.shape[0]
            print('epoch %d, loss %.6f, train acc %.3f, test acc %.3f, time %.1fs'% (epoch, train_l / batch_count, train_acc / m, test_acc / n, time.time() - start))
            torch.save(net, args.model+".pth")