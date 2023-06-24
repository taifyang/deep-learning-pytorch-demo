import cv2
import torch
import argparse
import importlib
from pathlib import Path
import torchvision.transforms.functional


def parse_args():
    parser = argparse.ArgumentParser('testing')
    parser.add_argument('--model',  default='mlp', help='model name [default: mlp]')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    model = importlib.import_module('models.' + args.model) 
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = model.net.to(device)
    net = torch.load(args.model+'.pth')
    net.eval()

    with torch.no_grad():
        imgs_path = Path(r"./Datasets/mnist_png/testing/0/").glob("*")
        for img_path in imgs_path:
            img = cv2.imread(str(img_path), 0)
            if args.model == 'alexnet' or args.model == 'vgg':
                img = cv2.resize(img, (224,224))
            if args.model == 'googlenet' or args.model == 'resnet':
                img = cv2.resize(img, (96,96))
            img_tensor = torchvision.transforms.functional.to_tensor(img)
            img_tensor = torch.unsqueeze(img_tensor, 0)
            print(net(img_tensor.to(device)).argmax(dim=1).item())