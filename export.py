import torch
import argparse
import importlib


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

    if args.model == 'mlp' or args.model == 'lenet':
        x = torch.rand(1, 1, 28, 28)
    if args.model == 'alexnet' or args.model == 'vgg':
        x = torch.rand(1, 1, 224, 224)
    if args.model == 'googlenet' or args.model == 'resnet':
        x = torch.rand(1, 1, 96, 96)
    x = x.to(device)

    traced_script_module = torch.jit.trace(net, x)
    traced_script_module.save(args.model+".pt")

    torch.onnx.export(net,
                    x,
                    args.model+".onnx",
                    opset_version = 11
                    )