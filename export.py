import torch
import argparse
import importlib


def parse_args():
    parser = argparse.ArgumentParser('exporting')
    parser.add_argument('--model',  default='lenet', help='model name [default: lenet]')
    parser.add_argument('--type', default='onnx', help='torchscript, onnx, openvino, engine')
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

    if args.type == 'torchscript':
        traced_script_module = torch.jit.trace(net, x)
        traced_script_module.save(args.model+".pt")
    if args.type == 'onnx':
        torch.onnx.export(net, x, args.model+".onnx", opset_version = 11)
    if args.type == 'openvino':
        torch.onnx.export(net, x, args.model+".onnx", opset_version = 11)
        from openvino.tools import mo
        from openvino.runtime import serialize
        model = mo.convert_model(args.model+".onnx", compress_to_fp16=False)
        serialize(model, args.model+".xml")
    if args.type == 'engine':
        torch.onnx.export(net, x, args.model+".onnx", opset_version = 11)
        import tensorrt as trt
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        success = parser.parse_from_file(args.model+".onnx")
        config = builder.create_builder_config()
        serialized_engine = builder.build_serialized_network(network, config)
        with open(args.model+".engine", "wb") as f:
            f.write(serialized_engine)