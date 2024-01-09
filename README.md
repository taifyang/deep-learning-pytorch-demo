# deep-learning-pytorch-demo

Train, test, export demo is based on Pytorch.
The models include MLP, LeNet, AlexNet, vgg, GoogleNet, ResNet.
The supported model types include torchscript, onnx, openvino, engine.

## install
```shell
git clone https://github.com/taifyang/deep-learning-pytorch-demo  # clone
cd deep-learning-pytorch-demo
pip install -r requirements.txt  # install
```

## Data Preparation
Download dataset **mnist** [here](https://github.com/myleott/mnist_png/blob/master/mnist_png.tar.gz)  and save in `./Dataset/mnist_png/`.

## simple usage
```shell
## train 
python train.py --models lenet

## test
python test.py --models lenet

## export
python export.py --models lenet --type onnx
```

## Reference By
[d2l-zh-pytorch](https://zh.d2l.ai/index.html)<br>

