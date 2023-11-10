# deep-learning-pytorch-demo

Train, test, export and deployment demo based on Pytorch.
The models including MLP, LeNet, AlexNet, vgg, GoogleNet, ResNet.

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
