import torch
from models.lenet import net
import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = net.to(device)
net = torch.load('lenet.pth')  
weights = net.state_dict()


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
config = builder.create_builder_config()
runtime = trt.Runtime(TRT_LOGGER)
config.set_flag(trt.BuilderFlag.REFIT)
config.max_workspace_size = 1 << 30


input_tensor = network.add_input(name="input", dtype=trt.float32, shape=(1, 1, 28, 28))

conv1_w = weights['conv.0.weight'].cpu().numpy()
conv1_b = weights['conv.0.bias'].cpu().numpy()
conv1 = network.add_convolution(input=input_tensor, num_output_maps=6, kernel_shape=(5, 5), kernel=conv1_w, bias=conv1_b)
conv1.stride = (1, 1)

sigmoid1 = network.add_activation(input=conv1.get_output(0), type=trt.ActivationType.SIGMOID)

pool1 = network.add_pooling(input=sigmoid1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
pool1.stride = (2, 2)

conv2_w = weights['conv.3.weight'].cpu().numpy()
conv2_b = weights['conv.3.bias'].cpu().numpy()
conv2 = network.add_convolution(input=pool1.get_output(0), num_output_maps=16,  kernel_shape=(5, 5), kernel=conv2_w, bias=conv2_b)
conv2.stride = (1, 1)

sigmoid2 = network.add_activation(input=conv2.get_output(0), type=trt.ActivationType.SIGMOID)

pool2 = network.add_pooling(sigmoid2.get_output(0), trt.PoolingType.MAX, (2, 2))
pool2.stride = (2, 2)

fc1_w = weights['fc.0.weight'].cpu().numpy()
fc1_b = weights['fc.0.bias'].cpu().numpy()
fc1 = network.add_fully_connected(input=pool2.get_output(0), num_outputs=120, kernel=fc1_w, bias=fc1_b)

sigmoid3 = network.add_activation(input=fc1.get_output(0), type=trt.ActivationType.SIGMOID)

fc2_w = weights['fc.2.weight'].cpu().numpy()
fc2_b = weights['fc.2.bias'].cpu().numpy()
fc2 = network.add_fully_connected(sigmoid3.get_output(0), num_outputs=84, kernel=fc2_w, bias=fc2_b)

sigmoid4 = network.add_activation(input=fc2.get_output(0), type=trt.ActivationType.SIGMOID)

fc3_w = weights['fc.4.weight'].cpu().numpy()
fc3_b = weights['fc.4.bias'].cpu().numpy()
fc3 = network.add_fully_connected(sigmoid4.get_output(0), num_outputs=10, kernel=fc3_w, bias=fc3_b)

fc3.get_output(0).name = "output"
network.mark_output(tensor=fc3.get_output(0))


plan = builder.build_serialized_network(network, config)
engine = runtime.deserialize_cuda_engine(plan)
context = engine.create_execution_context()

h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
stream = cuda.Stream()


img = cv2.imread("10.png", 0)
blob = cv2.dnn.blobFromImage(img, 1/255., size=(28,28), swapRB=True, crop=False)
np.copyto(h_input, blob.ravel())

with engine.create_execution_context() as context:
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    outputs = np.argmax(h_output)
    print(outputs)