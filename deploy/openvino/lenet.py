import cv2
import numpy as np
from openvino.inference_engine import IECore


img = cv2.imread("10.png", 0)
blob = cv2.dnn.blobFromImage(img, 1/255., size=(28,28), swapRB=True, crop=False)
ie = IECore()
#net = ie.read_network(model="lenet.onnx")
net = ie.read_network(model="lenet/lenet_fp16.xml", weights="lenet/lenet_fp16.bin")
exec_net = ie.load_network(network=net, device_name="CPU")
input_layer = next(iter(net.input_info))
infer_request_handle = exec_net.start_async(request_id=0, inputs={input_layer: blob})
if infer_request_handle.wait(-1) == 0:
    output_layer = infer_request_handle._outputs_list[0]
    outputs = infer_request_handle.output_blobs[output_layer] 
    #outputs = infer_request_handle.output_blobs["23"]
    print(np.argmax(outputs.buffer))
