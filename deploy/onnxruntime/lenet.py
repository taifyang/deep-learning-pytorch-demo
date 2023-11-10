import cv2
import numpy as np
import onnxruntime


img = cv2.imread("10.png", 0)
blob = cv2.dnn.blobFromImage(img, 1/255., size=(28,28), swapRB=True, crop=False)
onnx_session = onnxruntime.InferenceSession("enet.onnx", providers=['CPUExecutionProvider'])

input_name=[]
for node in onnx_session.get_inputs():
    input_name.append(node.name)

output_name=[]
for node in onnx_session.get_outputs():
    output_name.append(node.name)

input_feed={}
for name in input_name:
    input_feed[name] = blob

pred = onnx_session.run(None, input_feed)[0]
print(np.argmax(pred))