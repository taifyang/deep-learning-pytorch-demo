import cv2
import numpy as np


img = cv2.imread("10.png", 0)
net = cv2.dnn.readNet("lenet.onnx")
blob = cv2.dnn.blobFromImage(img, 1/255., size=(28,28), swapRB=True, crop=False)
net.setInput(blob)
pred = net.forward()
print(np.argmax(pred))