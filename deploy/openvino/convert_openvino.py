from openvino.tools import mo
from openvino.runtime import serialize


if __name__ == "__main__":
    onnx_path = f"./lenet.onnx"

    # fp32 IR model
    fp32_path = f"./lenet/lenet_fp32.xml"
    print(f"Export ONNX to OpenVINO FP32 IR to: {fp32_path}")
    model = mo.convert_model(onnx_path)
    serialize(model, fp32_path)

    # fp16 IR model
    fp16_path = f"./lenet/lenet_fp16.xml"
    print(f"Export ONNX to OpenVINO FP16 IR to: {fp16_path}")
    model = mo.convert_model(onnx_path, compress_to_fp16=True)
    serialize(model, fp16_path)
