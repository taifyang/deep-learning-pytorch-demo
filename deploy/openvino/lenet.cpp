#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>


int main(int argc, char* argv[])
{
	ov::Core core;
	//auto model = core.compile_model("lenet.onnx", "CPU");             
	auto model = core.compile_model("lenet/lenet_fp16.xml", "CPU");
	auto iq = model.create_infer_request();
	auto input = iq.get_input_tensor(0);
	auto outputs = iq.get_output_tensor(0);
	input.set_shape({ 1, 1, 28, 28 });
	float* inputs = input.data<float>();

	cv::Mat image = cv::imread("10.png", 0);
	image.convertTo(image, CV_32F, 1.0 / 255);
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			inputs[i * 28 + j] = image.at<float>(i, j);
		}
	}

	iq.infer();

	float* pred = outputs.data<float>();
	int predict_label = std::max_element(pred, pred + 10) - pred;
	std::cout << predict_label << std::endl;

	return 0;
}