#include <iostream>
#include <opencv2/opencv.hpp>


int main(int argc, char* argv[])
{
	std::string model = "lenet.onnx";
	cv::dnn::Net net = cv::dnn::readNet(model);

	cv::Mat image = cv::imread("10.png", 0), blob;
	cv::dnn::blobFromImage(image, blob, 1. / 255., cv::Size(28, 28), cv::Scalar(), true, false);

	net.setInput(blob);
	std::vector<cv::Mat> outputs;
	net.forward(outputs, net.getUnconnectedOutLayersNames());

	std::vector<float> values;
	for (size_t i = 0; i < outputs[0].cols; i++)
	{
		values.push_back(outputs[0].at<float>(0, i));
	}
	std::cout << std::distance(values.begin(), std::max_element(values.begin(), values.end())) << std::endl;

	return 0;
}