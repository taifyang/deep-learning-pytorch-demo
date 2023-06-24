#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h> 


int main(int argc, char* argv[])
{
	std::string model = argv[1];
	torch::jit::script::Module module = torch::jit::load(model);
	module.to(torch::kCUDA);

	cv::Mat image = cv::imread(argv[2], 0);
	image.convertTo(image, CV_32F, 1.0 / 255);
	if (model.find("alexnet.pt") != std::string::npos || model.find("vgg.pt") != std::string::npos)
		cv::resize(image, image, cv::Size(224, 224));
	if (model.find("googlenet.pt") != std::string::npos || model.find("resnet.pt") != std::string::npos)
		cv::resize(image, image, cv::Size(96, 96));
	at::Tensor img_tensor = torch::from_blob(image.data, { 1, 1, image.rows, image.cols}, torch::kFloat32).to(torch::kCUDA);

	torch::Tensor result = module.forward({ img_tensor }).toTensor();
	std::cout << result << std::endl;
	std::cout << result.argmax(1) << std::endl;

	return 0;
}