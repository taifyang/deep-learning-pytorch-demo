#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h> 


int main(int argc, char* argv[])
{
	std::string model = "lenet.pt";
	torch::jit::script::Module module = torch::jit::load(model);
	module.to(torch::kCUDA);

	cv::Mat image = cv::imread("10.png", 0);
	image.convertTo(image, CV_32F, 1.0 / 255);
	at::Tensor img_tensor = torch::from_blob(image.data, { 1, 1, image.rows, image.cols }, torch::kFloat32).to(torch::kCUDA);
	
	torch::Tensor result = module.forward({ img_tensor }).toTensor();
	std::cout << result << std::endl;
	std::cout << result.argmax(1) << std::endl;

	return 0;
}