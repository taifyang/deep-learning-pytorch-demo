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
	at::Tensor inputs = torch::from_blob(image.data, { 1, 1, image.rows, image.cols }, torch::kFloat32).to(torch::kCUDA);
	
	torch::Tensor outputs = module.forward({ inputs }).toTensor();
	std::cout << outputs << std::endl;
	std::cout << outputs.argmax(1) << std::endl;

	return 0;
}