#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>


int main(int argc, char* argv[])
{
	Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "lenet");
	Ort::SessionOptions session_options;
	session_options.SetIntraOpNumThreads(1);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

	OrtCUDAProviderOptions cuda_option;
	cuda_option.device_id = 0;
	cuda_option.arena_extend_strategy = 0;
	cuda_option.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
	cuda_option.gpu_mem_limit = SIZE_MAX;
	cuda_option.do_copy_in_default_stream = 1;
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
	session_options.AppendExecutionProvider_CUDA(cuda_option);

	const wchar_t* model_path = L"lenet.onnx";
	Ort::Session session(env, model_path, session_options);
	Ort::AllocatorWithDefaultOptions allocator;

	std::vector<const char*>  input_node_names;
	for (size_t i = 0; i < session.GetInputCount(); i++)
	{
		input_node_names.push_back(session.GetInputName(i, allocator));
	}

	std::vector<const char*> output_node_names;
	for (size_t i = 0; i < session.GetOutputCount(); i++)
	{
		output_node_names.push_back(session.GetOutputName(i, allocator));
	}

	const size_t input_tensor_size = 1 * 28 * 28;
	std::vector<float> input_tensor_values(input_tensor_size);

	cv::Mat image = cv::imread("10.png", 0);
	image.convertTo(image, CV_32F, 1.0 / 255);
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			input_tensor_values[i * 28 + j] = image.at<float>(i, j);
		}
	}

	std::vector<int64_t> input_node_dims = { 1, 1, 28, 28 };
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size());

	std::vector<Ort::Value> inputs;
	inputs.push_back(std::move(input_tensor));

	std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), inputs.data(), input_node_names.size(), output_node_names.data(), output_node_names.size());

	const float* rawOutput = output_tensors[0].GetTensorData<float>();
	std::vector<int64_t> outputShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	size_t count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
	std::vector<float> output(rawOutput, rawOutput + count);

	int predict_label = std::max_element(output.begin(), output.end()) - output.begin();
	std::cout << predict_label << std::endl;

	return 0;
}