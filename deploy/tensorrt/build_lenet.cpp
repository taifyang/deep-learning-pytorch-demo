// tensorRT include
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h> // onnx解析器的头文件

// cuda include
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

// system include
#include <stdio.h>
#include <fstream>
#include <assert.h>


inline const char* severity_string(nvinfer1::ILogger::Severity t) 
{
	switch (t) 
	{
	case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
	case nvinfer1::ILogger::Severity::kERROR:   return "error";
	case nvinfer1::ILogger::Severity::kWARNING: return "warning";
	case nvinfer1::ILogger::Severity::kINFO:    return "info";
	case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
	default: return "unknow";
	}
}


class TRTLogger : public nvinfer1::ILogger 
{
public:
	virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override 
	{
		if (severity <= Severity::kINFO) 
		{
			if (severity == Severity::kWARNING)
				printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
			else if (severity <= Severity::kERROR)
				printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
			else
				printf("%s: %s\n", severity_string(severity), msg);
		}
	}
} logger;


std::vector<unsigned char> load_file(const std::string& file) 
{
	std::ifstream in(file, std::ios::in | std::ios::binary);
	if (!in.is_open())
		return {};

	in.seekg(0, std::ios::end);
	size_t length = in.tellg();

	std::vector<uint8_t> data;
	if (length > 0) 
	{
		in.seekg(0, std::ios::beg);
		data.resize(length);
		in.read((char*)& data[0], length);
	}
	in.close();
	return data;
}


std::map<std::string, nvinfer1::Weights> loadWeights(const std::string& file)
{
	std::cout << "Loading weights: " << file << std::endl;

	// Open weights file
	std::ifstream input(file, std::ios::binary);
	assert(input.is_open() && "Unable to load weight file.");

	// Read number of weight blobs
	int32_t count;
	input >> count;
	assert(count > 0 && "Invalid weight map file.");

	std::map<std::string, nvinfer1::Weights> weightMap;
	while (count--)
	{
		nvinfer1::Weights wt{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
		int type;
		uint32_t size;

		// Read name and type of blob
		std::string name;
		input >> name >> std::dec >> size;
		wt.type = nvinfer1::DataType::kFLOAT;

		// Use uint32_t to create host memory to avoid additional conversion.
		uint32_t* val = new uint32_t[size];
		for (uint32_t x = 0; x < size; ++x)
		{
			input >> std::hex >> val[x];
		}
		wt.values = val;
		wt.count = size;
		weightMap[name] = wt;
	}
	return weightMap;
}


void build() 
{
	std::map<std::string, nvinfer1::Weights> mWeightMap = loadWeights("lenet.wts");
	TRTLogger logger;

	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);
	nvinfer1::ITensor* input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{ 1, 1, 28, 28 });

	auto conv1 = network->addConvolutionNd(*input, 6, nvinfer1::Dims{ 2, {5, 5} }, mWeightMap["conv.0.weight"], mWeightMap["conv.0.bias"]);
	conv1->setStride(nvinfer1::DimsHW{ 1, 1 });

	auto sigmoid1 = network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);

	auto pool1 = network->addPoolingNd(*sigmoid1->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::Dims{ 2, {2, 2} });
	pool1->setStride(nvinfer1::DimsHW{ 2, 2 });

	auto conv2 = network->addConvolutionNd(*pool1->getOutput(0), 16, nvinfer1::Dims{ 2, {5, 5} }, mWeightMap["conv.3.weight"], mWeightMap["conv.3.bias"]);
	conv2->setStride(nvinfer1::DimsHW{ 1, 1 });

	auto sigmoid2 = network->addActivation(*conv2->getOutput(0), nvinfer1::ActivationType::kSIGMOID);

	auto pool2 = network->addPoolingNd(*sigmoid2->getOutput(0), nvinfer1::PoolingType::kMAX, nvinfer1::Dims{ 2, {2, 2} });
	pool2->setStride(nvinfer1::DimsHW{ 2, 2 });

	auto ip1 = network->addFullyConnected(*pool2->getOutput(0), 120, mWeightMap["fc.0.weight"], mWeightMap["fc.0.bias"]);

	auto sigmoid3 = network->addActivation(*ip1->getOutput(0), nvinfer1::ActivationType::kSIGMOID);

	auto ip2 = network->addFullyConnected(*sigmoid3->getOutput(0), 84, mWeightMap["fc.2.weight"], mWeightMap["fc.2.bias"]);

	auto sigmoid4 = network->addActivation(*ip2->getOutput(0), nvinfer1::ActivationType::kSIGMOID);

	auto ip3 = network->addFullyConnected(*sigmoid4->getOutput(0), 10, mWeightMap["fc.4.weight"], mWeightMap["fc.4.bias"]);

	nvinfer1::ISoftMaxLayer* output = network->addSoftMax(*ip3->getOutput(0));
	output->getOutput(0)->setName("output");
	network->markOutput(*output->getOutput(0));

	nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	if (engine == nullptr) 
	{
		printf("Build engine failed.\n");
		return;
	}
	
	nvinfer1::IHostMemory* model_data = engine->serialize();
	FILE* f = fopen("lenet.engine", "wb");
	fwrite(model_data->data(), 1, model_data->size(), f);
	fclose(f);

	model_data->destroy();
	engine->destroy();
	network->destroy();
	config->destroy();
	builder->destroy();
}

void inference() 
{
	// ------------------------------ 1. 准备模型并加载   ----------------------------
	TRTLogger logger;
	auto engine_data = load_file("lenet.engine");
	// 执行推理前，需要创建一个推理的runtime接口实例。与builer一样，runtime需要logger：
	nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
	// 将模型从读取到engine_data中，则可以对其进行反序列化以获得engine
	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
	if (engine == nullptr) 
	{
		printf("Deserialize cuda engine failed.\n");
		runtime->destroy();
		return;
	}

	nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();
	cudaStream_t stream = nullptr;
	// 创建CUDA流，以确定这个batch的推理是独立的
	cudaStreamCreate(&stream);

	// ------------------------------ 2. 准备好要推理的数据并搬运到GPU   ----------------------------
	cv::Mat image = cv::imread("10.png", -1);
	std::vector<uint8_t> fileData(image.cols * image.rows);
	fileData = (std::vector<uint8_t>)(image.reshape(1, 1));
	int input_numel = 1 * 1 * image.rows * image.cols;

	float* input_data_host = nullptr;
	cudaMallocHost(&input_data_host, input_numel * sizeof(float));
	for (int i = 0; i < image.cols * image.rows; i++)
	{
		input_data_host[i] = float(fileData[i] / 255.0);
	}

	float* input_data_device = nullptr;
	float output_data_host[10];
	float* output_data_device = nullptr;
	cudaMalloc(&input_data_device, input_numel * sizeof(float));
	cudaMalloc(&output_data_device, sizeof(output_data_host));

	cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream);

	// 用一个指针数组指定input和output在gpu中的指针
	float* bindings[] = { input_data_device, output_data_device };

	// ------------------------------ 3. 推理并将结果搬运回CPU   ----------------------------
	bool success = execution_context->enqueueV2((void**)bindings, stream, nullptr);
	cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	int predict_label = std::max_element(output_data_host, output_data_host + 10) - output_data_host;
	std::cout <<"predict_label: " << predict_label << std::endl;

	// ------------------------------ 4. 释放内存 ----------------------------
	cudaStreamDestroy(stream);
	execution_context->destroy();
	engine->destroy();
	runtime->destroy();
}


int main() 
{
	build();
	inference();
	
	return 0;
}