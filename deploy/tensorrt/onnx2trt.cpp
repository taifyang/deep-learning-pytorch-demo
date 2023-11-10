#include <NvInfer.h> // �����õ�ͷ�ļ�
#include <NvOnnxParser.h> // onnx��������ͷ�ļ�
#include <NvInferRuntime.h> // �����õ�����ʱͷ�ļ�

#include <cuda_runtime.h> // cuda include

#include <stdio.h>


inline const char* severity_string(nvinfer1::ILogger::Severity t) {
	switch (t) {
	case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
	case nvinfer1::ILogger::Severity::kERROR:   return "error";
	case nvinfer1::ILogger::Severity::kWARNING: return "warning";
	case nvinfer1::ILogger::Severity::kINFO:    return "info";
	case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
	default: return "unknow";
	}
}

class TRTLogger : public nvinfer1::ILogger {
public:
	virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
		if (severity <= Severity::kINFO) {
			if (severity == Severity::kWARNING) 
				printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
			else if (severity <= Severity::kERROR) 
				printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
			else 
				printf("%s: %s\n", severity_string(severity), msg);
		}
	}
} logger;


bool build_model() {
	TRTLogger logger;

	// ----------------------------- 1. ���� builder, config ��network -----------------------------
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);

	// ----------------------------- 2. ���룬ģ�ͽṹ������Ļ�����Ϣ -----------------------------
	// ͨ��onnxparser�����Ľ������䵽network�У�����addConv�ķ�ʽ��ӽ�ȥ
	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
	if (!parser->parseFromFile("lenet.onnx", 1)) {
		printf("Failed to parser onnx\n");
		return false;
	}

	int maxBatchSize = 1;
	printf("Workspace Size = %.2f MB\n", (1 << 30) / 1024.0f / 1024.0f);
	config->setMaxWorkspaceSize(1 << 30);

	// --------------------------------- 2.1 ����profile ----------------------------------
	// ���ģ���ж�����룬�������profile
	auto profile = builder->createOptimizationProfile();
	auto input_tensor = network->getInput(0);
	int input_channel = input_tensor->getDimensions().d[1];

	// �����������С�����š����ķ�Χ
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, input_channel, 28, 28));
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, input_channel, 28, 28));
	profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, input_channel, 28, 28));
	// ��ӵ�����
	config->addOptimizationProfile(profile);

	nvinfer1::ICudaEngine * engine = builder->buildEngineWithConfig(*network, *config);
	if (engine == nullptr) {
		printf("Build engine failed.\n");
		return false;
	}

	// -------------------------- 3. ���л� ----------------------------------
	// ��ģ�����л���������Ϊ�ļ�
	nvinfer1::IHostMemory* model_data = engine->serialize();
	FILE* f = fopen("lenet.engine", "wb");
	fwrite(model_data->data(), 1, model_data->size(), f);
	fclose(f);

	// ж��˳���չ���˳����
	model_data->destroy();
	parser->destroy();
	engine->destroy();
	network->destroy();
	config->destroy();
	builder->destroy();

	return true;
}

int main() {
	build_model();
	return 0;
}