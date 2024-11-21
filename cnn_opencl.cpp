#pragma warning(disable : 4996)
#include "cnn.h"
#include <ctime>
#include <CL/cl.h>
#include <cmath>
#include <iostream>
#include <fstream>

#define CHECK_ERROR(err) \
	if(err != CL_SUCCESS) { \
		printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err);\
		exit(EXIT_FAILURE); \
	}

cl_int Err;
cl_platform_id Platform;
cl_device_id Device;
cl_context Context;
cl_command_queue Queue;

// kernels
cl_kernel ConvolutionKernel;
cl_kernel FCLayerKernel;
cl_kernel MaxPoolingKernel;

const int INPUT_DIM[] = {
	3, 64,
	64,

	64,128,
	128,

	128, 256, 256,
	256,

	256, 512, 512,
	512,

	512, 512, 512,
	512,

	512,
	512,
	512
};

const int OUTPUT_DIM[] = {
	64, 64,
	64,

	128, 128,
	128,

	256, 256, 256,
	256,

	512, 512, 512,
	512,

	512, 512, 512,
	512,

	512,
	512,
	10
};

// 이미지 파일의 크기
const int NBYN[] = {
	32, 32,
	16,

	16, 16,
	8,

	8, 8, 8,
	4,

	4, 4, 4,
	2,

	2, 2, 2,
	1,

	1,
	1,
	1
};

// 유틸리티 함수: OpenCL 소스 로드
char* get_source_code(const char* file_name, size_t* len) {
	FILE* file = fopen(file_name, "rb");
	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	size_t length = (size_t)ftell(file);
	rewind(file);

	char* source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);
	source_code[length] = '\0';
	fclose(file);
	*len = length;

	return source_code;
}

void build_error(cl_program program, cl_device_id device, cl_int err) {
	if (err == CL_BUILD_PROGRAM_FAILURE) {
		size_t log_size;
		char* log;

		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		CHECK_ERROR(err);

		log = (char*)malloc(log_size + 1);
		err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		CHECK_ERROR(err);

		log[log_size] = '\0';
		printf("Compiler error:\n%s\n", log);
		free(log);
		exit(0);
	};
}

void cnn_init() {
	// Platform ID
	Err = clGetPlatformIDs(1, &Platform, NULL);
	CHECK_ERROR(Err);

	// Device ID
	Err = clGetDeviceIDs(Platform, CL_DEVICE_TYPE_GPU, 1, &Device, NULL);
	CHECK_ERROR(Err);

	// Create Context
	Context = clCreateContext(NULL, 1, &Device, NULL, NULL, &Err);
	CHECK_ERROR(Err);

	// Create Command Queue
	Queue = clCreateCommandQueueWithProperties(Context, Device, 0, &Err);
	CHECK_ERROR(Err);

	// Create Program Object
	size_t kernel_source_size;
	char* kernel_source = get_source_code("kernel.cl", &kernel_source_size);
	cl_program program = clCreateProgramWithSource(Context, 1, (const char**)&kernel_source, &kernel_source_size, &Err);
	CHECK_ERROR(Err);

	// Build Program
	Err = clBuildProgram(program, 1, &Device, "", NULL, NULL);
	build_error(program, Device, Err);
	CHECK_ERROR(Err);
	//============================= 이 위로는 공통 코드 =============================


	ConvolutionKernel = clCreateKernel(program, "convolution", &Err);
	CHECK_ERROR(Err);

	FCLayerKernel = clCreateKernel(program, "fc_layer", &Err);
	CHECK_ERROR(Err);

	MaxPoolingKernel = clCreateKernel(program, "max_pooling", &Err);
	CHECK_ERROR(Err);
}

void convolution_cl(float* inputs, float* outputs, float* filter, float* biases, int inDim, int outDim, int nbyn) {
	size_t global_work_size[] = { (size_t)nbyn, (size_t)nbyn }; // 병렬 처리를 위한 워크 아이템 크기

	// ================== 버퍼 생성 ==================
	cl_mem input_buffer = clCreateBuffer(
		Context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * inDim * nbyn * nbyn, inputs,
		&Err
	);
	CHECK_ERROR(Err);

	cl_mem filter_buffer = clCreateBuffer(
		Context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * inDim * outDim * 3 * 3, filter,
		&Err
	);
	CHECK_ERROR(Err);

	cl_mem bias_buffer = clCreateBuffer(
		Context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * outDim, biases,
		&Err
	);
	CHECK_ERROR(Err);

	cl_mem output_buffer = clCreateBuffer(
		Context,
		CL_MEM_WRITE_ONLY,
		sizeof(float) * outDim * nbyn * nbyn,
		NULL,
		&Err
	);
	CHECK_ERROR(Err);

	// ================== Kernel Arguments ==================
	clSetKernelArg(ConvolutionKernel, 0, sizeof(cl_mem), &input_buffer);
	clSetKernelArg(ConvolutionKernel, 1, sizeof(cl_mem), &output_buffer);
	clSetKernelArg(ConvolutionKernel, 2, sizeof(cl_mem), &filter_buffer);
	clSetKernelArg(ConvolutionKernel, 3, sizeof(cl_mem), &bias_buffer);
	clSetKernelArg(ConvolutionKernel, 4, sizeof(int), &inDim);
	clSetKernelArg(ConvolutionKernel, 5, sizeof(int), &outDim);
	clSetKernelArg(ConvolutionKernel, 6, sizeof(int), &nbyn);

	// ================== 실행하고 결과 받기 ==================
	Err = clEnqueueNDRangeKernel(Queue, ConvolutionKernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
	CHECK_ERROR(Err);
	Err = clEnqueueReadBuffer(Queue, output_buffer, CL_TRUE, 0,
		sizeof(float) * outDim * nbyn * nbyn, outputs, 0, NULL, NULL);
	CHECK_ERROR(Err);

	Err = clFinish(Queue);
	CHECK_ERROR(Err);

	// ================== 메모리 해제 ==================
	clReleaseMemObject(input_buffer);
	clReleaseMemObject(filter_buffer);
	clReleaseMemObject(bias_buffer);
	clReleaseMemObject(output_buffer);
}

static void convolution(float* inputs, float* outputs, float* filter, float* biases, int inDim, int outDim, int nbyn) {

	memset(outputs, 0, nbyn * nbyn * outDim * sizeof(float));
	int x = 0, y = 0;
	int offset = nbyn * nbyn;
	float sum = 0, temp;
	float* input, * output;

	for (int outNeuron = 0; outNeuron < outDim; ++outNeuron) {
		input = inputs;
		for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
			output = outputs;
			for (int row = 0; row < nbyn; ++row) {
				for (int col = 0; col < nbyn; ++col) {
					sum = 0;
					for (int fRow = 0; fRow < 3; ++fRow) {
						for (int fCol = 0; fCol < 3; ++fCol) {
							x = col + fCol - 1;
							y = row + fRow - 1;

							if (x >= 0 && x < nbyn && y >= 0 && y < nbyn) {
								sum += input[nbyn * y + x] * filter[3 * fRow + fCol];
							}

						}
					}
					*(output++) += sum;
				}
			}
			filter += 9;
			input += offset;

		}
		for (int i = 0; i < offset; ++i) {
			(*outputs) = (*outputs) + (*biases);
			if (*outputs < 0) (*outputs) = 0;	//ReLU
			outputs++;
		}
		++biases;
	}

}

void max_pooling_cl(float* input, float* output, int dim, int nbyn) {

	cl_mem input_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * dim * nbyn * nbyn, input, &Err);
	CHECK_ERROR(Err);

	cl_mem output_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(float) * dim * nbyn * nbyn / 4, NULL, &Err);
	CHECK_ERROR(Err);

	// Set Kernel Args
	Err = clSetKernelArg(MaxPoolingKernel, 0, sizeof(cl_mem), &input_buffer);
	CHECK_ERROR(Err);
	Err = clSetKernelArg(MaxPoolingKernel, 1, sizeof(cl_mem), &output_buffer);
	CHECK_ERROR(Err);
	Err = clSetKernelArg(MaxPoolingKernel, 2, sizeof(cl_int), &nbyn);
	CHECK_ERROR(Err);

	// Set Work Size
	size_t global_item_size[] = { (size_t)dim, (size_t)(nbyn * nbyn / 4) };
	size_t local_item_size[] = { (size_t)nbyn / 2, (size_t)nbyn / 2 };

	// Run Kernel
	Err = clEnqueueNDRangeKernel(Queue, MaxPoolingKernel, 2, NULL, global_item_size, local_item_size, 0, NULL, NULL);
	CHECK_ERROR(Err);

	Err = clEnqueueReadBuffer(Queue, output_buffer, CL_TRUE, 0, sizeof(float) * dim * nbyn * nbyn / 4, output, 0, NULL, NULL);
	CHECK_ERROR(Err);

	Err = clFinish(Queue);
	CHECK_ERROR(Err);

	// Release Memory
	clReleaseMemObject(input_buffer);
	clReleaseMemObject(output_buffer);
}

void fc_layer_cl(float* inputs, float* outputs, float* weights, float* biases, int inDim, int outDim) {

	// ================== 버퍼 생성 ==================
	cl_mem input_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * inDim, inputs, &Err);
	CHECK_ERROR(Err);

	cl_mem output_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE,
		sizeof(float) * outDim, NULL, &Err);
	CHECK_ERROR(Err);

	cl_mem weight_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * inDim * outDim, weights, &Err);
	CHECK_ERROR(Err);

	cl_mem bias_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * outDim, biases, &Err);
	CHECK_ERROR(Err);

	int max = inDim > outDim ? inDim : outDim;
	size_t global[2] = { (size_t)max, 1 };

	// ================== Kernel Arguments ==================
	clSetKernelArg(FCLayerKernel, 0, sizeof(cl_mem), &input_buffer);
	clSetKernelArg(FCLayerKernel, 1, sizeof(float) * inDim, NULL);
	clSetKernelArg(FCLayerKernel, 2, sizeof(cl_mem), &output_buffer);
	clSetKernelArg(FCLayerKernel, 3, sizeof(cl_mem), &weight_buffer);
	clSetKernelArg(FCLayerKernel, 4, sizeof(cl_mem), &bias_buffer);
	clSetKernelArg(FCLayerKernel, 5, sizeof(int), &inDim);
	clSetKernelArg(FCLayerKernel, 6, sizeof(int), &outDim);

	// ================== 실행하고 결과 받기 ==================
	Err = clEnqueueNDRangeKernel(Queue, FCLayerKernel, 1, NULL, global, NULL, 0, NULL, NULL);
	CHECK_ERROR(Err);

	Err = clEnqueueReadBuffer(Queue, output_buffer, CL_TRUE, 0, sizeof(float) * outDim, outputs, 0, NULL, NULL);
	CHECK_ERROR(Err);

	Err = clFinish(Queue);
	CHECK_ERROR(Err);

	// ================== 메모리 해제 ==================
	clReleaseMemObject(input_buffer);
	clReleaseMemObject(output_buffer);
	clReleaseMemObject(weight_buffer);
	clReleaseMemObject(bias_buffer);
}

void fc_layer(float* input, float* output, float* weights, float* biases, int inDim, int outDim) {
	float sum;
	for (int outNeuron = 0; outNeuron < outDim; ++outNeuron) {
		sum = 0;
		for (int inNeuron = 0; inNeuron < inDim; ++inNeuron) {
			sum += input[inNeuron] * (*weights++);
		}
		sum += biases[outNeuron];
		if (sum > 0) output[outNeuron] = sum;	//ReLU
		else output[outNeuron] = 0;
	}
}

static void softmax_cl(float* input, int N) {
	int i;
	float max = input[0];
	for (i = 1; i < N; i++) {
		if (max < input[i]) max = input[i];
	}
	float sum = 0;
	for (i = 0; i < N; i++) {
		sum += exp(input[i] - max);
	}
	for (i = 0; i < N; i++) {
		input[i] = exp(input[i] - max) / (sum + 1e-7);
	}
}

static int find_max_cl(float* input, int classNum) {
	int i;
	int maxIndex = 0;
	float max = 0;
	for (i = 0; i < classNum; i++) {
		if (max < input[i]) {
			max = input[i];
			maxIndex = i;
		}
	}
	return maxIndex;
}

void cnn(float* images, float* network, int* labels, float* confidences, int num_images) {

	cnn_init();

	float* w[21];
	float* b[21];
	int offset = 0;
	// link weights and biases to network
	for (int i = 0; i < 17; ++i) {
		if (i == 2 || i == 5 || i == 9 || i == 13) i++;	// pooling layer has no weights and biases
		w[i] = network + offset;
		offset += 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i];
		b[i] = network + offset;
		offset += OUTPUT_DIM[i];
	}
	for (int i = 18; i < 21; ++i) {
		w[i] = network + offset;
		offset += INPUT_DIM[i] * OUTPUT_DIM[i];
		b[i] = network + offset;
		offset += OUTPUT_DIM[i];
	}


	// allocate memory for layer
	float* layer[21];
	for (int i = 0; i < 21; ++i) {
		layer[i] = (float*)malloc(sizeof(float) * OUTPUT_DIM[i] * NBYN[i] * NBYN[i]);
		if (layer[i] == NULL) {
			perror("malloc error");
		}
	}

	time_t start, end;
	start = clock();


	// run network
	for (int i = 0; i < num_images; ++i) {
		convolution_cl(images, layer[0], w[0], b[0], INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0]);
		convolution_cl(layer[0], layer[1], w[1], b[1], INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1]);
		max_pooling_cl(layer[1], layer[2], INPUT_DIM[2], NBYN[2] * 2);

		convolution_cl(layer[2], layer[3], w[3], b[3], INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3]);
		convolution_cl(layer[3], layer[4], w[4], b[4], INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4]);
		max_pooling_cl(layer[4], layer[5], INPUT_DIM[5], NBYN[5] * 2);

		convolution_cl(layer[5], layer[6], w[6], b[6], INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6]);
		convolution_cl(layer[6], layer[7], w[7], b[7], INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7]);
		convolution_cl(layer[7], layer[8], w[8], b[8], INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8]);
		max_pooling_cl(layer[8], layer[9], INPUT_DIM[9], NBYN[9] * 2);

		convolution_cl(layer[9], layer[10], w[10], b[10], INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10]);
		convolution_cl(layer[10], layer[11], w[11], b[11], INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11]);
		convolution_cl(layer[11], layer[12], w[12], b[12], INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12]);
		max_pooling_cl(layer[12], layer[13], INPUT_DIM[13], NBYN[13] * 2);

		convolution_cl(layer[13], layer[14], w[14], b[14], INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14]);
		convolution_cl(layer[14], layer[15], w[15], b[15], INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15]);
		convolution_cl(layer[15], layer[16], w[16], b[16], INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16]);
		max_pooling_cl(layer[16], layer[17], INPUT_DIM[17], NBYN[17] * 2);

		fc_layer_cl(layer[17], layer[18], w[18], b[18], INPUT_DIM[18], OUTPUT_DIM[18]);
		fc_layer_cl(layer[18], layer[19], w[19], b[19], INPUT_DIM[19], OUTPUT_DIM[19]);
		fc_layer_cl(layer[19], layer[20], w[20], b[20], INPUT_DIM[20], OUTPUT_DIM[20]);

		softmax_cl(layer[20], 10);

		labels[i] = find_max_cl(layer[20], 10);
		confidences[i] = layer[20][labels[i]];
		images += 32 * 32 * 3;
	}


	end = clock();
	printf("Elapsed time: %.2f sec\n", (double)(end - start) / CLK_TCK);


	for (int i = 0; i < 21; ++i) {
		free(layer[i]);
	}
}
