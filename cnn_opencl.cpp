#pragma warning(disable : 4996)
#include "cnn.h"
#include <ctime>
#include <cstdio>
#include <cmath>
#include <CL/cl.h>

#include <iostream>
#include <fstream>

#define CHECK_ERROR(err) \
	if(err != CL_SUCCESS) { \
		printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err);\
		exit(EXIT_FAILURE); \
	}

#define BATCH_SIZE (60)

#define DEBUG (0)

cl_int Err;
cl_platform_id Platform;
cl_device_id Device;
cl_context Context;
cl_command_queue Queue;

//프로파일링
cl_event read_event;
cl_ulong time_start, time_end;

//커널
cl_kernel ConvolutionKernel;
cl_kernel MaxPoolingKernel;
cl_kernel FCLayerKernel;


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
	cl_command_queue_properties properties[] = { CL_QUEUE_PROFILING_ENABLE, 0 };
	Queue = clCreateCommandQueueWithProperties(Context, Device, properties, &Err);
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

	MaxPoolingKernel = clCreateKernel(program, "max_pooling", &Err);
	CHECK_ERROR(Err);

	FCLayerKernel = clCreateKernel(program, "fc_layer", &Err);
	CHECK_ERROR(Err);


	size_t max_work_group_size;
	clGetDeviceInfo(Device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
	if (DEBUG==2) printf("Max work group size: %zu\n", max_work_group_size);

	size_t max_work_item_sizes[3];
	clGetDeviceInfo(Device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), max_work_item_sizes, NULL);
	if (DEBUG==2) printf("Max work item sizes: [%zu, %zu, %zu]\n",
		   max_work_item_sizes[0], max_work_item_sizes[1], max_work_item_sizes[2]);

}

void convolution(cl_mem inLayer, cl_mem outLayer, float* filter, float* biases, int inDim, int outDim, int nbyn)
{
	int batchSize = BATCH_SIZE;

	// ================== 버퍼 생성 ==================
	cl_mem filter_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 3 * 3 * inDim * outDim, filter, &Err);
	CHECK_ERROR(Err);

	cl_mem bias_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * outDim, biases, &Err);
	CHECK_ERROR(Err);

	cl_mem convolution_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(float) * nbyn * nbyn * inDim * outDim * batchSize, NULL, &Err);
	CHECK_ERROR(Err);

	// ================== 커널 매개변수 전달 ==================
	clSetKernelArg(ConvolutionKernel, 0, sizeof(cl_mem), &inLayer);
	clSetKernelArg(ConvolutionKernel, 1, sizeof(cl_mem), &outLayer);
	clSetKernelArg(ConvolutionKernel, 2, sizeof(cl_mem), &filter_buffer);
	clSetKernelArg(ConvolutionKernel, 3, sizeof(cl_mem), &bias_buffer);
	clSetKernelArg(ConvolutionKernel, 4, sizeof(cl_mem), &convolution_buffer);
	clSetKernelArg(ConvolutionKernel, 5, sizeof(cl_int), &inDim);
	clSetKernelArg(ConvolutionKernel, 6, sizeof(cl_int), &outDim);
	clSetKernelArg(ConvolutionKernel, 7, sizeof(cl_int), &nbyn);

	// ================== 글로벌 워크 아이템 및 로컬 워크 아이템 ==================
	size_t global_work_size[] = { (size_t)(outDim * inDim), (size_t)(nbyn*nbyn), (size_t)batchSize };
	size_t local_work_size[] = { (size_t)inDim, 1, 1 };
	// group_core_size[] = { outDim, index, batchSize }

	// ================== 실행 ==================
	if (DEBUG == 2) printf("Start Convolution\n");
	Err = clEnqueueNDRangeKernel(Queue, ConvolutionKernel, 3, NULL, global_work_size, local_work_size, 0, NULL, &read_event);
	CHECK_ERROR(Err);

	Err = clFinish(Queue);
	CHECK_ERROR(Err);
	if (DEBUG == 2) printf("End Convolution\n");

	// ================== 프로파일링 ==================
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);

	// ================== 메모리 해제 ==================
	clReleaseMemObject(filter_buffer);
	clReleaseMemObject(bias_buffer);
	clReleaseMemObject(convolution_buffer);

	// ================== 프로파일링 ==================
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
	if (DEBUG) printf("Convolution Layer Elapsed Time = %lu nsec\n", time_end - time_start);
}


void max_pooling(cl_mem inLayer, cl_mem outLayer, int dim, int nbyn)
{
	// ================== 글로벌 워크 아이템 및 로컬 워크 아이템 ==================
	size_t global_work_size[] = { (size_t)dim, (size_t)(nbyn / 2), (size_t)(nbyn / 2) };

	size_t local_work_size[] = { 1, (size_t)nbyn / 2, (size_t)nbyn / 2 };

	// ================== 커널 매개변수 전달 ==================
	int batchSize = BATCH_SIZE;
	Err = clSetKernelArg(MaxPoolingKernel, 0, sizeof(cl_int), &batchSize);
	CHECK_ERROR(Err);

	Err = clSetKernelArg(MaxPoolingKernel, 1, sizeof(cl_mem), &inLayer);
	CHECK_ERROR(Err);

	Err = clSetKernelArg(MaxPoolingKernel, 2, sizeof(cl_mem), &outLayer);
	CHECK_ERROR(Err);

	Err = clSetKernelArg(MaxPoolingKernel, 3, sizeof(cl_int), &dim);
	CHECK_ERROR(Err);

	Err = clSetKernelArg(MaxPoolingKernel, 4, sizeof(cl_int), &nbyn);
	CHECK_ERROR(Err);

	// ================== 실행하고 결과 받기 ==================
	Err = clEnqueueNDRangeKernel(Queue, MaxPoolingKernel, 3, nullptr, global_work_size, local_work_size, 0, nullptr, &read_event);
	CHECK_ERROR(Err);

	Err = clFinish(Queue);
	CHECK_ERROR(Err);

	// ================== 프로파일링 ==================
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, nullptr);
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, nullptr);
	if (DEBUG) printf("MaxPooling Layer Elapsed Time = %lu nsec\n", time_end - time_start);
}


void fc_layer(cl_mem inLayer, cl_mem outLayer, float* weights, float* biases, int inDim, int outDim)
{
	// ================== 버퍼 생성 ==================
	cl_mem weight_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * inDim * outDim, weights, &Err);
	CHECK_ERROR(Err);

	cl_mem bias_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * outDim, biases, &Err);
	CHECK_ERROR(Err);

	size_t global_work_size[2] = { (size_t)outDim, 1 };

	// ================== 커널 매개변수 전달 ==================
	int batchSize = BATCH_SIZE;
	clSetKernelArg(FCLayerKernel, 0, sizeof(int), &batchSize);
	clSetKernelArg(FCLayerKernel, 1, sizeof(cl_mem), &inLayer);
	clSetKernelArg(FCLayerKernel, 2, sizeof(cl_mem), &outLayer);
	clSetKernelArg(FCLayerKernel, 3, sizeof(cl_mem), &weight_buffer);
	clSetKernelArg(FCLayerKernel, 4, sizeof(cl_mem), &bias_buffer);
	clSetKernelArg(FCLayerKernel, 5, sizeof(int), &inDim);
	clSetKernelArg(FCLayerKernel, 6, sizeof(int), &outDim);

	// ================== 실행하고 결과 받기 ==================
	Err = clEnqueueNDRangeKernel(Queue, FCLayerKernel, 2, NULL, global_work_size, NULL, 0, NULL, &read_event);
	CHECK_ERROR(Err);

	Err = clFinish(Queue);
	CHECK_ERROR(Err);

	// ================== 프로파일링 ==================
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
	if (DEBUG) printf("FCLayer Elapsed Time = %lu nsec\n", time_end - time_start);

	// ================== 메모리 해제 ==================
	clReleaseMemObject(weight_buffer);
	clReleaseMemObject(bias_buffer);
}


static void softmax(float* input, int N) {
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


static int find_max(float* input, int classNum) {
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


void cnn(float* images, float* network, int* labels, float* confidences, int num_of_image)
{
	cnn_init();

	//배치 사이즈만큼의 이미지를 디바이스에 복사한다
	const size_t IMG_SIZE = 3 * 32 * 32;
	cl_mem images_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(float) * IMG_SIZE * BATCH_SIZE, NULL, &Err);
	CHECK_ERROR(Err);

	//신경망에 가중치와 편차 연결
	float* w[21];
	float* b[21];
	int offset = 0;
	for (int i = 0; i < 17; ++i)
	{
		//최대 풀링 레이어는 가중치와 편차 없음
		if (i == 2 || i == 5 || i == 9 || i == 13)
			i++;
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

	//각 레이어에서 사용할 디바이스 버퍼
	cl_mem layers_buffer[21];
	for (int i = 0; i < 21; i++)
	{
		layers_buffer[i] = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(float) * OUTPUT_DIM[i] * NBYN[i] * NBYN[i] * BATCH_SIZE, NULL, &Err);
		CHECK_ERROR(Err);
	}

	//시간 측정
	printf("Parallel : \n");
	time_t start, end;
	start = clock();

	// run network
	for (int i = 0; i < num_of_image / BATCH_SIZE; ++i) {
		clEnqueueWriteBuffer(Queue, images_buffer, CL_TRUE, 0, sizeof(float) * IMG_SIZE * BATCH_SIZE, (images + IMG_SIZE * BATCH_SIZE * i), 0, NULL, NULL);

		convolution(images_buffer, layers_buffer[0], w[0], b[0], INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0]);
		convolution(layers_buffer[0], layers_buffer[1], w[1], b[1], INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1]);
		max_pooling(layers_buffer[1], layers_buffer[2], INPUT_DIM[2], NBYN[2] * 2);

		convolution(layers_buffer[2], layers_buffer[3], w[3], b[3], INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3]);
		convolution(layers_buffer[3], layers_buffer[4], w[4], b[4], INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4]);
		max_pooling(layers_buffer[4], layers_buffer[5], INPUT_DIM[5], NBYN[5] * 2);

		convolution(layers_buffer[5], layers_buffer[6], w[6], b[6], INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6]);
		convolution(layers_buffer[6], layers_buffer[7], w[7], b[7], INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7]);
		convolution(layers_buffer[7], layers_buffer[8], w[8], b[8], INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8]);
		max_pooling(layers_buffer[8], layers_buffer[9], INPUT_DIM[9], NBYN[9] * 2);

		convolution(layers_buffer[9], layers_buffer[10], w[10], b[10], INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10]);
		convolution(layers_buffer[10], layers_buffer[11], w[11], b[11], INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11]);
		convolution(layers_buffer[11], layers_buffer[12], w[12], b[12], INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12]);
		max_pooling(layers_buffer[12], layers_buffer[13], INPUT_DIM[13], NBYN[13] * 2);

		convolution(layers_buffer[13], layers_buffer[14], w[14], b[14], INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14]);
		convolution(layers_buffer[14], layers_buffer[15], w[15], b[15], INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15]);
		convolution(layers_buffer[15], layers_buffer[16], w[16], b[16], INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16]);
		max_pooling(layers_buffer[16], layers_buffer[17], INPUT_DIM[17], NBYN[17] * 2);

		fc_layer(layers_buffer[17], layers_buffer[18], w[18], b[18], INPUT_DIM[18], OUTPUT_DIM[18]);
		fc_layer(layers_buffer[18], layers_buffer[19], w[19], b[19], INPUT_DIM[19], OUTPUT_DIM[19]);
		fc_layer(layers_buffer[19], layers_buffer[20], w[20], b[20], INPUT_DIM[20], OUTPUT_DIM[20]);


		// 연산 완료
		float* softmax_layer = (float*)malloc(sizeof(float) * 10 * BATCH_SIZE);
		Err = clEnqueueReadBuffer(Queue, layers_buffer[20], CL_TRUE, 0, sizeof(float) * 10 * BATCH_SIZE, softmax_layer, 0, NULL, NULL);
		CHECK_ERROR(Err);

		clFinish(Queue);
		CHECK_ERROR(Err);

		for (int batch = 0; batch < BATCH_SIZE; batch++)
		{
			float* current = softmax_layer + 10 * batch;
			softmax(current, 10);
			labels[BATCH_SIZE * i + batch] = find_max(current, 10);
			confidences[BATCH_SIZE * i + batch] = current[labels[BATCH_SIZE * i + batch]];
		}

		free(softmax_layer);
	}
	//시간 측정
	end = clock();
	printf("Pararrel Elapsed time: %.2f sec\n", (double)(end - start) / CLK_TCK);


	// 메모리 정리
	clReleaseMemObject(images_buffer);
	for (int i = 0; i < 21; i++)
		clReleaseMemObject(layers_buffer[i]);

}
