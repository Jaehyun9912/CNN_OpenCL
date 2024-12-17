#pragma warning(disable : 4996)
#include "cnn.h"
#include <ctime>
#include <cmath>
#include <cstdio>
#include <CL/cl.h>

#include <iostream>
#include <fstream>

#define CHECK_ERROR(err) \
	if(err != CL_SUCCESS) { \
		printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err);\
		exit(EXIT_FAILURE); \
	}

#define BATCH_SIZE 500

cl_int Err;
cl_platform_id Platform;
cl_device_id Device;
cl_context Context;
cl_command_queue Queue;

//프로파일링
cl_event read_event;
cl_ulong time_start, time_end;

//커널
cl_kernel ConvolutionBatchKernel;
cl_kernel MakeFeatureBatchKernel;
cl_kernel ConvolutionBatchKernel2;
cl_kernel ConvolutionBatchKernel3;


cl_kernel ConvolutionBatchKernel4;
cl_kernel MakeFeatureBatchKernel4;



cl_kernel MaxPoolingBatchKernel;
cl_kernel FCLayerBatchKernel;

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
	Queue = clCreateCommandQueue(Context, Device, CL_QUEUE_PROFILING_ENABLE, &Err);
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

	// 최대 작업 그룹 크기 쿼리
	size_t max_work_group_size;
	clGetDeviceInfo(Device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
	printf("최대 작업 그룹 크기: %zu\n", max_work_group_size);

	// 최대 워크 아이템 크기 쿼리
	size_t max_work_item_sizes[3];
	clGetDeviceInfo(Device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), max_work_item_sizes, NULL);
	printf("최대 워크 아이템 크기: %zu x %zu x %zu\n", max_work_item_sizes[0], max_work_item_sizes[1], max_work_item_sizes[2]);

	ConvolutionBatchKernel = clCreateKernel(program, "convolution_batch_optimized", &Err);
	CHECK_ERROR(Err);

	MakeFeatureBatchKernel = clCreateKernel(program, "make_feature_batch_optimized", &Err);
	CHECK_ERROR(Err);

	ConvolutionBatchKernel2 = clCreateKernel(program, "convolution_batch_optimized_2", &Err);
	CHECK_ERROR(Err);

	ConvolutionBatchKernel3 = clCreateKernel(program, "convolution_batch_optimized_3", &Err);
	CHECK_ERROR(Err);



	ConvolutionBatchKernel4 = clCreateKernel(program, "convolution_batch_optimized_4", &Err);
	CHECK_ERROR(Err);
	MakeFeatureBatchKernel4 = clCreateKernel(program, "make_feature_batch_optimized_4", &Err);
	CHECK_ERROR(Err);



	MaxPoolingBatchKernel = clCreateKernel(program, "max_pooling_batch_optimized", &Err);
	CHECK_ERROR(Err);

	FCLayerBatchKernel = clCreateKernel(program, "fc_layer_batch_optimized", &Err);
	CHECK_ERROR(Err);
}

void convolution_batch_optimized(cl_mem inLayer, cl_mem outLayer, float* filter, float* biases, int inDim, int outDim, int nbyn)
{
	int batchSize = BATCH_SIZE;

	// ================== 버퍼 생성 ==================
	cl_mem filter_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inDim * 3 * 3 * outDim, filter, &Err);
	CHECK_ERROR(Err);

	cl_mem bias_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * outDim, biases, &Err);
	CHECK_ERROR(Err);

	cl_mem convolution_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(float) * BATCH_SIZE * inDim * nbyn * nbyn * outDim, NULL, &Err);
	CHECK_ERROR(Err);

	// ================== 커널 매개변수 전달 ==================
	clSetKernelArg(ConvolutionBatchKernel, 0, sizeof(int), &batchSize);
	clSetKernelArg(ConvolutionBatchKernel, 1, sizeof(cl_mem), &inLayer);
	clSetKernelArg(ConvolutionBatchKernel, 2, sizeof(cl_mem), &filter_buffer);
	clSetKernelArg(ConvolutionBatchKernel, 3, sizeof(cl_mem), &convolution_buffer);
	clSetKernelArg(ConvolutionBatchKernel, 4, sizeof(int), &inDim);
	clSetKernelArg(ConvolutionBatchKernel, 5, sizeof(int), &outDim);
	clSetKernelArg(ConvolutionBatchKernel, 6, sizeof(int), &nbyn);

	// ================== 글로벌 워크 아이템 및 로컬 워크 아이템 ==================
	size_t global_work_size_1[] = { (size_t)outDim, (size_t)(nbyn * nbyn), (size_t)inDim };
	size_t local_work_size_1[] = { 1, (size_t)(nbyn * nbyn), 1 };

	// ================== 실행 ==================	
	Err = clEnqueueNDRangeKernel(Queue, ConvolutionBatchKernel, 3, NULL, global_work_size_1, local_work_size_1, 0, NULL, &read_event);
	CHECK_ERROR(Err);

	Err = clFinish(Queue);
	CHECK_ERROR(Err);

	// ================== 프로파일링 ==================
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);

	// ================== 메모리 해제 ==================
	clReleaseMemObject(filter_buffer);

	// ================== 커널 매개변수 전달 ==================
	clSetKernelArg(MakeFeatureBatchKernel, 0, sizeof(int), &batchSize);
	clSetKernelArg(MakeFeatureBatchKernel, 1, sizeof(cl_mem), &convolution_buffer);
	clSetKernelArg(MakeFeatureBatchKernel, 2, sizeof(cl_mem), &bias_buffer);
	clSetKernelArg(MakeFeatureBatchKernel, 3, sizeof(cl_mem), &outLayer);
	clSetKernelArg(MakeFeatureBatchKernel, 4, sizeof(int), &inDim);
	clSetKernelArg(MakeFeatureBatchKernel, 5, sizeof(int), &outDim);
	clSetKernelArg(MakeFeatureBatchKernel, 6, sizeof(int), &nbyn);

	// ================== 글로벌 워크 아이템 및 로컬 워크 아이템 ==================
	size_t global_work_size_2[] = { (size_t)outDim, (size_t)(nbyn * nbyn) };
	size_t local_work_size_2[] = { 1, (size_t)(nbyn * nbyn) };

	// ================== 실행 ==================
	Err = clEnqueueNDRangeKernel(Queue, MakeFeatureBatchKernel, 2, NULL, global_work_size_2, local_work_size_2, 0, NULL, &read_event);
	CHECK_ERROR(Err);

	Err = clFinish(Queue);
	CHECK_ERROR(Err);

	// ================== 프로파일링 ==================
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
	printf("Convolution Layer Elapsed Time = %lu nsec\n", time_end - time_start);

	// ================== 메모리 해제 ==================
	clReleaseMemObject(convolution_buffer);
	clReleaseMemObject(bias_buffer);
}
void convolution_batch_optimized_2(cl_mem inLayer, cl_mem outLayer, float* filter, float* biases, int inDim, int outDim, int nbyn)
{
	int batchSize = BATCH_SIZE;

	// ================== 버퍼 생성 ==================
	cl_mem filter_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inDim * 3 * 3 * outDim, filter, &Err);
	CHECK_ERROR(Err);

	cl_mem bias_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * outDim, biases, &Err);
	CHECK_ERROR(Err);

	// ================== 커널 매개변수 전달 ==================
	clSetKernelArg(ConvolutionBatchKernel2, 0, sizeof(int), &batchSize);
	clSetKernelArg(ConvolutionBatchKernel2, 1, sizeof(cl_mem), &inLayer);
	clSetKernelArg(ConvolutionBatchKernel2, 2, sizeof(cl_mem), &outLayer);
	clSetKernelArg(ConvolutionBatchKernel2, 3, sizeof(cl_mem), &filter_buffer);
	clSetKernelArg(ConvolutionBatchKernel2, 4, sizeof(cl_mem), &bias_buffer);
	clSetKernelArg(ConvolutionBatchKernel2, 5, sizeof(int), &inDim);
	clSetKernelArg(ConvolutionBatchKernel2, 6, sizeof(int), &outDim);
	clSetKernelArg(ConvolutionBatchKernel2, 7, sizeof(int), &nbyn);

	// ================== 글로벌 워크 아이템 및 로컬 워크 아이템 ==================
	size_t global_work_size[] = { (size_t)(nbyn * nbyn * batchSize), (size_t)inDim, (size_t)outDim };
	size_t local_work_size[] = { 1, (size_t)inDim, 1 };

	// ================== 실행 ==================	
	Err = clEnqueueNDRangeKernel(Queue, ConvolutionBatchKernel2, 3, NULL, global_work_size, local_work_size, 0, NULL, &read_event);
	CHECK_ERROR(Err);

	Err = clFinish(Queue);
	CHECK_ERROR(Err);

	// ================== 프로파일링 ==================
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
	printf("Convolution Layer Task 2 Elapsed Time = %lu nsec\n", time_end - time_start);

	// ================== 메모리 해제 ==================
	clReleaseMemObject(filter_buffer);
	clReleaseMemObject(bias_buffer);
}
void convolution_batch_optimized_3(cl_mem inLayer, cl_mem outLayer, float* filter, float* biases, int inDim, int outDim, int nbyn)
{
	int batchSize = BATCH_SIZE;

	// ================== 버퍼 생성 ==================
	cl_mem filter_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inDim * 3 * 3 * outDim, filter, &Err);
	CHECK_ERROR(Err);

	cl_mem bias_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * outDim, biases, &Err);
	CHECK_ERROR(Err);

	// ================== 커널 매개변수 전달 ==================
	clSetKernelArg(ConvolutionBatchKernel3, 0, sizeof(int), &batchSize);
	clSetKernelArg(ConvolutionBatchKernel3, 1, sizeof(cl_mem), &inLayer);
	clSetKernelArg(ConvolutionBatchKernel3, 2, sizeof(cl_mem), &outLayer);
	clSetKernelArg(ConvolutionBatchKernel3, 3, sizeof(cl_mem), &filter_buffer);
	clSetKernelArg(ConvolutionBatchKernel3, 4, sizeof(cl_mem), &bias_buffer);
	clSetKernelArg(ConvolutionBatchKernel3, 5, sizeof(int), &inDim);
	clSetKernelArg(ConvolutionBatchKernel3, 6, sizeof(int), &outDim);
	clSetKernelArg(ConvolutionBatchKernel3, 7, sizeof(int), &nbyn);

	// ================== 글로벌 워크 아이템 및 로컬 워크 아이템 ==================
	size_t global_work_size[] = { (size_t)(nbyn * nbyn), (size_t)inDim, (size_t)(outDim * batchSize) };
	size_t local_work_size[] = { 2 * 2, (size_t)inDim, 1 };

	// ================== 실행 ==================	
	Err = clEnqueueNDRangeKernel(Queue, ConvolutionBatchKernel3, 3, NULL, global_work_size, local_work_size, 0, NULL, &read_event);
	CHECK_ERROR(Err);

	Err = clFinish(Queue);
	CHECK_ERROR(Err);

	// ================== 프로파일링 ==================
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
	printf("Convolution Layer V3 Elapsed Time = %lu nsec\n", time_end - time_start);

	// ================== 메모리 해제 ==================
	clReleaseMemObject(filter_buffer);
	clReleaseMemObject(bias_buffer);
}
void convolution_batch_optimized_4(cl_mem inLayer, cl_mem outLayer, float* filter, float* biases, int inDim, int outDim, int nbyn)
{
	int batchSize = BATCH_SIZE;

	// ================== 버퍼 생성 ==================
	cl_mem filter_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * inDim * 3 * 3 * outDim, filter, &Err);
	CHECK_ERROR(Err);

	cl_mem bias_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * outDim, biases, &Err);
	CHECK_ERROR(Err);

	cl_mem reduction_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(float) * nbyn * nbyn * 8 * outDim * batchSize, NULL, &Err);
	CHECK_ERROR(Err);

	// ================== 커널 매개변수 전달 ==================
	clSetKernelArg(ConvolutionBatchKernel4, 0, sizeof(int), &batchSize);
	clSetKernelArg(ConvolutionBatchKernel4, 1, sizeof(cl_mem), &inLayer);
	clSetKernelArg(ConvolutionBatchKernel4, 2, sizeof(cl_mem), &filter_buffer);
	clSetKernelArg(ConvolutionBatchKernel4, 3, sizeof(cl_mem), &reduction_buffer);
	clSetKernelArg(ConvolutionBatchKernel4, 4, sizeof(int), &inDim);
	clSetKernelArg(ConvolutionBatchKernel4, 5, sizeof(int), &outDim);
	clSetKernelArg(ConvolutionBatchKernel4, 6, sizeof(int), &nbyn);

	// ================== 글로벌 워크 아이템 및 로컬 워크 아이템 ==================
	size_t global_work_size_1[] = { (size_t)(nbyn * nbyn), (size_t)inDim, (size_t)outDim * batchSize };
	size_t local_work_size_1[] = { 4 * 4, (size_t)(inDim / 8), 1 };

	// ================== 실행 ==================	
	Err = clEnqueueNDRangeKernel(Queue, ConvolutionBatchKernel4, 3, NULL, global_work_size_1, local_work_size_1, 0, NULL, &read_event);
	CHECK_ERROR(Err);

	Err = clFinish(Queue);
	CHECK_ERROR(Err);

	// ================== 프로파일링 ==================
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);

	// ================== 메모리 해제 ==================
	clReleaseMemObject(filter_buffer);

	// ================== 커널 매개변수 전달 ==================
	clSetKernelArg(MakeFeatureBatchKernel4, 0, sizeof(int), &batchSize);
	clSetKernelArg(MakeFeatureBatchKernel4, 1, sizeof(cl_mem), &reduction_buffer);
	clSetKernelArg(MakeFeatureBatchKernel4, 2, sizeof(cl_mem), &bias_buffer);
	clSetKernelArg(MakeFeatureBatchKernel4, 3, sizeof(cl_mem), &outLayer);
	clSetKernelArg(MakeFeatureBatchKernel4, 4, sizeof(int), &inDim);
	clSetKernelArg(MakeFeatureBatchKernel4, 5, sizeof(int), &outDim);
	clSetKernelArg(MakeFeatureBatchKernel4, 6, sizeof(int), &nbyn);

	// ================== 글로벌 워크 아이템 및 로컬 워크 아이템 ==================
	size_t global_work_size_2[] = { (size_t)(nbyn * nbyn), (size_t)(outDim * batchSize) };

	// ================== 실행 ==================	
	Err = clEnqueueNDRangeKernel(Queue, MakeFeatureBatchKernel4, 2, NULL, global_work_size_2, NULL, 0, NULL, &read_event);
	CHECK_ERROR(Err);

	Err = clFinish(Queue);
	CHECK_ERROR(Err);

	// ================== 프로파일링 ==================
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
	printf("Convolution Layer V4 Elapsed Time = %lu nsec\n", time_end - time_start);

	// ================== 메모리 해제 ==================
	clReleaseMemObject(bias_buffer);
}



void max_pooling_batch_optimized(cl_mem inLayer, cl_mem outLayer, int dim, int nbyn)
{
	//첫번째 최대 풀링 레이어라고 가정하면...

	// ================== 글로벌 워크 아이템 및 로컬 워크 아이템 ==================
	size_t global_work_size[] = { (size_t)dim, (size_t)(nbyn / 2), (size_t)(nbyn / 2) };

	size_t local_work_size[] = { 1, (size_t)nbyn / 2, (size_t)nbyn / 2 };

	// ================== 커널 매개변수 전달 ==================
	int batchSize = BATCH_SIZE;
	Err = clSetKernelArg(MaxPoolingBatchKernel, 0, sizeof(cl_int), &batchSize);
	CHECK_ERROR(Err);

	Err = clSetKernelArg(MaxPoolingBatchKernel, 1, sizeof(cl_mem), &inLayer);
	CHECK_ERROR(Err);

	Err = clSetKernelArg(MaxPoolingBatchKernel, 2, sizeof(cl_mem), &outLayer);
	CHECK_ERROR(Err);

	Err = clSetKernelArg(MaxPoolingBatchKernel, 3, sizeof(cl_int), &dim);
	CHECK_ERROR(Err);

	Err = clSetKernelArg(MaxPoolingBatchKernel, 4, sizeof(cl_int), &nbyn);
	CHECK_ERROR(Err);

	// ================== 실행하고 결과 받기 ==================
	Err = clEnqueueNDRangeKernel(Queue, MaxPoolingBatchKernel, 3, nullptr, global_work_size, local_work_size, 0, nullptr, &read_event);
	CHECK_ERROR(Err);

	Err = clFinish(Queue);
	CHECK_ERROR(Err);

	// ================== 프로파일링 ==================
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, nullptr);
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, nullptr);
	printf("MaxPooling Layer Elapsed Time = %lu nsec\n", time_end - time_start);
}
void fc_layer_batch_optimized(cl_mem inLayer, cl_mem outLayer, float* weights, float* biases, int inDim, int outDim)
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
	clSetKernelArg(FCLayerBatchKernel, 0, sizeof(int), &batchSize);
	clSetKernelArg(FCLayerBatchKernel, 1, sizeof(cl_mem), &inLayer);
	clSetKernelArg(FCLayerBatchKernel, 2, sizeof(cl_mem), &outLayer);
	clSetKernelArg(FCLayerBatchKernel, 3, sizeof(cl_mem), &weight_buffer);
	clSetKernelArg(FCLayerBatchKernel, 4, sizeof(cl_mem), &bias_buffer);
	clSetKernelArg(FCLayerBatchKernel, 5, sizeof(int), &inDim);
	clSetKernelArg(FCLayerBatchKernel, 6, sizeof(int), &outDim);

	// ================== 실행하고 결과 받기 ==================
	Err = clEnqueueNDRangeKernel(Queue, FCLayerBatchKernel, 2, NULL, global_work_size, NULL, 0, NULL, &read_event);
	CHECK_ERROR(Err);

	Err = clFinish(Queue);
	CHECK_ERROR(Err);

	// ================== 프로파일링 ==================
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
	printf("FCLayer Elapsed Time = %lu nsec\n", time_end - time_start);

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

cl_mem createLayerBuffer(int i)
{
	cl_mem buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(float) * OUTPUT_DIM[i] * NBYN[i] * NBYN[i] * BATCH_SIZE, NULL, &Err);
	CHECK_ERROR(Err);
	return buffer;
}

void cnn(float* images, float* network, int* labels, float* confidences, int num_images)
{
	cnn_init();

	//배치 사이즈만큼의 이미지를 디바이스에 복사한다
	cl_mem images_buffer;
	images_buffer = clCreateBuffer(Context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * 3 * 32 * 32 * BATCH_SIZE, images, &Err);
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

	//시간 측정
	time_t start, end;
	start = clock();

#pragma region 1단계
	layers_buffer[0] = createLayerBuffer(0);
	convolution_batch_optimized(images_buffer, layers_buffer[0], w[0], b[0], INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0]);
	clReleaseMemObject(images_buffer);

	layers_buffer[1] = createLayerBuffer(1);
	convolution_batch_optimized_4(layers_buffer[0], layers_buffer[1], w[1], b[1], INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1]);
	clReleaseMemObject(layers_buffer[0]);

	layers_buffer[2] = createLayerBuffer(2);
	max_pooling_batch_optimized(layers_buffer[1], layers_buffer[2], INPUT_DIM[2], NBYN[2] * 2);
	clReleaseMemObject(layers_buffer[1]);
#pragma endregion

#pragma region 2단계
	layers_buffer[3] = createLayerBuffer(3);
	convolution_batch_optimized_4(layers_buffer[2], layers_buffer[3], w[3], b[3], INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3]);
	clReleaseMemObject(layers_buffer[2]);

	layers_buffer[4] = createLayerBuffer(4);
	convolution_batch_optimized_4(layers_buffer[3], layers_buffer[4], w[4], b[4], INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4]);
	clReleaseMemObject(layers_buffer[3]);

	layers_buffer[5] = createLayerBuffer(5);
	max_pooling_batch_optimized(layers_buffer[4], layers_buffer[5], INPUT_DIM[5], NBYN[5] * 2);
	clReleaseMemObject(layers_buffer[4]);
#pragma endregion

#pragma region 3단계
	layers_buffer[6] = createLayerBuffer(6);
	convolution_batch_optimized_4(layers_buffer[5], layers_buffer[6], w[6], b[6], INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6]);
	clReleaseMemObject(layers_buffer[5]);

	layers_buffer[7] = createLayerBuffer(7);
	convolution_batch_optimized_4(layers_buffer[6], layers_buffer[7], w[7], b[7], INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7]);
	clReleaseMemObject(layers_buffer[6]);

	layers_buffer[8] = createLayerBuffer(8);
	convolution_batch_optimized_4(layers_buffer[7], layers_buffer[8], w[8], b[8], INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8]);
	clReleaseMemObject(layers_buffer[7]);

	layers_buffer[9] = createLayerBuffer(9);
	max_pooling_batch_optimized(layers_buffer[8], layers_buffer[9], INPUT_DIM[9], NBYN[9] * 2);
	clReleaseMemObject(layers_buffer[8]);
#pragma endregion

#pragma region 4단계
	layers_buffer[10] = createLayerBuffer(10);
	convolution_batch_optimized_4(layers_buffer[9], layers_buffer[10], w[10], b[10], INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10]);
	clReleaseMemObject(layers_buffer[9]);

	layers_buffer[11] = createLayerBuffer(11);
	convolution_batch_optimized_4(layers_buffer[10], layers_buffer[11], w[11], b[11], INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11]);
	clReleaseMemObject(layers_buffer[10]);

	layers_buffer[12] = createLayerBuffer(12);
	convolution_batch_optimized_4(layers_buffer[11], layers_buffer[12], w[12], b[12], INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12]);
	clReleaseMemObject(layers_buffer[11]);

	layers_buffer[13] = createLayerBuffer(13);
	max_pooling_batch_optimized(layers_buffer[12], layers_buffer[13], INPUT_DIM[13], NBYN[13] * 2);
	clReleaseMemObject(layers_buffer[12]);
#pragma endregion

#pragma region 5단계
	layers_buffer[14] = createLayerBuffer(14);
	convolution_batch_optimized_2(layers_buffer[13], layers_buffer[14], w[14], b[14], INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14]);
	clReleaseMemObject(layers_buffer[13]);

	layers_buffer[15] = createLayerBuffer(15);
	convolution_batch_optimized_2(layers_buffer[14], layers_buffer[15], w[15], b[15], INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15]);
	clReleaseMemObject(layers_buffer[14]);

	layers_buffer[16] = createLayerBuffer(16);
	convolution_batch_optimized_2(layers_buffer[15], layers_buffer[16], w[16], b[16], INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16]);
	clReleaseMemObject(layers_buffer[15]);

	layers_buffer[17] = createLayerBuffer(17);
	max_pooling_batch_optimized(layers_buffer[16], layers_buffer[17], INPUT_DIM[17], NBYN[17] * 2);
	clReleaseMemObject(layers_buffer[16]);
#pragma endregion

#pragma region 6단계
	layers_buffer[18] = createLayerBuffer(18);
	fc_layer_batch_optimized(layers_buffer[17], layers_buffer[18], w[18], b[18], INPUT_DIM[18], OUTPUT_DIM[18]);
	clReleaseMemObject(layers_buffer[17]);

	layers_buffer[19] = createLayerBuffer(19);
	fc_layer_batch_optimized(layers_buffer[18], layers_buffer[19], w[19], b[19], INPUT_DIM[19], OUTPUT_DIM[19]);
	clReleaseMemObject(layers_buffer[18]);

	layers_buffer[20] = createLayerBuffer(20);
	fc_layer_batch_optimized(layers_buffer[19], layers_buffer[20], w[20], b[20], INPUT_DIM[20], OUTPUT_DIM[20]);
	clReleaseMemObject(layers_buffer[19]);
#pragma endregion

#pragma region 결과 읽기
	float* softmax_layer = (float*)malloc(sizeof(float) * 10 * BATCH_SIZE);
	Err = clEnqueueReadBuffer(Queue, layers_buffer[20], CL_TRUE, 0, sizeof(float) * 10 * BATCH_SIZE, softmax_layer, 0, NULL, NULL);
	CHECK_ERROR(Err);
	clReleaseMemObject(layers_buffer[20]);
#pragma endregion

#pragma region 소프트맥스
	for (int batch = 0; batch < BATCH_SIZE; batch++)
	{
		float* current = softmax_layer + 10 * batch;
		softmax(current, 10);
		labels[batch] = find_max(current, 10);
		confidences[batch] = current[labels[batch]];
	}
#pragma endregion

	//시간 측정
	end = clock();
	printf("Elapsed time: %.2f sec\n", (double)(end - start) / CLK_TCK);

	//할당 해제
	free(softmax_layer);
}