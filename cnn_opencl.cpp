#pragma warning(disable : 4996)
#include "cnn.h"
#include <CL/cl.h>
#include <ctime>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>

#define CHECK_ERROR(err) \
	if(err != CL_SUCCESS) { \
		printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err);\
		exit(EXIT_FAILURE); \
	}

// CL variables
cl_int Err;
cl_platform_id Platform;
cl_device_id Device;
cl_context Context;
cl_command_queue Queue;
cl_event read_event;
cl_ulong time_start, time_end;

// kernels
cl_kernel ConvolutionKernel;
cl_kernel MaxPoolingKernel;
cl_kernel FCLayer512to512Kernel;
cl_kernel FCLayer512to10Kernel;

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

// Initialize CNN
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


	ConvolutionKernel = clCreateKernel(program, "convolution", &Err);
	CHECK_ERROR(Err);

	FCLayer512to512Kernel = clCreateKernel(program, "fc_layer_optimized_512_512", &Err);
	CHECK_ERROR(Err);

	FCLayer512to10Kernel = clCreateKernel(program, "fc_layer_optimized_512_10", &Err);
	CHECK_ERROR(Err);

	MaxPoolingKernel = clCreateKernel(program, "max_pooling", &Err);
	CHECK_ERROR(Err);
}


// 이미지 크기에 따른 워크 그룹 동적 조정
void adjust_work_group_size(int nbyn, size_t& local_size) {
	if (nbyn < 4) {
		local_size = 1;  // 매우 작은 이미지
	}
	else if (nbyn < 8) {
		local_size = 2;
	}
	else if (nbyn < 16) {
		local_size = 4;
	}
	else if (nbyn < 32) {
		local_size = 8;
	}
	else {
		local_size = 16;
	}
}


// 컨볼루션 병렬화
void convolution_cl(float* inputs, float* outputs, float* filter, float* biases, int inDim, int outDim, int nbyn) {
	size_t global_work_size[2] = { (size_t)nbyn, (size_t)nbyn };

	size_t l_size = 0;
	adjust_work_group_size(nbyn, l_size);

	size_t local_work_size[2] = { l_size, l_size };

	// 버퍼 생성
	cl_mem input_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY, sizeof(float) * inDim * nbyn * nbyn, NULL, &Err);
	CHECK_ERROR(Err);

	cl_mem output_buffer = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, sizeof(float) * nbyn * nbyn, NULL, &Err);
	CHECK_ERROR(Err);

	cl_mem filter_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY, sizeof(float) * 3 * 3 * inDim, NULL, &Err);
	CHECK_ERROR(Err);

	cl_mem bias_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY, sizeof(float), NULL, &Err);
	CHECK_ERROR(Err);

	// 입력 데이터 버퍼에 복사 (반복문 밖에서 한 번만)
	Err = clEnqueueWriteBuffer(Queue, input_buffer, CL_TRUE, 0, sizeof(float) * inDim * nbyn * nbyn, inputs, 0, NULL, NULL);
	CHECK_ERROR(Err);

	for (int cur_out = 0; cur_out < outDim; cur_out++) {
		float* cur_output = outputs + cur_out * nbyn * nbyn;
		float* cur_filter = filter + cur_out * 3 * 3 * inDim;
		float* cur_bias = biases + cur_out; // 바이어스는 출력 채널당 하나의 값

		// 필터와 바이어스 버퍼에 데이터 복사
		Err = clEnqueueWriteBuffer(Queue, filter_buffer, CL_TRUE, 0, sizeof(float) * 3 * 3 * inDim, cur_filter, 0, NULL, NULL);
		CHECK_ERROR(Err);

		Err = clEnqueueWriteBuffer(Queue, bias_buffer, CL_TRUE, 0, sizeof(float), cur_bias, 0, NULL, NULL);
		CHECK_ERROR(Err);

		// 커널 인자 설정
		Err = clSetKernelArg(ConvolutionKernel, 0, sizeof(cl_mem), &input_buffer);
		CHECK_ERROR(Err);
		Err = clSetKernelArg(ConvolutionKernel, 1, sizeof(cl_mem), &output_buffer);
		CHECK_ERROR(Err);
		Err = clSetKernelArg(ConvolutionKernel, 2, sizeof(cl_mem), &filter_buffer);
		CHECK_ERROR(Err);
		Err = clSetKernelArg(ConvolutionKernel, 3, sizeof(cl_mem), &bias_buffer);
		CHECK_ERROR(Err);
		Err = clSetKernelArg(ConvolutionKernel, 4, sizeof(int), &nbyn);
		CHECK_ERROR(Err);
		Err = clSetKernelArg(ConvolutionKernel, 5, sizeof(int), &inDim);
		CHECK_ERROR(Err);

		// 커널 실행
		Err = clEnqueueNDRangeKernel(Queue, ConvolutionKernel, 2, NULL, global_work_size, local_work_size, 0, NULL, &read_event);
		CHECK_ERROR(Err);

		// 프로파일링 정보 수집
		clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
		clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
		//printf("Convolution Layer Elapsed Time = %lu nsec\n", time_end - time_start);

		// 이벤트 해제
		clReleaseEvent(read_event);

		// 결과 읽기
		Err = clEnqueueReadBuffer(Queue, output_buffer, CL_TRUE, 0, sizeof(float) * nbyn * nbyn, cur_output, 0, NULL, NULL);
		CHECK_ERROR(Err);
	}

	// 버퍼 해제
	clReleaseMemObject(input_buffer);
	clReleaseMemObject(output_buffer);
	clReleaseMemObject(filter_buffer);
	clReleaseMemObject(bias_buffer);
}


// Max Pooling 병렬화
void max_pooling_cl(float* input, float* output, int dim, int nbyn) {

	cl_mem input_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * dim * nbyn * nbyn, input, &Err);
	CHECK_ERROR(Err);

	cl_mem output_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(float) * dim * nbyn * nbyn / 4, nullptr, &Err);
	CHECK_ERROR(Err);

	// Set Kernel Args
	Err = clSetKernelArg(MaxPoolingKernel, 0, sizeof(cl_mem), &input_buffer);
	CHECK_ERROR(Err);
	Err = clSetKernelArg(MaxPoolingKernel, 1, sizeof(cl_mem), &output_buffer);
	CHECK_ERROR(Err);
	Err = clSetKernelArg(MaxPoolingKernel, 2, sizeof(cl_int), &nbyn);
	CHECK_ERROR(Err);

	// Set Work Size
	// { dim, nbn * nbyn / 4 }
	// { nbyn / 2, nbyn / 2}
	size_t global_item_size[] = { (size_t)dim, (size_t)(nbyn /2), (size_t)(nbyn /2) };
	size_t local_item_size[] = { 1, (size_t)nbyn / 2, (size_t)nbyn / 2};
	// Run Kernel
	Err = clEnqueueNDRangeKernel(Queue, MaxPoolingKernel, 3, nullptr, global_item_size, local_item_size, 0, nullptr, &read_event);
	CHECK_ERROR(Err);

	Err = clEnqueueReadBuffer(Queue, output_buffer, CL_TRUE, 0, sizeof(float) * dim * nbyn * nbyn / 4, output, 0, nullptr, nullptr);
	CHECK_ERROR(Err);

	Err = clFinish(Queue);
	CHECK_ERROR(Err);

	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, nullptr);
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, nullptr);
	std::cout << "MaxPooling time: " << time_end - time_start << "ns" << std::endl;

	// Release Memory
	clReleaseMemObject(input_buffer);
	clReleaseMemObject(output_buffer);
}


//512입력차원에서 512출력차원으로 가는 완전연결신경망에 최적화된 커널을 호출
void fc_layer_optimized_512_512(float* inputs, float* outputs, float* weights, float* biases, int inDim, int outDim) {
	// ================== 버퍼 생성 ==================
	cl_mem input_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * inDim, inputs, &Err);
	CHECK_ERROR(Err);

	cl_mem output_buffer = clCreateBuffer(Context, CL_MEM_WRITE_ONLY,
		sizeof(float) * outDim, NULL, &Err);
	CHECK_ERROR(Err);

	cl_mem weight_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * inDim * outDim, weights, &Err);
	CHECK_ERROR(Err);

	cl_mem bias_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * outDim, biases, &Err);
	CHECK_ERROR(Err);

	size_t global[2] = { (size_t)outDim, 1 };
	size_t local[2] = { (size_t)outDim / 32, 1 };

	//32개 혹은 64개가 Nvidia GPU에 최적화되어 있는 워크 그룹의 개수이다
	//글로벌 워크 사이즈가 512이고 목표로 하는 워크 그룹의 개수가 32개라면
	//로컬 워크 사이즈는 512 / 32 = 16가 된다

	// ================== 커널 매개변수 전달 ==================
	clSetKernelArg(FCLayer512to512Kernel, 0, sizeof(cl_mem), &input_buffer);
	clSetKernelArg(FCLayer512to512Kernel, 1, sizeof(cl_mem), &output_buffer);
	clSetKernelArg(FCLayer512to512Kernel, 2, sizeof(cl_mem), &weight_buffer);
	clSetKernelArg(FCLayer512to512Kernel, 3, sizeof(cl_mem), &bias_buffer);
	clSetKernelArg(FCLayer512to512Kernel, 4, sizeof(int), &inDim);
	clSetKernelArg(FCLayer512to512Kernel, 5, sizeof(int), &outDim);

	// ================== 실행하고 결과 받기 ==================
	Err = clEnqueueNDRangeKernel(Queue, FCLayer512to512Kernel, 1, NULL, global, local, 0, NULL, &read_event);
	CHECK_ERROR(Err);

	Err = clEnqueueReadBuffer(Queue, output_buffer, CL_TRUE, 0, sizeof(float) * outDim, outputs, 0, NULL, NULL);
	CHECK_ERROR(Err);

	Err = clFinish(Queue);
	CHECK_ERROR(Err);

	// ================== 프로파일링 ==================
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
	printf("FCLayer Elapsed Time = %lu ns\n", time_end - time_start);

	// ================== 메모리 해제 ==================
	clReleaseMemObject(input_buffer);
	clReleaseMemObject(output_buffer);
	clReleaseMemObject(weight_buffer);
	clReleaseMemObject(bias_buffer);
}


//512입력차원에서 10출력차원으로 가는 완전연결신경망에 최적화된 커널을 호출
void fc_layer_optimized_512_10(float* inputs, float* outputs, float* weights, float* biases, int inDim, int outDim) {
	// ================== 버퍼 생성 ==================
	cl_mem input_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * inDim, inputs, &Err);
	CHECK_ERROR(Err);

	cl_mem output_buffer = clCreateBuffer(Context, CL_MEM_WRITE_ONLY,
		sizeof(float) * outDim, NULL, &Err);
	CHECK_ERROR(Err);

	cl_mem weight_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * inDim * outDim, weights, &Err);
	CHECK_ERROR(Err);

	cl_mem bias_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(float) * outDim, biases, &Err);
	CHECK_ERROR(Err);

	size_t global[2] = { (size_t)outDim, 1 };
	size_t local[2] = { (size_t)outDim, 1 };

	// ================== 커널 매개변수 전달 ==================
	clSetKernelArg(FCLayer512to10Kernel, 0, sizeof(cl_mem), &input_buffer);\
	clSetKernelArg(FCLayer512to10Kernel, 1, sizeof(cl_mem), &output_buffer);
	clSetKernelArg(FCLayer512to10Kernel, 2, sizeof(cl_mem), &weight_buffer);
	clSetKernelArg(FCLayer512to10Kernel, 3, sizeof(cl_mem), &bias_buffer);
	clSetKernelArg(FCLayer512to10Kernel, 4, sizeof(int), &inDim);
	clSetKernelArg(FCLayer512to10Kernel, 5, sizeof(int), &outDim);

	// ================== 실행하고 결과 받기 ==================
	Err = clEnqueueNDRangeKernel(Queue, FCLayer512to10Kernel, 1, NULL, global, local, 0, NULL, &read_event);
	CHECK_ERROR(Err);

	Err = clEnqueueReadBuffer(Queue, output_buffer, CL_TRUE, 0, sizeof(float) * outDim, outputs, 0, NULL, NULL);
	CHECK_ERROR(Err);

	Err = clFinish(Queue);
	CHECK_ERROR(Err);

	// ================== 프로파일링 ==================
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, NULL);
	clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &time_end, NULL);
	printf("FCLayer Elapsed Time = %lu ns\n", time_end - time_start);

	// ================== 메모리 해제 ==================
	clReleaseMemObject(input_buffer);
	clReleaseMemObject(output_buffer);
	clReleaseMemObject(weight_buffer);
	clReleaseMemObject(bias_buffer);
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


//////////// CNN 메인 코드 /////////////
void cnn(float* images, float* network, int* labels, float* confidences, int num_images) {
	cnn_init();

	std::cout << "Par allel" << std::endl;

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

		fc_layer_optimized_512_512(layer[17], layer[18], w[18], b[18], INPUT_DIM[18], OUTPUT_DIM[18]);
		fc_layer_optimized_512_512(layer[18], layer[19], w[19], b[19], INPUT_DIM[19], OUTPUT_DIM[19]);
		fc_layer_optimized_512_10(layer[19], layer[20], w[20], b[20], INPUT_DIM[20], OUTPUT_DIM[20]);

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
