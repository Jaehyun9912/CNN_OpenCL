const int MAX_TILE_SIZE = 16;

__kernel void convolution(
	__global const float* input,
	__global float* output,
	__global const float* filter,
	__global const float* bias,
	const int nbyn,
	const int inDim
) {
	int i = get_global_id(0);
	int j = get_global_id(1);

	int padding = 1;
	int nbyn_padded = nbyn + 2 * padding;

	__local float local_input[MAX_TILE_SIZE + 2][MAX_TILE_SIZE + 2];
	__local float local_filter[3][3];

	for (int y = 0; y < 3; y++) {
		for (int x = 0; x < 3; x++) {

		}
	}

	if (i >= nbyn || j >= nbyn) {
		return;
	}

	float sum = 0.0f;
	for (int dim = 0; dim < inDim; dim++) {
		int offset = dim * nbyn * nbyn;
		int offset_filter = dim * 9;

		for (int filter_i = -1; filter_i <= 1; filter_i++) {
			for (int filter_j = -1; filter_j <= 1; filter_j++) {
				int input_i = i + filter_i;
				int input_j = j + filter_j;

				// Boundary check
				if (input_i >= 0 && input_i < nbyn && input_j >= 0 && input_j < nbyn) {
					int filter_index = (filter_i + 1) * 3 + (filter_j + 1) + offset_filter;
					int input_index = nbyn * input_i + input_j + offset;
					sum += input[input_index] * filter[filter_index];
				}
			}
		}
	}

	// Add bias (assuming bias is per output channel)
	sum += bias[0]; // Adjust as necessary for multiple output channels

	// Apply ReLU activation function
	output[nbyn * i + j] = fmax(sum, 0.0f);
}


__kernel void fc_layer_optimized_512_512(
	__global const float* inputs,
	__global float* outputs,
	__global const float* weights,
	__global const float* biases,
	const int inDim,
	const int outDim
) {
	// 로컬 메모리 선언 (워크그룹 내 공유 메모리)
	__local float local_inputs[512];

	//글로벌 인덱스는 0부터 511까지의 값을 가지게 된다
	int output_id = get_global_id(0);

	//로컬 인덱스는 0부터 로컬 사이즈까지의 값을 가지게 된다
	//이 알고리즘에서는 사실상 0부터 15까지의 값을 가지게 된다
	int local_id = get_local_id(0);

	//로컬 사이즈는 최적의 워크 그룹 개수 32개에 따라서 값을 가지게 된다
	//이 알고리즘에서는 사실상 512 / 32 = 16의 크기를 가지게 된다
	int local_size = get_local_size(0);

	//32개의 워크 그룹에는 0부터 15까지의 로컬 인덱스를 가지는 워크 아이템들이 있다
	//로컬 인덱스 00번은 for문을 돌면서 글로벌 메모리 00-16-32-48 ... 496를 로컬 메모리에 복사하고
	//로컬 인덱스 01번은 for문을 돌면서 글로벌 메모리 01-17-33-49 ... 497를 로컬 메모리에 복사하고
	//로컬 인덱스 02번은 for문을 돌면서 글로벌 메모리 02-18-34-50 ... 498를 로컬 메모리에 복사하고
	//...
	//로컬 인덱스 13번은 for문을 돌면서 글로벌 메모리 13-29-45-61 ... 509를 로컬 메모리에 복사하고
	//로컬 인덱스 14번은 for문을 돌면서 글로벌 메모리 14-30-46-62 ... 510를 로컬 메모리에 복사하고
	//로컬 인덱스 15번은 for문을 돌면서 글로벌 메모리 15-31-47-63 ... 511를 로컬 메모리에 복사하고
	for (int i = local_id; i < inDim; i += local_size)
		local_inputs[i] = inputs[i];
	barrier(CLK_LOCAL_MEM_FENCE);

	//이 시점에서 각 워크 그룹의 로컬 메모리에는 글로벌 메모리의 입력뉴런 값이 저장되어 있다
	//글로벌 메모리만 사용했을 경우 512x512=262,144번의 글로벌 메모리 접근이 필요했지만
	//32개의 워크 그룹으로 분할되어 512*32=16,384번의 글로벌 메모리 접근만 필요하게 된다
	//워크 그룹 또한 Nvidia GPU에서 적절한 코어 숫자인 32개로 되어 있어 성능을 최대한 끌어낸다

	//로컬 메모리에 접근하여 완전연결레이어 연산을 수행한다
	float sum = 0.0f;
	for (int i = 0; i < inDim; i++)
		sum += local_inputs[i] * weights[output_id * inDim + i];
	sum += biases[output_id];
	outputs[output_id] = max(sum, 0.0f);
}


__kernel void fc_layer_optimized_512_10(
	__global float* inputs,
	__global float* outputs,
	__global float* weights,
	__global float* biases,
	int inDim,
	int outDim) {
	__local float local_inputs[512];

	//워크 그룹의 크기를 출력뉴런의 차원과 일치시켰다
	//따라서 워크 그룹의 크기는 10이고 개수는 1개가 된다
	int global_id = get_global_id(0);
	int local_id = get_local_id(0);
	int local_size = get_local_size(0);

	//1개의 워크 그룹에는 0부터 9까지의 로컬 인덱스를 가지는 워크 아이템들이 있으며 이게 전부다
	//로컬 인덱스 00번은 for문을 돌면서 글로벌 메모리 00-10-20-30 ... 500 510를 로컬 메모리에 복사하고
	//로컬 인덱스 01번은 for문을 돌면서 글로벌 메모리 01-11-21-31 ... 501 511를 로컬 메모리에 복사하고
	//로컬 인덱스 02번은 for문을 돌면서 글로벌 메모리 02-12-22-32 ... 502 xxx를 로컬 메모리에 복사하고
	//...
	//로컬 인덱스 07번은 for문을 돌면서 글로벌 메모리 07-17-27-37 ... 507 xxx를 로컬 메모리에 복사하고
	//로컬 인덱스 08번은 for문을 돌면서 글로벌 메모리 08-30-46-62 ... 508 xxx를 로컬 메모리에 복사하고
	//로컬 인덱스 09번은 for문을 돌면서 글로벌 메모리 09-31-47-63 ... 509 xxx를 로컬 메모리에 복사하고
	for (int i = global_id; i < inDim; i += local_size)
		local_inputs[i] = inputs[i];
	barrier(CLK_LOCAL_MEM_FENCE);

	float sum = 0.0f;
	for (int i = 0; i < inDim; i++)
		sum += local_inputs[i] * weights[global_id * inDim + i];
	sum += biases[global_id];
	outputs[global_id] = max(sum, 0.0f);
}

const int STRIDE = 2;

__kernel void max_pooling(
	__global float* input,
	__global float* output,
	int nbyn)
{
	//채널
	int dim = get_group_id(0);

	int row = get_local_id(1);
	int col = get_local_id(2);

	float max = -FLT_MAX;
	for (int y = 0; y < 2; y++) {
		for (int x = 0; x < 2; x++) {
			float temp = input[(nbyn * nbyn * dim) + nbyn * (2 * row + y) + 2 * col + x];
			if (max < temp)
				max = temp;
		}
	}

	output[(dim * nbyn / 2 * nbyn / 2) + (row * nbyn / 2) + col] = max;
}
