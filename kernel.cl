__kernel void convolution(
	__global const float* inputs,    // 입력 데이터
	__global float* outputs,         // 출력 데이터
	__global const float* filters,   // 필터(가중치)
	__global const float* biases,    // 바이어스
	int inDim,                       // 입력 채널 개수
	int outDim,                      // 출력 채널 개수
	int nbyn                         // 입력/출력 텐서의 가로/세로 크기
) {
	// 워크 아이템 ID
	int out_x = get_global_id(0); // x 좌표 (너비 방향)
	int out_y = get_global_id(1); // y 좌표 (높이 방향)

	// 워크 아이템이 출력 텐서 내 유효한 위치에 있는지 확인
	if (out_x >= nbyn || out_y >= nbyn) {
		return;
	}

	// 출력 채널 루프
	for (int out_channel = 0; out_channel < outDim; out_channel++) {
		// 바이어스를 초기 값으로 설정
		float sum = biases[out_channel];

		// 입력 채널 루프
		for (int in_channel = 0; in_channel < inDim; in_channel++) {
			// 필터와 입력 데이터의 시작 위치 계산
			int filter_offset = ((out_channel * inDim) + in_channel) * 3 * 3;
			int input_offset = (in_channel * nbyn * nbyn);

			// 3x3 필터 적용
			for (int filter_y = 0; filter_y < 3; filter_y++) {
				for (int filter_x = 0; filter_x < 3; filter_x++) {
					// 입력 데이터의 위치 계산 (경계 처리 포함)
					int input_x = out_x + filter_x - 1; // 필터 중심 기준으로 위치 조정
					int input_y = out_y + filter_y - 1;

					// 경계를 벗어난 경우 0으로 처리
					if (input_x >= 0 && input_x < nbyn && input_y >= 0 && input_y < nbyn) {
						int input_idx = input_offset + (input_y * nbyn) + input_x;
						int filter_idx = filter_offset + (filter_y * 3) + filter_x;

						// 합성곱 연산 수행
						sum += inputs[input_idx] * filters[filter_idx];
					}
				}
			}
		}

		// 출력 데이터 저장
		int output_idx = (out_channel * nbyn * nbyn) + (out_y * nbyn) + out_x;
		outputs[output_idx] = sum;
	}
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
	int outDim
) {
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