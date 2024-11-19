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

__kernel void fc_layer(
	__global float* global_input,
	__local float* local_input,
	__global float* global_output,
	__global float* weights,
	__global float* biases,
	int inDim,
	int outDim
) {

	//global_input 글로벌메모리의 입력뉴런
	//local_input 로컬메모리의 입력뉴런
	//global_output 출력뉴런의 값을 적는 글로벌메모리
	//weight 글로벌메모리의 가중치
	//biases 글로벌메모리의 편차
	//inDim 입력 차원의 크기
	//outDim 출력 차원의 크기

	//1.
	//주어진 신경망에서 완전연결레이어는 inDim개 입력뉴런과 outDim개 출력뉴런으로 구성된다
	//각 출력뉴런은 자신의 값을 구하기 위해서 inDim개의 입력뉴런의 값을 매번 읽어야 한다
	//하지만 입력뉴런의 값은 동일하므로 글로벌 메모리에 계속 들고 있다면 오버헤드가 발생한다
	//따라서 글로벌메모리에 있는 입력뉴런을 로컬메모리인 local_input으로 복사한다

	//2.
	//가중치의 경우 출력 뉴런이 각 간선을 공유하지는 않으므로 어차피 1번 접근하게 됨
	//바이어스의 경우 각 출력 뉴런이 하나씩 가지고 있으므로 어차피 1번 접근하게 됨
	//따라서 로컬메모리에 복사하더라도 이득이 없다

	//워크아이템은 Max(inDim, outDim)만큼 존재한다

	//입력 뉴런의 id를 얻는다
	int gid = get_global_id(0);

	//입력차원만큼 로컬메모리에 로드해야 한다 - 그 외는 로드할 필요 없음
	if (gid < inDim)
	{
		//각자 입력뉴런의 값을 글로벌메모리에서 로드하여 로컬메모리에 로드한다
		local_input[gid] = global_input[gid];
	}

	//모든 워크아이템이 로컬메모리 로드를 복사를 완료할 때까지 대기한다
	barrier(CLK_LOCAL_MEM_FENCE);

	//이제 로컬메모리에 inDim개의 입력뉴런의 값이 전부 복사되어 있다

	//출력차원만큼 연산이 이루어진다 - 그 외는 연산할 필요 없음
	if (gid >= outDim)
		return;

	//출력뉴런에 따라서 입력뉴런과 이루는 간선의 가중치가 달라진다
	float sum = 0.0f;
	for (int i = 0; i < inDim; ++i)
		sum += local_input[i] * weights[gid * inDim + i];

	//출력뉴런의 편차 적용
	sum += biases[gid];

	//활성화 함수 적용
	if (sum > 0) global_output[gid] = sum;
	else global_output[gid] = 0;
}

const int STRIDE = 2;

__kernel void max_pooling(
	__global float* input,
	__global float* output,
	int nbyn
) {
	int dim = get_global_id(0);
	int row = get_local_id(0);
	int col = get_local_id(1);

	float max = -FLT_MAX;
	for (int y = 0; y < STRIDE; y++) {
		for (int x = 0; x < STRIDE; x++) {
			float temp = input[nbyn * (row + y) + col + x];
			if (max < temp) max = temp;
		}
	}

	output[(dim * nbyn * nbyn) + (nbyn * row + col)] = max;
}