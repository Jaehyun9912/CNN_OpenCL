const int FILTER_SIZE = 3;          // 필터 크기 (3x3)


const int FILTER_OFFSET = FILTER_SIZE * FILTER_SIZE;

__kernel void convolution(
	__global float* inputs,
	__global float* outputs,
	__global float* filters,
	__global float* biases,
	__local float* l_filter,
	__local float* buffer,
	int inDimSize,
	int outDimSize,
	int nbyn
) {
	const int IMG_SIZE = nbyn * nbyn;  // size of ONE Image

	// 현재 배치
	int batch = get_group_id(2);

	/// 현재 이미지
	int outDim = get_group_id(0);   // 필터셋 인덱스
    int inDim = get_local_id(0);

    // 현재 오프셋 (n x n)
	int idx = get_group_id(1);  // 이미지 내 인덱스
	int row = idx / nbyn;
	int col = idx % nbyn;

    // input에서 연산할 좌표
	int offset = 0;
	offset += batch * (inDimSize * IMG_SIZE);    // 배치만큼 이동 (특정 이미지까지)
	offset += inDim * IMG_SIZE;    // 이미지에서 imDim까지 이동
	offset += idx;     // 레이어에서 특정 좌표까지 이동 (목표까지)

	// 연산에 적용할 필터 불러오기
	int filterOffset = 0;
	filterOffset += outDim * (inDimSize * FILTER_OFFSET);   // 필터 그룹만큼 이동
	if (inDim == 0)
	{
	    for(int i = 0; i < inDimSize * FILTER_OFFSET; i++)
	        l_filter[i] = filters[filterOffset + i];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// 하나의 셀 연산 (필터 9칸 연산결과 합)
	float cellSum = 0;
	for (int y = -1; y < 2; y++)
	{
		if ((row + y) < 0 || (row + y) >= nbyn) continue;
		for (int x = -1; x < 2; x++)
		{
			if ((col + x) < 0 || (col + x) >= nbyn) continue;

			cellSum += inputs[offset + y * nbyn + x] * l_filter[inDim * FILTER_OFFSET + y * FILTER_SIZE + x];
		}
	}
	buffer[inDim] = cellSum;


	barrier(CLK_LOCAL_MEM_FENCE);


	/// 예외 처리 (inDimSize == 3일때)
	if (inDimSize == 3)
	{
		if (inDim == 0)
		{
			for (int p = 1; p < inDimSize; p++)
			{
				buffer[inDim] += buffer[inDim + p];
			}
			outputs[batch * outDimSize * IMG_SIZE + outDim * IMG_SIZE + idx] = fmax(buffer[inDim] + biases[outDim], 0.0f);
			return;
		}
	}

	/// 필터 연산 결과(inDim)별로 덧셈
	/// Reduction 연산 진행 (buffer에서 읽음)
	for (int p = inDimSize / 2; p >= 1; p = p >> 1)
	{
		if (inDim < p) buffer[inDim] += buffer[inDim + p];  // 내 위치로 합 연산 (나 + 나의 다음 inDim offset)
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	}

	// 마지막 reduction 후 결과 저장
	if (inDim == 0)
	{
		outputs[batch * outDimSize * IMG_SIZE + outDim * IMG_SIZE + idx] = fmax(buffer[0] + biases[outDim], 0.0f);
	}
}


const int STRIDE = 2;

__kernel void max_pooling(
	int batchSize,
	__global float* inputs,
	__global float* outputs,
	int dimSize,
	int nbyn
) {
	//차원의 인덱스
	int dim = get_group_id(0);

	//최대 풀링이 완료된 특성 맵의 크기
	int mapOffset = nbyn * nbyn / 4;

	//최대 풀링이 완료된 특성 맵에서의 위치
	int row = get_local_id(1);
	int col = get_local_id(2);

	for (int batch = 0; batch < batchSize; batch++)
	{
		float max = -FLT_MAX;
		for (int y = 0; y < 2; y++) {
			for (int x = 0; x < 2; x++) {
				float temp = inputs[(batch * dimSize * nbyn * nbyn) + (dim * nbyn * nbyn) + nbyn * (2 * row + y) + 2 * col + x];
				if (max < temp)
					max = temp;
			}
		}

		//배치-차원-맵 위치를 고려하여 글로벌 메모리에 작성
		outputs[(batch * dimSize * mapOffset) + (dim * mapOffset) + (row * nbyn / 2) + col] = max;
	}
}

__kernel void fc_layer(
	int batchSize,
	__global float* inputs,
	__global float* outputs,
	__global float* weights,
	__global float* biases,
	int inDimSize,
	int outDimSize
) {
	//출력 차원
	int outDim = get_global_id(0);

	for (int batch = 0; batch < batchSize; batch++)
	{
		float sum = 0.0f;

		//입력차원이 512라고 한다면
		for (int inDim = 0; inDim < inDimSize; inDim++)
			sum += inputs[(batch * inDimSize) + inDim] * weights[outDim * inDimSize + inDim];

		//편차 적용
		sum += biases[outDim];

		//배치만큼 이동 후, 이 워크 아이템의 출력차원에 활성화 함수를 적용한 값을 작성
		outputs[(batch * outDimSize) + outDim] = fmax(sum, 0.0f);
	}
}
