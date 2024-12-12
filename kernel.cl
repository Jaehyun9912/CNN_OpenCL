const int FILTER_SIZE = 3;
const int FILTER_OFFSET = 9;

__kernel void convolution_batch_optimized(
	int batchSize,
	__global float* inputs,
	__global float* filters,
	__global float* convolutions,
	int inDimSize,
	int outDimSize,
	int nbyn
) {
	//입력 및 출력 차원의 맵 크기
	int mapOffset = nbyn * nbyn;

	//입력 위치
	int idx = get_local_id(1);
	int x = idx / nbyn;
	int y = idx % nbyn;

	//출력 차원의 인덱스
	int outDim = get_group_id(0);

	//입력 차원의 인덱스
	int inDim = get_group_id(2);

	//배치 전환
	for (int batch = 0; batch < batchSize; batch++)
	{
		//현 입력 차원 및 맵 위치에 대한 컨볼루션
		float sum = 0.0f;

		for (int fx = 0; fx < FILTER_SIZE; fx++)
		{
			for (int fy = 0; fy < FILTER_SIZE; fy++)
			{
				//필터와 곱할 위치를 연산
				int inputsX = x + fx - 1;
				int inputsY = y + fy - 1;

				//필터와 곱셈 후 결과값에 누적
				if (inputsX >= 0 && inputsX < nbyn && inputsY >= 0 && inputsY < nbyn)
				{
					//필터는 배치가 바뀌어도 일정함
					float filter = filters[(outDim * FILTER_OFFSET * inDimSize) + (inDim * FILTER_OFFSET) + (fx * FILTER_SIZE + fy)];

					//입력은 배치가 바뀌면 달라짐
					float input = inputs[(batch * inDimSize * mapOffset) + (inDim * mapOffset) + (inputsX * nbyn + inputsY)];

					//현 맵 위치에 대한 컨볼루션 연산
					sum += input * filter;
				}
			}
		}

		// 입력 차원에 대한 컨볼루션 값
		convolutions[
			(batch * outDimSize * inDimSize * mapOffset) +
				(outDim * inDimSize * mapOffset) +
				(inDim * mapOffset) +
				idx]
			= sum;
	}
}

__kernel void make_feature_batch_optimized(
	int batchSize,
	__global float* convolutions,
	__global float* biases,
	__global float* outputs,
	int inDimSize,
	int outDimSize,
	int nbyn
) {
	//입력 및 출력 차원의 맵 크기
	int mapOffset = nbyn * nbyn;

	//입력 위치
	int idx = get_local_id(1);

	//출력 차원의 인덱스
	int outDim = get_group_id(0);

	for (int batch = 0; batch < batchSize; batch++)
	{
		//모든 입력차원의 값을 더한다
		float sum = 0.0f;

		//입력차원 순회
		for (int inDim = 0; inDim < inDimSize; inDim++)
		{
			sum += convolutions[(batch * outDimSize * inDimSize * mapOffset) +
				(outDim * inDimSize * mapOffset) +
				(inDim * mapOffset) +
				idx];
		}

		sum += biases[outDim];

		outputs[(batch * outDimSize * mapOffset) +
			(outDim * mapOffset) +
			idx] = fmax(sum, 0.0f);
	}
}

__kernel void convolution_batch_optimized_2(
	int batchSize,
	__global float* inputs,
	__global float* outputs,
	__global float* filters,
	__global float* biases,
	int inDimSize,
	int outDimSize,
	int nbyn
) {
	//입력 및 출력 차원의 맵 크기
	int mapOffset = nbyn * nbyn;

	//배치
	int batch = get_group_id(0) / mapOffset;

	//입력 위치
	int idx = get_group_id(0) % mapOffset;
	int x = idx / nbyn;
	int y = idx % nbyn;

	//입력 차원의 인덱스
	int inDim = get_local_id(1);

	//출력 차원의 인덱스
	int outDim = get_group_id(2);

	//각 입력 차원의 컨볼루션 값을 저장하는 로컬 메모리
	__local float conv[512];

	//현 입력 차원 및 맵 위치에 대한 컨볼루션
	float sum = 0.0f;

	for (int fx = 0; fx < FILTER_SIZE; fx++)
	{
		for (int fy = 0; fy < FILTER_SIZE; fy++)
		{
			//필터와 곱할 위치를 연산
			int inputsX = x + fx - 1;
			int inputsY = y + fy - 1;

			//필터와 곱셈 후 결과값에 누적
			if (inputsX >= 0 && inputsX < nbyn && inputsY >= 0 && inputsY < nbyn)
			{
				//필터는 배치가 바뀌어도 일정함
				float filter = filters[(outDim * FILTER_OFFSET * inDimSize) + (inDim * FILTER_OFFSET) + (fx * FILTER_SIZE + fy)];

				//입력은 배치가 바뀌면 달라짐
				float input = inputs[(batch * inDimSize * mapOffset) + (inDim * mapOffset) + (inputsX * nbyn + inputsY)];

				//현 맵 위치에 대한 컨볼루션 연산
				sum += input * filter;
			}
		}
	}

	//각 입력 차원에서의 컨볼루션 값
	conv[inDim] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);

	//리덕션
	for (int p = get_local_size(1) / 2; p >= 1; p = p >> 1) {
		if (inDim < p)
			conv[inDim] += conv[inDim + p];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (inDim == 0)
	{
		//합
		float value = conv[0];

		//편차
		value += biases[outDim];

		//특성 맵 작성
		outputs[(batch * outDimSize * mapOffset) +
			(outDim * mapOffset) +
			idx] = fmax(value, 0.0f);
	}
}


__kernel void convolution_batch_optimized_3(
	int batchSize,
	__global float* inputs,
	__global float* outputs,
	__global float* filters,
	__global float* biases,
	int inDimSize,
	int outDimSize,
	int nbyn
) {
	//입력 및 출력 차원의 맵 크기
	int mapOffset = nbyn * nbyn;

	//배치
	int batch = get_group_id(2) / outDimSize;

	//입력 위치
	int lidx = get_local_id(0);
	int lx = lidx / 2;
	int ly = lidx % 2;

	int gidx = 2 * 2 * get_group_id(0) + lidx;
	int gx = gidx / nbyn;
	int gy = gidx % nbyn;

	//입력 차원의 인덱스
	int inDim = get_local_id(1);

	//출력 차원의 인덱스
	int outDim = get_group_id(2) % outDimSize;

	//각 입력 차원의 컨볼루션 값을 저장하는 로컬 메모리
	__local float conv[1024];

	//현 입력 차원 및 맵 위치에 대한 컨볼루션
	float sum = 0.0f;

	int inputsX;
	int inputsY;
	float filter;
	int filterBegin = (outDim * FILTER_OFFSET * inDimSize) + (inDim * FILTER_OFFSET);
	float input;
	int inputBegin = (batch * inDimSize * mapOffset) + (inDim * mapOffset);

	for (int fx = 0; fx < FILTER_SIZE; fx++)
	{
		inputsX = gx + fx - 1;
		if (inputsX < 0 || inputsX >= nbyn) continue;

		for (int fy = 0; fy < FILTER_SIZE; fy++)
		{
			inputsY = gy + fy - 1;

			if (inputsY >= 0 && inputsY < nbyn)
			{
				filter = filters[filterBegin + (fx * FILTER_SIZE + fy)];
				input = inputs[inputBegin + (inputsX * nbyn + inputsY)];
				sum += input * filter;
			}
		}
	}

	conv[(lidx * inDimSize) + inDim] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);

	//리덕션
	for (int p = get_local_size(1) / 2; p >= 1; p = p >> 1) {
		if (inDim < p)
			conv[(lidx * inDimSize) + inDim] += conv[(lidx * inDimSize) + inDim + p];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (inDim == 0)
	{
		//합
		float value = conv[(lidx * inDimSize)];

		//편차
		value += biases[outDim];

		//특성 맵 작성
		outputs[(batch * outDimSize * mapOffset) +
				(outDim * mapOffset) +
				gidx] = fmax(value, 0.0f);
	}
}






__kernel void convolution_batch_optimized_4(
	int batchSize,
	__global float* inputs,
	__global float* filters,
	__global float* reducts,
	int inDimSize,
	int outDimSize,
	int nbyn
) {
	//입력 및 출력 차원의 맵 크기
	int mapOffset = nbyn * nbyn;

	//배치
	int batch = get_group_id(2) / outDimSize;

	//입력 위치
	int lidx = get_local_id(0);
	int lx = lidx / 4;
	int ly = lidx % 4;

	int gidx = get_global_id(0);
	int gx = gidx / nbyn;
	int gy = gidx % nbyn;

	//입력 차원의 인덱스
	int l_inDim = get_local_id(1);
	int l_inDimSize = get_local_size(1);
	int g_inDim = l_inDimSize * get_group_id(1) + l_inDim;

	//출력 차원의 인덱스
	int outDim = get_group_id(2) % outDimSize;

	//각 입력 차원의 컨볼루션 값을 저장하는 로컬 메모리
	__local float conv[1024];

	//현 입력 차원 및 맵 위치에 대한 컨볼루션
	float sum = 0.0f;

	int inputsX;
	int inputsY;
	float filter;
	int filterBegin = (outDim * FILTER_OFFSET * inDimSize) + (g_inDim * FILTER_OFFSET);
	float input;
	int inputBegin = (batch * inDimSize * mapOffset) + (g_inDim * mapOffset);

	for (int fx = 0; fx < FILTER_SIZE; fx++)
	{
		inputsX = gx + fx - 1;
		if (inputsX < 0 || inputsX >= nbyn) continue;

		for (int fy = 0; fy < FILTER_SIZE; fy++)
		{
			inputsY = gy + fy - 1;

			if (inputsY >= 0 && inputsY < nbyn)
			{
				filter = filters[filterBegin + (fx * FILTER_SIZE + fy)];
				input = inputs[inputBegin + (inputsX * nbyn + inputsY)];
				sum += input * filter;
			}
		}
	}

	conv[(lidx * l_inDimSize) + l_inDim] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);

	//리덕션
	for (int p = l_inDimSize / 2; p >= 1; p = p >> 1) {
		if (l_inDim < p)
			conv[(lidx * l_inDimSize) + l_inDim] += conv[(lidx * l_inDimSize) + l_inDim + p];
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// 픽셀위치 * l_inDimSize(8) + get_group_id(1);
	int mapPos = (batch * outDimSize * mapOffset) + (outDim * mapOffset);
	int reductionPos = gidx * 8 + get_group_id(1);
	reducts[mapPos + reductionPos] = conv[(lidx * l_inDimSize)];
}

__kernel void make_feature_batch_optimized_4(
	int batchSize,
	__global float* reducts,
	__global float* biases,
	__global float* outputs,
	int inDimSize,
	int outDimSize,
	int nbyn
) {
	int mapOffset = nbyn * nbyn;

	int batch = get_global_id(1) / outDimSize;

	int outDim = get_global_id(1) % outDimSize;

	int gidx = get_global_id(0);

	float sum = 0.0f;

	int mapPos = (batch * outDimSize * mapOffset) + (outDim * mapOffset);

	for (int group_id = 0; group_id < 8; group_id++)
	{
		int reductionPos = gidx * 8 + group_id;
		sum += reducts[mapPos + reductionPos];
	}

	//편차
	sum += biases[outDim];

	//특성 맵 작성
	outputs[(batch * outDimSize * mapOffset) +
		(outDim * mapOffset) +
		gidx] = fmax(sum, 0.0f);
}





const int STRIDE = 2;

__kernel void max_pooling_batch_optimized(
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

__kernel void fc_layer_batch_optimized(
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