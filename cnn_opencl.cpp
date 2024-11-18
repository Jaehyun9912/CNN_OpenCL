#include "cnn.h"
#include <time.h>
#include <stdio.h>

void cnn_init() {
	
	//TODO

}

void cnn(float* images, float** network, int* labels, float* confidences, int num_images) {

	//cnn_init();

	time_t start, end;
	start = clock();
	//TODO
	end = clock();
	printf("Elapsed time: %.2f sec\n", (double)(end - start) / CLK_TCK);

}
