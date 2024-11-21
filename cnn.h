#ifndef _CNN_H
#define _CNN_H

#define CLK_TCK CLOCKS_PER_SEC

void cnn_seq(float* images, float* network, int* labels, float* confidences, int num_of_image);
void cnn(float* images, float* network, int* labels, float* confidences, int num_images);
void compare(const char* filename, int num_of_image);
#endif 