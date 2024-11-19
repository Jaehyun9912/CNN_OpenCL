//
// Created by Ozeco-Labtop on 2024. 11. 19..
//
#include "CL/cl.h"
#include "cnn.h"

cl_int err;
cl_platform_id Platform;
cl_device_id Device;
cl_context Context;
cl_command_queue Queue;

// Create kernel
cl_kernel MaxPoolingKernel = clCreateKernel(program, "max_pooling", NULL);
CHECK_ERROR(err);

void max_pooling(float* input, float* output, int dim, int nbyn) {

    cl_mem input_buffer = clCreateBuffer(Context, CL_MEM_READ_ONLY, sizeof(float) * dim * nbyn * nbyn, NULL, &err);
    CHECK_ERROR(err);

    cl_mem output_buffer = clCreateBuffer(Context, CL_MEM_READ_WRITE, sizeof(float) * dim * nbyn * nbyn, NULL, &err);
    CHECK_ERROR(err);


    // Set Kernel Args
    err = clSetKernelArg(MaxPoolingKernel, 0, sizeof(cl_mem), &input_buffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(MaxPoolingKernel, 1, sizeof(cl_mem), &output_buffer);
    CHECK_ERROR(err);
    err = clSetKernelArg(MaxPoolingKernel, 2, sizeof(cl_mem), &nbyn);
    CHECK_ERROR(err);

    // Set Work Size
    size_t global_item_size[] = { (size_t)dim, (size_t)(nbyn * nbyn / 4) };
    size_t local_item_size[] = {(size_t)nbyn/2, (size_t)nbyn/2};

    // Run Kernel
    err = clEnqueueNDRangeKernel(Queue, MaxPoolingKernel, 2, global_item_size, local_item_size, NULL, 0, NULL, NULL);
    CHECK_ERROR(err);

    err = clEnqueueReadBuffer(Queue, output_buffer, CL_TRUE, 0, sizeof(float) * dim * nbyn * nbyn, output, 0, NULL, NULL);
    CHECK_ERROR(err);


    // Release Memory
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseKernel(MaxPoolingKernel);
}
