
const int STRIDE = 2

__kernel void max_pooling(__global float* input,
                          __global float* output,
                          int nbyn) {
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