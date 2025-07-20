// vim: et:ts=4:sts=4:sw=4

// SPDX-License-Identifier: MIT
// Copyright © 2025 David Llewellyn-Jones

#include <cstdlib>
#include <cstdio>

__global__ void my_first_kernel(float *x);

int main(int argc, char **argv) {
    float *h_x, *d_x;
    int nblocks = 2, nthreads = 8, nsize = 2 * 8;

    h_x = (float*)malloc(nsize * sizeof(float));
    cudaMalloc((void**) &d_x, nsize * sizeof(float));

    my_first_kernel<<<nblocks, nthreads>>>(d_x);

    cudaMemcpy(h_x, d_x, nsize * sizeof(float), cudaMemcpyDeviceToHost);

    for (int n = 0; n < nsize; n++) {
        printf(" n, x = %d %f\n", n, h_x[n]);
    }

    cudaFree(d_x);
    free(h_x);
}

