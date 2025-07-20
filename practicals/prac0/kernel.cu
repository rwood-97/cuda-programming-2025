// vim: et:ts=4:sts=4:sw=4

// SPDX-License-Identifier: MIT
// Copyright © 2025 David Llewellyn-Jones

#include <helper_cuda.h>

__global__ void my_first_kernel(float *x) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    x[tid] = (float)threadIdx.x;
}

