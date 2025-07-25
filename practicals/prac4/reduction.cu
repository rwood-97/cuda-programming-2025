////////////////////////////////////////////////////////////////////////
//
// Practical 4 -- initial code for shared memory reduction for 
//                a single block which is a power of two in size
//
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// CPU routine
////////////////////////////////////////////////////////////////////////

float reduction_gold(float* idata, int len) 
{
  float sum = 0.0f;
  for(int i=0; i<len; i++) sum += idata[i];

  printf("Actual result: %f\n", sum);
  return sum;
}

////////////////////////////////////////////////////////////////////////
// GPU routine
////////////////////////////////////////////////////////////////////////

__global__ void reduction(float *g_odata, float *g_idata)
{
    // dynamically allocated shared memory

    float temp;

    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int id = tid + blockDim.x*bid;

    // first, each thread loads data into shared memory

    temp = g_idata[id];
    __syncthreads();

    // next, we perform binary tree reduction

    // find the largest power of 2 less than blocksize
    int m;
    for (m=1; m < blockDim.x; m=2*m) {}
    m=m/2; // value has been doulbed one too many times

    if (tid == 0) {
      printf("m start value = %d\n", m);
    }

    for (int d=m; d>0; d=d/2) {
      __syncthreads();  // ensure previous step completed
      if (tid == 0) {
        printf("m = %d\n", d);
      }
      temp += __shfl_down_sync(-1, temp, d);
    }

    // finally, first thread puts result into global memory

    if (tid == 0) {
      printf("saving value .. %f\n", temp);
      g_odata[bid] = temp;
    }
}


////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////

int main( int argc, const char** argv) 
{
  int num_blocks, num_threads, num_elements, mem_size, shared_mem_size;

  float *h_data, *d_idata, *d_odata;

  // initialise card

  findCudaDevice(argc, argv);

  num_blocks   = 2;  // start with only 1 thread block
  num_threads  = 30;
  num_elements = num_blocks*num_threads;
  mem_size     = sizeof(float) * num_elements;

  // allocate host memory to store the input data
  // and initialize to integer values between 0 and 10

  h_data = (float*) malloc(mem_size);
      
  for(int i = 0; i < num_elements; i++) 
    h_data[i] = floorf(10.0f*(rand()/(float)RAND_MAX));

  // compute reference solution

  float sum = reduction_gold(h_data, num_elements);

  // allocate device memory input and output arrays

  checkCudaErrors( cudaMalloc((void**)&d_idata, mem_size) );
  checkCudaErrors( cudaMalloc((void**)&d_odata, num_blocks*sizeof(float)) );

  // copy host memory to device input array

  checkCudaErrors( cudaMemcpy(d_idata, h_data, mem_size,
                              cudaMemcpyHostToDevice) );

  // execute the kernel

  shared_mem_size = sizeof(float) * num_threads;
  reduction<<<num_blocks,num_threads,shared_mem_size>>>(d_odata,d_idata);
  getLastCudaError("reduction kernel execution failed");

  // copy result from device to host

  checkCudaErrors( cudaMemcpy(h_data, d_odata, num_blocks*sizeof(float),
                              cudaMemcpyDeviceToHost) );

  float total = 0;
  for (int i=0; i<num_blocks; i++) {
    total += h_data[i];
  }

  // check results

  printf("reduction error = %f\n",total-sum);

  // cleanup memory

  free(h_data);
  checkCudaErrors( cudaFree(d_idata) );
  checkCudaErrors( cudaFree(d_odata) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();
}
