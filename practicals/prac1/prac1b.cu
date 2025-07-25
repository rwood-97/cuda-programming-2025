//
// include files
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <helper_cuda.h>


//
// kernel routine
// 

__global__ void my_first_kernel(float *d_va, float *d_vb, float *d_x)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;

  d_x[tid] = d_va[tid] + d_vb[tid];
}


//
// main code
//

int main(int argc, const char **argv)
{
  int   nblocks, nthreads, nsize, n; 
  float h_va[] = {0.0, 1.0, 10.0};
  float h_vb[] = {1.0, 0.0, 10.0};
  float *h_x;
  float *d_va;
  float *d_vb;
  float *d_x;

  // initialise card

  findCudaDevice(argc, argv);

  // set number of blocks, and threads per block

  nblocks  = 1;
  nthreads = sizeof(h_va)/ sizeof(float);
  nsize    = nblocks*nthreads ;

  // allocate memory for array
  h_x = (float *)malloc(sizeof(h_va)); 
 
  checkCudaErrors(cudaMalloc((void **)&d_va, sizeof(h_va)));
  checkCudaErrors(cudaMalloc((void **)&d_vb, sizeof(h_vb)));

  checkCudaErrors(cudaMalloc((void **)&d_x, sizeof(h_va)));
 
  // move from host to device
  checkCudaErrors( cudaMemcpy(d_va,h_va, sizeof(h_va),
                 cudaMemcpyHostToDevice) );
  checkCudaErrors( cudaMemcpy(d_vb,h_vb, sizeof(h_vb),
                 cudaMemcpyHostToDevice) );

  // execute kernel
  
  my_first_kernel<<<nblocks,nthreads>>>(d_va, d_va, d_x);
  getLastCudaError("my_first_kernel execution failed\n");

  // copy back results and print them out

  checkCudaErrors( cudaMemcpy(h_x,d_x, sizeof(h_x),
                 cudaMemcpyDeviceToHost) );

  for (n=0; n<sizeof(h_va); n++) printf(" res %d= %f \n", n,h_x[n]);

  // free memory 

  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_va));
  checkCudaErrors(cudaFree(d_vb));
  free(h_x);

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

  return 0;
}
