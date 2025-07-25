
////////////////////////////////////////////////////////////////////////
// GPU version of Monte Carlo algorithm using NVIDIA's CURAND library
////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>

#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////
// CUDA global constants
////////////////////////////////////////////////////////////////////////

__constant__ int   N;
__constant__ float d_a, d_b, d_c;


////////////////////////////////////////////////////////////////////////
// kernel routine
////////////////////////////////////////////////////////////////////////


__global__ void pathcalc(float *d_z, float *d_v)
{
  float s1, z, y1;
  int   ind;

  // move array pointers to correct position

  // version 1
  ind = threadIdx.x + N*blockIdx.x*blockDim.x;

  // path calculation

  s1 = 0.0f;

  for (int n=0; n<N; n++) {
    z = d_z[ind];
    y1 = d_a*z*z + d_b*z + d_c;
    s1 += y1;

    // version 1
    ind += blockDim.x;      // shift pointer to next element
  }

  // put av value into device array
  
  d_v[threadIdx.x + blockIdx.x*blockDim.x] = s1/N;
}


////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////

int main(int argc, const char **argv){
    
  int     NPATH=9600000, h_N=100;
  float  *h_v, *d_v, *d_z;
  double  sum1;

  // initialise card

  findCudaDevice(argc, argv);

  // initialise CUDA timing

  float milli;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // allocate memory on host and device

  h_v = (float *)malloc(sizeof(float)*NPATH);

  checkCudaErrors( cudaMalloc((void **)&d_v, sizeof(float)*NPATH) );
  checkCudaErrors( cudaMalloc((void **)&d_z, sizeof(float)*h_N*NPATH) );

  // define constants and transfer to GPU

  checkCudaErrors( cudaMemcpyToSymbol(N,    &h_N,    sizeof(h_N)) );
  
  float h_a = 10.0f;
  float h_b = 100.0f;
  float h_c = 1.0f;
  
  checkCudaErrors( cudaMemcpyToSymbol(d_a, &h_a, sizeof(h_a)) );
  checkCudaErrors( cudaMemcpyToSymbol(d_b, &h_b, sizeof(h_b)) );
  checkCudaErrors( cudaMemcpyToSymbol(d_c, &h_c, sizeof(h_c)) );
  
  // random number generation

  curandGenerator_t gen;
  checkCudaErrors( curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) );
  checkCudaErrors( curandSetPseudoRandomGeneratorSeed(gen, 1234ULL) );

  cudaEventRecord(start);
  checkCudaErrors( curandGenerateNormal(gen, d_z, h_N*NPATH, 0.0f, 1.0f) );
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  printf("CURAND normal RNG  execution time (ms): %f,  samples/sec: %e \n",
          milli, h_N*NPATH/(0.001*milli));

  // execute kernel and time it

  cudaEventRecord(start);
  pathcalc<<<NPATH/128, 128>>>(d_z, d_v);
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milli, start, stop);

  getLastCudaError("pathcalc execution failed\n");
  printf("Monte Carlo kernel execution time (ms): %f \n",milli);

  // copy back results

  checkCudaErrors( cudaMemcpy(h_v, d_v, sizeof(float)*NPATH,
                   cudaMemcpyDeviceToHost) );

  // compute average

  sum1 = 0.0;
  for (int i=0; i<NPATH; i++) {
    sum1 += h_v[i];
  }

  printf("\nAverage value and standard deviation of error  = %13.8f\n\n",
	 sum1/NPATH);

  // Tidy up library

  checkCudaErrors( curandDestroyGenerator(gen) );

  // Release memory and exit cleanly

  free(h_v);
  checkCudaErrors( cudaFree(d_v) );
  checkCudaErrors( cudaFree(d_z) );

  // CUDA exit -- needed to flush printf write buffer

  cudaDeviceReset();

}
