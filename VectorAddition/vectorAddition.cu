#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>


using std::vector;


//Kernel - the code that is beign run on the GPU
__global__ void vectorAdd1D(int *vectorA, int *vectorB, int *vectorOutput, int amountElements)
{
   //calculating the thread we are on
   int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

   //making sure we are not out of bounds, because there might be more threads involved then the amount of elements
   if (tid < amountElements)
   {
      //calculating
      vectorOutput[tid] = vectorA[tid] + vectorB[tid];
   }
}

int main(void)
{
   constexpr int MAX_ELEMENTS = 1 << 16;
   constexpr int NUM_THREADS = 1 << 10;
   constexpr int NUM_BLOCKS = (MAX_ELEMENTS + NUM_THREADS - 1) / NUM_THREADS;

   int vectorSize = sizeof(int) * MAX_ELEMENTS;

   //size will be defined in the for loops
   vector<int> vectorA;
   vector<int> vectorB;
   //needs to be of defined size, because elemnts will be copied over pointers
   vector<int> vectorC(MAX_ELEMENTS);

   //generating random vectors
   for (int i = 0; i < MAX_ELEMENTS; i++) { vectorA.push_back(rand() % 100); }
   for (int i = 0; i < MAX_ELEMENTS; i++) { vectorB.push_back(rand() % 100); }

   //pointers for GPU memory
   int *gpuVectorA;
   int *gpuVectorB;
   int *gpuVectorC;

   //Allocating GPU memory
   cudaMalloc(&gpuVectorA, vectorSize);
   cudaMalloc(&gpuVectorB, vectorSize);
   cudaMalloc(&gpuVectorC, vectorSize);

   //Transfering data to from CPU to GPU
   cudaMemcpy(gpuVectorA, vectorA.data(), vectorSize, cudaMemcpyHostToDevice);
   cudaMemcpy(gpuVectorB, vectorB.data(), vectorSize, cudaMemcpyHostToDevice);

   //calling the function
   vectorAdd1D <<<NUM_BLOCKS, NUM_THREADS>>> (gpuVectorA, gpuVectorB, gpuVectorC, MAX_ELEMENTS);

   //retrieveing data from GPU
   cudaMemcpy(vectorC.data(), gpuVectorC, vectorSize, cudaMemcpyDeviceToHost);

   cudaFree(gpuVectorA);
   cudaFree(gpuVectorB);
   cudaFree(gpuVectorC);

   std::cout << "end" << std::endl;

   return 1;
}