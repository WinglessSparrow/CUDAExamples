#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <chrono>


using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::cout;
using std::endl;

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
   constexpr int MAX_ELEMENTS = 1 << 4;
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

   for (int i = 0; i < MAX_ELEMENTS; i++)
   {
      cout << vectorA[i] << "+" << vectorB[i] << endl;
   }


   //pointers for GPU memory
   int *d_VectorA;
   int *d_VectorB;
   int *d_VectorC;

   auto timer1 = high_resolution_clock::now();
   //Allocating GPU memory
   cudaMalloc(&d_VectorA, vectorSize);
   cudaMalloc(&d_VectorB, vectorSize);
   cudaMalloc(&d_VectorC, vectorSize);

   //Transfering data to from CPU to GPU
   cudaMemcpy(d_VectorA, vectorA.data(), vectorSize, cudaMemcpyHostToDevice);
   cudaMemcpy(d_VectorB, vectorB.data(), vectorSize, cudaMemcpyHostToDevice);

   //calling the function
   vectorAdd1D << <NUM_BLOCKS, NUM_THREADS >> > (d_VectorA, d_VectorB, d_VectorC, MAX_ELEMENTS);

   //retrieveing data from GPU
   cudaMemcpy(vectorC.data(), d_VectorC, vectorSize, cudaMemcpyDeviceToHost);

   auto timer2 = high_resolution_clock::now();

   auto ms_int = duration_cast<milliseconds>(timer2 - timer1);
   duration<double, std::milli> ms_double = timer2 - timer1;

   cudaFree(d_VectorA);
   cudaFree(d_VectorB);
   cudaFree(d_VectorC);

   for (int i = 0; i < MAX_ELEMENTS; i++)
   {
      cout << vectorA[i] << "+" << vectorB[i] << "=" << vectorC[i] << endl;
   }

   std::cout << ms_int.count() << "ms\n";
   std::cout << ms_double.count() << "ms";

   std::cout << "end" << std::endl;

   return 1;
}