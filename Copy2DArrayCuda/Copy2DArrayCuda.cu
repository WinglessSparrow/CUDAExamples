#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <chrono>

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32

#define COLLUMNS 4
#define ROWS 5

using namespace std;

int divideAndRound(int numberElements, int blockSize)
{
   return ((numberElements % blockSize) != 0) ? (numberElements / blockSize + 1) : (numberElements / blockSize);
}

__global__ void addToEachValueInMatrix(int *d_matrix, int xDim, int yDim, size_t pitch)
{
   int x = blockIdx.x * blockDim.x + threadIdx.x;
   int y = blockIdx.y * blockDim.y + threadIdx.y;

   if (x < xDim && y < yDim)
   {
      //with pointer arythmetics
      /*int *value = (int *)((char *)(d_matrix + y * pitch)) + x;
      (*value)++;*/

      //less cancer
      d_matrix[x * xDim + y + pitch]++;

   }
}

int main()
{
   //2D array
   int matrix[COLLUMNS][ROWS];

   //fill with numbers
   int count = 0;
   for (int i = 0; i < COLLUMNS; i++)
   {
      for (int j = 0; j < ROWS; j++)
      {
         count++;
         matrix[i][j] = count;
         printf("%u\n", matrix[i][j]);
      }
   }

   printf("=======\n");

   //device mem
   int *d_matrix;
   //pitch = offset in 1d array, because cuda is not a big friend of 2 arrays
   size_t pitch; extern void CudaMain(void);

   cudaMallocPitch(&d_matrix, &pitch, COLLUMNS * sizeof(int), ROWS);
   cudaMemcpy2D(d_matrix, pitch, matrix, COLLUMNS * sizeof(int), COLLUMNS * sizeof(int), ROWS, cudaMemcpyHostToDevice);

   dim3 grid(divideAndRound(COLLUMNS, BLOCKSIZE_X), divideAndRound(ROWS, BLOCKSIZE_Y));
   dim3 block(BLOCKSIZE_Y, BLOCKSIZE_X);

   addToEachValueInMatrix << <block, grid >> > (d_matrix, COLLUMNS, ROWS, pitch);

   //cudaMemcpy2D(matrix, pitch, d_matrix, X * sizeof(int), X * sizeof(int), Y, cudaMemcpyDeviceToHost);
   cudaMemcpy2D(matrix, COLLUMNS * sizeof(int), d_matrix, pitch, COLLUMNS * sizeof(int), ROWS, cudaMemcpyDeviceToHost);

   cudaFree(d_matrix);

   for (int i = 0; i < COLLUMNS; i++)
   {
      for (int j = 0; j < ROWS; j++)
      {
         printf("%u\n", matrix[i][j]);
      }
   }
}