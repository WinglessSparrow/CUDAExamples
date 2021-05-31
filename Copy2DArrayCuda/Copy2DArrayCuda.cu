#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <iostream>
#include <chrono>

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32

#define columns 4
#define rows 5

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
      //d_matrix[x * xDim + y + pitch]++;
      printf("X: %d Y: %d Val: %d\n", x, y, d_matrix[y * pitch + x]);
      d_matrix[y * pitch + x]++;

   }
}

int main()
{
   //2D array
   int matrix[columns][rows];

   //fill with numbers
   int count = 0;
   for (int i = 0; i < columns; i++)
   {
      for (int j = 0; j < rows; j++)
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

   cudaMallocPitch((void **)&d_matrix, (size_t *)&pitch, (size_t)columns * sizeof(int), (size_t)rows);
   cudaMemcpy2D(d_matrix, pitch, matrix, columns * sizeof(int), columns * sizeof(int), rows, cudaMemcpyHostToDevice);

   dim3 grid(divideAndRound(columns, BLOCKSIZE_X), divideAndRound(rows, BLOCKSIZE_Y));
   dim3 block(BLOCKSIZE_Y, BLOCKSIZE_X);

   addToEachValueInMatrix << <block, grid >> > (d_matrix, columns, rows, pitch / sizeof(int));

   //cudaMemcpy2D(matrix, pitch, d_matrix, X * sizeof(int), X * sizeof(int), Y, cudaMemcpyDeviceToHost);
   cudaMemcpy2D(matrix, columns * sizeof(int), d_matrix, pitch, columns * sizeof(int), rows, cudaMemcpyDeviceToHost);

   cudaFree(d_matrix);

   for (int i = 0; i < columns; i++)
   {
      for (int j = 0; j < rows; j++)
      {
         printf("%u\n", matrix[i][j]);
      }
   }
}