#include "DataParallelNoOverlap.cuh"

#include "DataParallel.cuh"


void DataParallelNoOverlap::executeCalculation(int *matrixA, int *matrixB, int *matrixC, int *oldBoard, int *newBoard, const size_t rows, const size_t cols)
{
   //allocation for game of life
   int *d_oldBoard;
   int *d_newBoard;

   size_t pitchOld;
   size_t pitchNew;

   //allocation for matrix multiplication
   int *d_matrixA;
   int *d_matrixB;
   int *d_matrixC;

   size_t pitchMA;
   size_t pitchMB;
   size_t pitchMC;


   //allocating all the necessary memory
   cudaMallocPitch((void **)&d_oldBoard, (size_t *)&pitchOld, (size_t)cols * sizeof(int), (size_t)rows);
   cudaMallocPitch((void **)&d_newBoard, (size_t *)&pitchNew, (size_t)cols * sizeof(int), (size_t)rows);
   cudaMallocPitch((void **)&d_matrixA, (size_t *)&pitchMA, (size_t)cols * sizeof(int), (size_t)rows);
   cudaMallocPitch((void **)&d_matrixB, (size_t *)&pitchMB, (size_t)cols * sizeof(int), (size_t)rows);
   cudaMallocPitch((void **)&d_matrixC, (size_t *)&pitchMC, (size_t)cols * sizeof(int), (size_t)rows);

   //defining block and grid size
   dim3 grid(divideAndRound(rows, 16), divideAndRound(cols, 16));
   dim3 block(16, 16);

   cudaMemcpy2D(d_oldBoard, pitchOld, oldBoard, cols * sizeof(int), cols * sizeof(int), rows, cudaMemcpyHostToDevice);

   numberAliveAround << <block, grid >> > (d_oldBoard, d_newBoard, cols, rows, pitchOld, pitchNew);
   determineNextState << <block, grid >> > (d_oldBoard, d_newBoard, cols, rows, pitchOld, pitchNew);

   cudaMemcpy2D(newBoard, cols * sizeof(int), d_newBoard, pitchNew, cols * sizeof(int), rows, cudaMemcpyDeviceToHost);

   cudaMemcpy2D(d_matrixA, pitchMA, matrixA, cols * sizeof(int), cols * sizeof(int), rows, cudaMemcpyHostToDevice);
   cudaMemcpy2D(d_matrixB, pitchMB, matrixB, cols * sizeof(int), cols * sizeof(int), rows, cudaMemcpyHostToDevice);

   multiplyMatrix << <block, grid >> > (d_matrixA, d_matrixB, d_matrixC, cols, rows, pitchMA, pitchMB, pitchMC);

   cudaMemcpy2D(matrixC, cols * sizeof(int), d_matrixC, pitchMC, cols * sizeof(int), rows, cudaMemcpyDeviceToHost);

   //dealocating memory
   cudaFree(d_oldBoard);
   cudaFree(d_newBoard);
   cudaFree(d_matrixA);
   cudaFree(d_matrixB);
   cudaFree(d_matrixC);
}

string DataParallelNoOverlap::getName()
{
    return string("Data Parallel without Overlap");
}

int DataParallelNoOverlap::divideAndRound(int numberElements, int blockSize)
{
   return ((numberElements % blockSize) != 0) ? (numberElements / blockSize + 1) : (numberElements / blockSize);
}