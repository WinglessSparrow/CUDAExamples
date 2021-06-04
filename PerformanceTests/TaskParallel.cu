#include "TaskParallel.cuh"

#include "DataParallel.cuh"

void TaskParallel::executeCalculation(int *matrixA, int *matrixB, int *matrixC, int *oldBoard, int *newBoard, const size_t rows, const size_t cols)
{
   const int CUDA_CORES = 1920;
   const int dataSize = rows * cols;

   // not all cuda cores are being utilized by the graphics card
   dim3 grid(2, 2);
   dim3 block(16, 16);

   const int streamWidth = grid.x * grid.y * block.x * block.y;
   const int numKernelCalls = (dataSize / streamWidth < 1) ? 1 : (dataSize / streamWidth + 1);


   //creating streams
   cudaStream_t *streams = new cudaStream_t[numKernelCalls * 2];
   for (int i = 0; i < 2; i++)
   {
      cudaStreamCreate(&streams[i]);
   }


   //allocation for game of life
   int *d_oldBoard;
   int *d_newBoard;

   pitchesBoard boardPitches;

   //allocation for matrix multiplication
   int *d_matrixA;
   int *d_matrixB;
   int *d_matrixC;

   pitchesMatrix matrixPitches;

   //allocating memory
   cudaMallocPitch((void **)&d_oldBoard, (size_t *)&boardPitches.pitchOld, (size_t)cols * sizeof(int), (size_t)rows);
   cudaMallocPitch((void **)&d_newBoard, (size_t *)&boardPitches.pitchNew, (size_t)cols * sizeof(int), (size_t)rows);
   cudaMallocPitch((void **)&d_matrixA, (size_t *)&matrixPitches.pitchMA, (size_t)cols * sizeof(int), (size_t)rows);
   cudaMallocPitch((void **)&d_matrixB, (size_t *)&matrixPitches.pitchMB, (size_t)cols * sizeof(int), (size_t)rows);
   cudaMallocPitch((void **)&d_matrixC, (size_t *)&matrixPitches.pitchMC, (size_t)cols * sizeof(int), (size_t)rows);

   cudaMemcpy2D(d_oldBoard, boardPitches.pitchOld, oldBoard, cols * sizeof(int), cols * sizeof(int), rows, cudaMemcpyHostToDevice);
   cudaMemcpy2D(d_matrixA, matrixPitches.pitchMA, matrixA, cols * sizeof(int), cols * sizeof(int), rows, cudaMemcpyHostToDevice);
   cudaMemcpy2D(d_matrixB, matrixPitches.pitchMB, matrixB, cols * sizeof(int), cols * sizeof(int), rows, cudaMemcpyHostToDevice);

   //starting the streams
   int offset = 0;
   for (int i = 0; i < numKernelCalls; i++)
   {
      numberAliveAroundOffset << <block, grid, 0, streams[0] >> > (d_oldBoard, d_newBoard, cols, rows, boardPitches, offset * streamWidth);
      determineNextStateOffset << <block, grid, 0, streams[0] >> > (d_oldBoard, d_newBoard, cols, rows, boardPitches, offset * streamWidth);
      multiplyMatrixOffset << <block, grid, 0, streams[1] >> > (d_matrixA, d_matrixB, d_matrixC, cols, rows, matrixPitches, offset * streamWidth);
      offset++;
   }

   for (int i = 0; i < 2; i++)
   {
      cudaStreamSynchronize(streams[i]);
      cudaStreamDestroy(streams[i]);
   }

   cudaMemcpy2D(newBoard, cols * sizeof(int), d_newBoard, boardPitches.pitchNew, cols * sizeof(int), rows, cudaMemcpyDeviceToHost);
   cudaMemcpy2D(matrixC, cols * sizeof(int), d_matrixC, matrixPitches.pitchMC, cols * sizeof(int), rows, cudaMemcpyDeviceToHost);

   cudaFree(d_oldBoard);
   cudaFree(d_newBoard);
   cudaFree(d_matrixA);
   cudaFree(d_matrixB);
   cudaFree(d_matrixC);

   delete[] streams;
}

string TaskParallel::getName()
{
   return string("Task Parallel");
}

int TaskParallel::divideAndRound(int numberElements, int blockSize)
{
   return ((numberElements % blockSize) != 0) ? (numberElements / blockSize + 1) : (numberElements / blockSize);
}
