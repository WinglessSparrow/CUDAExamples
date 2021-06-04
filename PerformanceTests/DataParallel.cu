#include "DataParallel.cuh"

void DataParallel::executeCalculation(int *matrixA, int *matrixB, int *matrixC, int *oldBoard, int *newBoard, const size_t rows, const size_t cols)
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

   //creating streams for memmory overlap
   cudaStream_t stream1, stream2;

   cudaStreamCreate(&stream1);
   cudaStreamCreate(&stream2);

   //allocating all the necessary memory
   cudaMallocPitch((void **)&d_oldBoard, (size_t *)&pitchOld, (size_t)cols * sizeof(int), (size_t)rows);
   cudaMallocPitch((void **)&d_newBoard, (size_t *)&pitchNew, (size_t)cols * sizeof(int), (size_t)rows);
   cudaMallocPitch((void **)&d_matrixA, (size_t *)&pitchMA, (size_t)cols * sizeof(int), (size_t)rows);
   cudaMallocPitch((void **)&d_matrixB, (size_t *)&pitchMB, (size_t)cols * sizeof(int), (size_t)rows);
   cudaMallocPitch((void **)&d_matrixC, (size_t *)&pitchMC, (size_t)cols * sizeof(int), (size_t)rows);

   //defining block and grid size
   dim3 grid(divideAndRound(rows, 16), divideAndRound(cols, 16));
   dim3 block(16, 16);

   //stream1
   cudaMemcpy2DAsync(d_oldBoard, pitchOld, oldBoard, cols * sizeof(int), cols * sizeof(int), rows, cudaMemcpyHostToDevice, stream1);

   numberAliveAround << <block, grid, 0, stream1 >> > (d_oldBoard, d_newBoard, cols, rows, pitchOld, pitchNew);
   determineNextState << <block, grid, 0, stream1 >> > (d_oldBoard, d_newBoard, cols, rows, pitchOld, pitchNew);

   //stream 2
   cudaMemcpy2DAsync(d_matrixA, pitchMA, matrixA, cols * sizeof(int), cols * sizeof(int), rows, cudaMemcpyHostToDevice, stream2);
   cudaMemcpy2DAsync(d_matrixB, pitchMB, matrixB, cols * sizeof(int), cols * sizeof(int), rows, cudaMemcpyHostToDevice, stream2);

   multiplyMatrix << <block, grid, 0, stream1 >> > (d_matrixA, d_matrixB, d_matrixC, cols, rows, pitchMA, pitchMB, pitchMC);

   cudaMemcpy2DAsync(newBoard, cols * sizeof(int), d_newBoard, pitchNew, cols * sizeof(int), rows, cudaMemcpyDeviceToHost, stream1);

   cudaMemcpy2DAsync(matrixC, cols * sizeof(int), d_matrixC, pitchMC, cols * sizeof(int), rows, cudaMemcpyDeviceToHost, stream2);

   cudaDeviceSynchronize();

   //dealocating memory
   cudaFree(d_oldBoard);
   cudaFree(d_newBoard);
   cudaFree(d_matrixA);
   cudaFree(d_matrixB);
   cudaFree(d_matrixC);

   cudaStreamDestroy(stream1);
   cudaStreamDestroy(stream2);
}

string DataParallel::getName()
{
   return string("Data parallel with overlap");
}

int DataParallel::divideAndRound(int numberElements, int blockSize)
{
   return ((numberElements % blockSize) != 0) ? (numberElements / blockSize + 1) : (numberElements / blockSize);
}