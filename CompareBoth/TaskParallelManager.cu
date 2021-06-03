#include "TaskParallelManager.cuh"

__device__ void prepareIndexexPitches(int *row, int *column, size_t *pitchOld, size_t *pitchNew)
{
   *row = (blockIdx.x * blockDim.x) + threadIdx.x;
   *column = (blockIdx.y * blockDim.y) + threadIdx.y;

   *pitchOld = *pitchOld / sizeof(int);
   *pitchNew = *pitchNew / sizeof(int);
}

__global__ void checkAbove(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew)
{
   int row, column;

   prepareIndexexPitches(&row, &column, &pitchOld, &pitchNew);

   if (row < rows && column < columns)
   {
      int idx = ((column - 1 + columns) % columns) * (int)pitchOld + row;
      newBoard[column * pitchNew + row] += board[idx];
   }
}
__global__ void checkUnder(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew)
{
   int row, column;

   prepareIndexexPitches(&row, &column, &pitchOld, &pitchNew);

   if (row < rows && column < columns)
   {
      int idx = ((column + 1) % columns) * pitchOld + row;
      newBoard[column * pitchNew + row] += board[idx];
   }
}
__global__ void checkRight(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew)
{
   int row, column;

   prepareIndexexPitches(&row, &column, &pitchOld, &pitchNew);

   if (row < rows && column < columns)
   {
      int idx = column * pitchOld + ((row + 1) % rows);
      newBoard[column * pitchNew + row] += board[idx];
   }
}
__global__ void checkLeft(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew)
{
   int row, column;

   prepareIndexexPitches(&row, &column, &pitchOld, &pitchNew);

   if (row < rows && column < columns)
   {
      int idx = column * pitchOld + (((row - 1) + rows) % rows);
      newBoard[column * pitchNew + row] += board[idx];
   }
}
__global__ void checkRightUnder(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew)
{
   int row, column;

   prepareIndexexPitches(&row, &column, &pitchOld, &pitchNew);

   if (row < rows && column < columns)
   {
      int idx = ((column + 1) % columns) * pitchOld + ((row + 1) % rows);
      newBoard[column * pitchNew + row] += board[idx];
   }
}
__global__ void checkLeftUnder(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew)
{
   int row, column;

   prepareIndexexPitches(&row, &column, &pitchOld, &pitchNew);

   if (row < rows && column < columns)
   {
      int idx = ((column + 1) % columns) * pitchOld + ((row - 1 + rows) % rows);
      newBoard[column * pitchNew + row] += board[idx];
   }
}
__global__ void checkRightAbove(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew)
{
   int row, column;

   prepareIndexexPitches(&row, &column, &pitchOld, &pitchNew);

   if (row < rows && column < columns)
   {
      int idx = ((column - 1 + columns) % columns) * pitchOld + ((row + 1) % rows);
      newBoard[column * pitchNew + row] += board[idx];
   }
}
__global__ void checkLeftAbove(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew)
{
   int row, column;

   prepareIndexexPitches(&row, &column, &pitchOld, &pitchNew);

   if (row < rows && column < columns)
   {
      int idx = ((column - 1 + columns) % columns) * pitchOld + ((row - 1 + rows) % rows);
      newBoard[column * pitchNew + row] += board[idx];
   }
}


__global__ void  determineNextState(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew)
{
   //getting coordintates of the thread
   int x = (blockIdx.x * blockDim.x) + threadIdx.x;
   int y = (blockIdx.y * blockDim.y) + threadIdx.y;

   //adjusting the pitch
   size_t pitchOldAdjusted = pitchOld / sizeof(int);
   size_t pitchNewAdjusted = pitchNew / sizeof(int);

   if (x < rows && y < columns)
   {
      int idxNew = y * pitchNewAdjusted + x;
      int idxOld = y * pitchOldAdjusted + x;

      //remebering the old state
      int state = board[idxOld];

      int output = DEAD;

      //checking if any alive condition is met
      if (state == ALIVE)
      {
         if ((newBoard[idxNew] == 2 || newBoard[idxNew] == 3))
         {
            output = ALIVE;
         }
      }
      else
      {
         if (newBoard[idxNew] == 3)
         {
            output = ALIVE;
         }
      }

      newBoard[idxNew] = output;
   }
}

//void __host__ TaskParallelManager::sendToCuda(int *oldBoard, int *newBoard, size_t rows, size_t columns)
__host__ void sendToCuda(int *oldBoard, int *newBoard, size_t rows, size_t columns)
{
   //CUDA pointers
   int *d_oldBoard;
   int *d_newBoard;

   size_t pitchOld;
   size_t pitchNew;

   cudaMallocPitch((void **)&d_oldBoard, (size_t *)&pitchOld, (size_t)columns * sizeof(int), (size_t)rows);
   cudaMallocPitch((void **)&d_newBoard, (size_t *)&pitchNew, (size_t)columns * sizeof(int), (size_t)rows);

   cudaMemcpy2D(d_oldBoard, pitchOld, oldBoard, columns * sizeof(int), columns * sizeof(int), rows, cudaMemcpyHostToDevice);

   dim3 grid(divideAndRound(rows, BLOCKSIZE_X), divideAndRound(columns, BLOCKSIZE_Y));
   dim3 block(BLOCKSIZE_Y, BLOCKSIZE_X);

   //checkAbove << <block, grid, 0 >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   //checkUnder << <block, grid, 1 >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   //checkRight << <block, grid, 2 >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   //checkLeft << <block, grid, 3 >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   //checkLeftAbove << <block, grid, 4 >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   //checkLeftUnder << <block, grid, 5 >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   //checkRightAbove << <block, grid, 6 >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   //checkRightUnder << <block, grid, 7 >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   //cudaDeviceSynchronize();

   //determineNextState << <block, grid >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   cudaDeviceSynchronize();

   cudaMemcpy2D(newBoard, columns * sizeof(int), d_newBoard, pitchNew, columns * sizeof(int), rows, cudaMemcpyDeviceToHost);

   cudaFree(d_oldBoard);
   cudaFree(d_newBoard);
}