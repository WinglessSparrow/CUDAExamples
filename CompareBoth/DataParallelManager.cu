#include "DataParallelManager.cuh"

void DataParallelManager::SendToCuda(int *oldBoard, int *newBoard, size_t rows, size_t columns)
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

   numberAliveAround << <block, grid >> > (d_oldBoard, d_newBoard, columns, rows, pitchOld, pitchNew);
   cudaDeviceSynchronize();
   determineNextState << <block, grid >> > (d_oldBoard, d_newBoard, columns, rows, pitchOld, pitchNew);
   cudaDeviceSynchronize();

   cudaMemcpy2D(newBoard, columns * sizeof(int), d_newBoard, pitchNew, columns * sizeof(int), rows, cudaMemcpyDeviceToHost);

   cudaFree(d_oldBoard);
   cudaFree(d_newBoard);
}



__global__ void DataParallelManager::numberAliveAround(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew)
{
   //calculating the thread we are on
   int row = (blockIdx.x * blockDim.x) + threadIdx.x;
   int column = (blockIdx.y * blockDim.y) + threadIdx.y;

   //adjusting pitch, because it's the ammount of bytes and not integer array width
   int pitchOldAdjusted = (int)pitchOld / sizeof(int);
   int pitchNewAdjusted = (int)pitchNew / sizeof(int);

   if (row < rows && column < columns)
   {

      int outputNumber = 0;
      int idx = 0, xMod = 0, yMod = 0;

      //over
      yMod = (column - 1 + columns) % columns;
      idx = yMod * pitchOldAdjusted + row;
      outputNumber += board[idx];

      //under
      yMod = (column + 1) % columns;
      idx = yMod * pitchOldAdjusted + row;
      outputNumber += board[idx];

      //right
      xMod = (row + 1) % rows;
      idx = column * pitchOldAdjusted + xMod;
      outputNumber += board[idx];

      //left
      xMod = ((row - 1) + rows) % rows;
      idx = column * pitchOldAdjusted + xMod;
      outputNumber += board[idx];

      //right bottom corner
      xMod = (row + 1) % rows;
      yMod = (column + 1) % columns;
      idx = yMod * pitchOldAdjusted + xMod;
      outputNumber += board[idx];

      //left bottom corner
      xMod = (row - 1 + rows) % rows;
      yMod = (column + 1) % columns;
      idx = yMod * pitchOldAdjusted + xMod;
      outputNumber += board[idx];

      //right upper corner
      xMod = (row + 1) % rows;
      yMod = (column - 1 + columns) % columns;
      idx = yMod * pitchOldAdjusted + xMod;
      outputNumber += board[idx];

      //left upper corner
      xMod = (row - 1 + rows) % rows;
      yMod = (column - 1 + columns) % columns;
      idx = yMod * pitchOldAdjusted + xMod;
      outputNumber += board[idx];

      newBoard[column * pitchNewAdjusted + row] = outputNumber;
   }
}

__global__ void  DataParallelManager::determineNextState(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew)
{
   //getting coordintates of the thread
   int x = (blockIdx.x * blockDim.x) + threadIdx.x;
   int y = (blockIdx.y * blockDim.y) + threadIdx.y;

   //adjusting the pitch
   size_t pitchOldAdjusted = pitchOld / sizeof(int);
   size_t pitchNewAdjusted = pitchNew / sizeof(int);

   if (x < rows && y < columns)
   {
      size_t idxNew = y * pitchNewAdjusted + x;
      size_t idxOld = y * pitchOldAdjusted + x;

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