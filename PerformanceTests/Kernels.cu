#include "Kernels.cuh"

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

__global__ void numberAliveAround(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew)
{
   //calculating the thread we are on
   int row = (blockIdx.x * blockDim.x) + threadIdx.x;
   int column = (blockIdx.y * blockDim.y) + threadIdx.y;

   //adjusting pitch, because it's the ammount of bytes and not integer array width
   size_t pitchOldAdjusted = pitchOld / sizeof(int);
   size_t pitchNewAdjusted = pitchNew / sizeof(int);

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

__global__ void multiplyMatrix(int *matrixA, int *matrixB, int *matrixC, int rows, int cols, size_t pitchA, size_t pitchB, size_t pitchC)
{
   int row = (blockIdx.x * blockDim.x) + threadIdx.x;
   int column = (blockIdx.y * blockDim.y) + threadIdx.y;

   //adjusting pitch, because it's the ammount of bytes and not integer array width
   size_t pitchAAdjusted = pitchA / sizeof(int);
   size_t pitchBAdjusted = pitchB / sizeof(int);
   size_t pitchCAdjusted = pitchC / sizeof(int);

   if (row < rows && column < cols)
   {
      for (int i = 0; i < pitchAAdjusted; i++)
      {
         matrixC[column * pitchCAdjusted + row] += matrixA[i * pitchAAdjusted + column] * matrixB[column * pitchBAdjusted + i];
      }
   }
}

__global__ void determineNextStateOffset(int *board, int *newBoard, int rows, int columns, pitchesBoard pitches, int offset)
{
   //getting coordintates of the thread
   //offset because these are neing called in small batches
   int x = offset + (blockIdx.x * blockDim.x) + threadIdx.x;
   int y = (blockIdx.y * blockDim.y) + threadIdx.y;

   //adjusting the pitch
   size_t pitchOldAdjusted = pitches.pitchOld / sizeof(int);
   size_t pitchNewAdjusted = pitches.pitchNew / sizeof(int);

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

__global__ void numberAliveAroundOffset(int *board, int *newBoard, int rows, int columns, pitchesBoard pitches, int offset)
{
   //calculating the thread we are on
   int row = offset + (blockIdx.x * blockDim.x) + threadIdx.x;
   int column = (blockIdx.y * blockDim.y) + threadIdx.y;

   //adjusting pitch, because it's the ammount of bytes and not integer array width
   size_t pitchOldAdjusted = pitches.pitchOld / sizeof(int);
   size_t pitchNewAdjusted = pitches.pitchNew / sizeof(int);

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

__global__ void multiplyMatrixOffset(int *matrixA, int *matrixB, int *matrixC, int rows, int cols, pitchesMatrix pitches, int offset)
{
   int row = offset + (blockIdx.x * blockDim.x) + threadIdx.x;
   int column = (blockIdx.y * blockDim.y) + threadIdx.y;

   //adjusting pitch, because it's the ammount of bytes and not integer array width
   size_t pitchAAdjusted = pitches.pitchMA / sizeof(int);
   size_t pitchBAdjusted = pitches.pitchMB / sizeof(int);
   size_t pitchCAdjusted = pitches.pitchMC / sizeof(int);

   if (row < rows && column < cols)
   {
      for (int i = 0; i < pitchAAdjusted; i++)
      {
         matrixC[column * pitchCAdjusted + row] += matrixA[i * pitchAAdjusted + column] * matrixB[column * pitchBAdjusted + i];
      }
   }
}


