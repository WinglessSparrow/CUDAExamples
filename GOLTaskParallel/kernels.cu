#include "kernels.cuh"


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
      int idx = ((column - 1 + columns) % columns) * pitchOld + row;
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
