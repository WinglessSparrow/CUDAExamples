#include "ParallelCPU.h"
#include <thread>

using namespace std;

void ParallelCPU::calculateData(int *matrixA, int *matrixB, int *matrixC, int *oldBoard, int *newBoard, size_t rows, size_t cols)
{
   gol_args argsGOL = {oldBoard, newBoard, rows, cols};
   matrixMul_args argsMatr = { matrixA, matrixB, matrixC, rows, cols };

   //starting the threads
   //thread t1(&ParallelCPU::game_of_life, oldBoard, newBoard, rows, cols);
   thread t1([&](ParallelCPU* parallel));
   //thread t2(&ParallelCPU::matrix_multiplication, matrixA, matrixB, matrixC, rows, cols);
   thread t2(&ParallelCPU::matrix_multiplication, matrixA, matrixB, matrixC, rows, cols);

   //waiting for them to finish
   t1.join();
   t2.join();
}

void ParallelCPU::matrix_multiplication(matrixMul_args args)
{
   for (int i = 0; i < args.rows; i++)
   {
      for (int j = 0; j < args.cols; j++)
      {
         for (int k = 0; k < args.rows; k++)
         {
            args.matrixC[j * args.rows + i] += args.matrixA[k * args.rows + i] * args.matrixB[j * args.rows + k];
         }
      }
   }
}

void ParallelCPU::game_of_life(gol_args args)
{
   for (int k = 0; k < args.cols; k++)
   {
      for (int j = 0; j < args.rows; j++)
      {
         int ALIVECellsAround = numberALIVEAround(args.oldBoard, args.cols, args.rows, k, j);
         args.newBoard[j * args.rows + k] = determineNextState(ALIVECellsAround, args.oldBoard[j * args.rows + k]);
      }
   }
}

int ParallelCPU::determineNextState(int numberALIVECellsArround, int state)
{
   int outputState = DEAD;

   switch (state)
   {
   case ALIVE:
      if ((numberALIVECellsArround == 2 || numberALIVECellsArround == 3))
      {
         outputState = ALIVE;
      }
      break;
   case DEAD:
      if ((numberALIVECellsArround == 3))
      {
         outputState = ALIVE;
      }
      break;
   }


   return outputState;
}

int ParallelCPU::numberALIVEAround(int *board, int xSize, int ySize, int xCell, int yCell)
{
   int outputNumber = 0;
   int x = 0, y = 0;

   x = (xCell + 1) % xSize;
   y = yCell;
   outputNumber += board[y * xSize + x];
   x = ((xCell - 1) % xSize + xSize) % xSize;
   y = yCell;
   outputNumber += board[y * xSize + x];
   x = xCell;
   y = ((yCell + 1) % ySize + ySize) % ySize;
   outputNumber += board[y * xSize + x];
   x = xCell;
   y = ((yCell - 1) % ySize + ySize) % ySize;
   outputNumber += board[y * xSize + x];
   x = ((xCell + 1) % xSize + xSize) % xSize;
   y = ((yCell + 1) % ySize + ySize) % ySize;
   outputNumber += board[y * xSize + x];
   x = ((xCell - 1) % xSize + xSize) % xSize;
   y = ((yCell - 1) % ySize + ySize) % ySize;
   outputNumber += board[y * xSize + x];
   x = ((xCell + 1) % xSize + xSize) % xSize;
   y = ((yCell - 1) % ySize + ySize) % ySize;
   outputNumber += board[y * xSize + x];
   x = ((xCell - 1) % xSize + xSize) % xSize;
   y = ((yCell + 1) % ySize + ySize) % ySize;
   outputNumber += board[y * xSize + x];

   return outputNumber;
}