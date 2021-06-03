#pragma once
#include "TestBase.h"

struct gol_args
{
   int *oldBoard;
   int *newBoard;
   int rows;
   int cols;
};

struct matrixMul_args
{
   int *matrixA;
   int *matrixB;
   int *matrixC;
   int rows;
   int cols;
};

class ParallelCPU:
   public TestBase
{
public:
   void calculateData(int *matrixA, int *matrixB, int *matrixC, int *oldBoard, int *newBoard, size_t rows, size_t cols);
private:
   int numberALIVEAround(int *board, int xSize, int ySize, int xCell, int yCell);
   int determineNextState(int numberALIVECellsArround, int state);
   void game_of_life(gol_args args);
   void matrix_multiplication(matrixMul_args args);
};

