#pragma once

#include "TestBase.h"

class DataParallelNoOverlap: public TestBase
{
public:
   virtual void executeCalculation(int *matrixA, int *matrixB, int *matrixC, int *oldBoard, int *newBoard, size_t rows, size_t cols);
   virtual string getName();
private:
   int divideAndRound(int numberElements, int blockSize);
};