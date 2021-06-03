#pragma once
#define DEAD 0
#define ALIVE 1

class TestBase
{
public:
   virtual void calculateData(int *matrixA, int *matrixB, int *matrixC, int *oldBoard, int *newBoard, size_t rows, size_t cols) = 0;
};
