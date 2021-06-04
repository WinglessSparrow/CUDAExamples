#pragma once

#include <string>

#include "Kernels.cuh"

using std::string;

class TestBase
{
public:
   virtual void executeCalculation(int *matrixA, int *matrixB, int *matrixC, int *oldBoard, int *newBoard, const size_t rows, const size_t cols) = 0;
   virtual string getName() = 0;
};