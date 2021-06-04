#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define DEAD 0
#define ALIVE 1

struct pitchesBoard
{
   size_t pitchOld;
   size_t pitchNew;
};

struct pitchesMatrix
{
   size_t pitchMA;
   size_t pitchMB;
   size_t pitchMC;
};

__global__ void  determineNextState(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void numberAliveAround(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void multiplyMatrix(int *matrixA, int *matrixB, int *matrixC, int rows, int cols, size_t pitchA, size_t pitchB, size_t pitchC);

__global__ void  determineNextStateOffset(int *board, int *newBoard, int rows, int columns, pitchesBoard pitches, int offset);
__global__ void numberAliveAroundOffset(int *board, int *newBoard, int rows, int columns, pitchesBoard pitches, int offset);
__global__ void multiplyMatrixOffset(int *matrixA, int *matrixB, int *matrixC, int rows, int cols, pitchesMatrix pitches, int offset);