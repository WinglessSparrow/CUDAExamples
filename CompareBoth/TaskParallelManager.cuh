#pragma once
#include "KernelManager.cuh"

void sendToCuda(int *oldBoard, int *newBoard, size_t rows, size_t columns);


__global__ void checkAbove(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void checkUnder(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void checkRight(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void checkLeft(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void checkRightUnder(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void checkLeftUnder(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void checkRightAbove(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void checkLeftAbove(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void  determineNextState(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__device__ void prepareIndexexPitches(int *row, int *column, size_t *pitchOld, size_t *pitchNew);