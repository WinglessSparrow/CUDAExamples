#pragma once
#include "KernelManager.cuh"


void sendToCuda(int *oldBoard, int *newBoard, size_t rows, size_t columns);


__global__ void numberAliveAround(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void  determineNextState(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);