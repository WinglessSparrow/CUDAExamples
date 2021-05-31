#pragma once
#include "KernelManager.cuh"

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32

class DataParallelManager:
   public KernelManager
{
public:
   virtual void SendToCuda(int *oldBoard, int *newBoard, size_t rows, size_t columns);
private:
   __global__ void numberAliveAround(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
   __global__ void  determineNextState(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
};

