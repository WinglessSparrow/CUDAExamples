#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#define DEAD 0
#define ALIVE 1

class KernelManager
{
public:
   virtual void sendToCuda(int *oldBoard, int *newBoard, size_t rows, size_t columns) = 0;
   int divideAndRound(int numberElements, int blockSize);
};
