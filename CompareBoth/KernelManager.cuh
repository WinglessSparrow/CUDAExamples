#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

#define DEAD 0
#define ALIVE 1

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32

int divideAndRound(int numberElements, int blockSize);
