#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

__global__ void checkAbove(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void checkUnder(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void checkRight(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void checkLeft(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void checkRightUnder(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void checkLeftUnder(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void checkRightAbove(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);
__global__ void checkLeftAbove(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew);