#include <iostream>
#include <chrono>
#include <thread>
#include "../AdditionalFunctionallity/Timer.h"
#include "../AdditionalFunctionallity/2DManipulator.h"
#include "KernelManager.cuh"
#include "DataParallelManager.cuh"
#include "TaskParallelManager.cuh"

using namespace std;

class GameOfLife
{
public:
   Timer ExecuteGOL(size_t rows, size_t columns, size_t numRuns, void sendToCuda(int *oldBoard, int *newBoard, size_t rows, size_t columns));
};
