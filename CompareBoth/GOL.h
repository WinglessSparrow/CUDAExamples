#include <iostream>
#include <chrono>
#include <thread>
#include "../AdditionalFunctionallity/Timer.h"
#include "../AdditionalFunctionallity/2DManipulator.h"
#include "KernelManager.cuh"

using namespace std;

class GameOfLife
{
public:
   Timer ExecuteGOL(int rows, size_t columns, size_t numRuns, KernelManager *manager);
};
