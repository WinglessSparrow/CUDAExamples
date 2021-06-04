#pragma once

#include <thread>

#include "DataParallel.cuh"
#include "DataParallelNoOverlap.cuh"
#include "TaskParallel.cuh"

#include "../AdditionalFunctionallity/Timer.h"
#include "../AdditionalFunctionallity/2DManipulator.h"


class Driver
{
public:
   Timer runTest(size_t rows, size_t cols, int numRuns, TestBase *test);
};

