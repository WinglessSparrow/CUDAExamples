#pragma once

#include <iostream>
#include <chrono>
#include <thread>
#include "../AdditionalFunctionallity/Timer.h"
#include "../AdditionalFunctionallity/2DManipulator.h"
#include "KernelManager.cuh"
#include "TestBase.h"

class DriverCode
{
public:
   Timer executeTest(size_t rows, size_t columns, size_t numRuns, TestBase* test);
};

