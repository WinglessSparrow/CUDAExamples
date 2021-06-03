#include "KernelManager.cuh"

int divideAndRound(int numberElements, int blockSize)
{
   return ((numberElements % blockSize) != 0) ? (numberElements / blockSize + 1) : (numberElements / blockSize);
}
