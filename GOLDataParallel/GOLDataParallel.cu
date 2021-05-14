#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <chrono>
#include <thread>
#include "../AdditionalFunctionallity/Timer.cpp"
#include <random>

using std::cout;
using std::endl;
using std::this_thread::sleep_for;
using std::chrono::milliseconds;
using std::copy;

#define X_DIMENSION 40
#define Y_DIMENSION 40
#define AMM_RUNS 1
#define ALIVE 1
#define DEAD 1

__global__ void  determineNextState(int **board, int **newBoard, int ySize, int xSize)
{

   int tidX = (blockIdx.x * blockDim.x) + threadIdx.x;
   int tidY = (blockIdx.x * blockDim.x) + threadIdx.x;

   if (tidX < xSize && tidY < ySize)
   {
      int x = tidX;
      int y = tidY;

      int state = board[x][y];
      board[x][y] = DEAD;
      if (state == ALIVE)
      {
         if ((newBoard[x][y] == 2 || newBoard[x][y] == 3))
         {
            newBoard[x][y] = ALIVE;
         }
      }
      else
      {
         if (newBoard[x][y] == 3)
         {
            board[x][y] = ALIVE;
         }
      }
   }
}

__global__ void numberAliveAround(int **board, int **newBoard, int ySize, int xSize)
{
   //calculating the thread we are on
   int tidX = (blockIdx.x * blockDim.x) + threadIdx.x;
   int tidY = (blockIdx.x * blockDim.x) + threadIdx.x;

   if (tidX < xSize)
   {

      /*tidX /= ySize;
      tidY %= ySize;*/

      int outputNumber = 0;
      int x = 0, y = 0;

      //represents a MOD operator, because % operator ist not quite the same
      //((tidX - 1) % xSize + xSize) % xSize;

      x = (tidX + 1) % xSize;
      y = tidY;
      outputNumber += board[x][y];
      x = ((tidX - 1) % xSize + xSize) % xSize;
      y = tidY;
      outputNumber += board[x][y];
      x = tidX;
      y = ((tidY + 1) % ySize + ySize) % ySize;
      outputNumber += board[x][y];
      x = tidX;
      y = ((tidY - 1) % ySize + ySize) % ySize;
      outputNumber += board[x][y];
      x = ((tidX + 1) % xSize + xSize) % xSize;
      y = ((tidY + 1) % ySize + ySize) % ySize;
      outputNumber += board[x][y];
      x = ((tidX - 1) % xSize + xSize) % xSize;
      y = ((tidY - 1) % ySize + ySize) % ySize;
      outputNumber += board[x][y];
      x = ((tidX + 1) % xSize + xSize) % xSize;
      y = ((tidY - 1) % ySize + ySize) % ySize;
      outputNumber += board[x][y];
      x = ((tidX - 1) % xSize + xSize) % xSize;
      y = ((tidY + 1) % ySize + ySize) % ySize;
      outputNumber += board[x][y];

      newBoard[tidX][tidY] = outputNumber;
   }
}

void SendToCUDA(int **oldBoard, int **newBoard)
{
   //CUDA pointers
   int *gpuOldBoard;
   int *gpuNewBoard;

   cudaMalloc((void **)&gpuOldBoard, sizeof(int) * X_DIMENSION * Y_DIMENSION);
   cudaMalloc((void **)&gpuNewBoard, sizeof(int) * X_DIMENSION * Y_DIMENSION);

   cudaMemcpy(gpuOldBoard, oldBoard[0], sizeof(int) * X_DIMENSION * Y_DIMENSION, cudaMemcpyHostToDevice);

   constexpr int NUM_THREADS = 1 << 10;
   constexpr int NUM_BLOCK = ((X_DIMENSION * Y_DIMENSION) + NUM_THREADS - 1) / NUM_THREADS;

   numberAliveAround << <NUM_BLOCK, NUM_THREADS >> > ((int **)gpuOldBoard, (int **)gpuNewBoard, X_DIMENSION, Y_DIMENSION);
   determineNextState << <NUM_BLOCK, NUM_THREADS >> > ((int **)gpuOldBoard, (int **)gpuNewBoard, X_DIMENSION, Y_DIMENSION);

   cudaMemcpy(newBoard[0], &gpuNewBoard, sizeof(int) * X_DIMENSION * Y_DIMENSION, cudaMemcpyDeviceToHost);

   cudaFree(gpuNewBoard);
   cudaFree(gpuOldBoard);
}

void displayGame(int **board, int xSize, int ySize);

//code gracefully stolen from here:
//http://www.trevorsimonton.com/blog/2016/11/16/transfer-2d-array-memory-to-cuda.html
int **mallocFlatt2DArray(int xSize, int ySize)
{
   //2d array
   int **output = new int *[xSize];
   //point first pointer to flat representation of an array
   output[0] = new int[xSize * ySize];

   //linkin further pointers with gaps of ySize
   for (int i = 1; i < xSize; ++i)
   {
      output[i] = output[i - 1] + ySize;
   }

   return output;
}

int main()
{
   auto newBoard = mallocFlatt2DArray(X_DIMENSION, Y_DIMENSION);
   auto oldBoard = mallocFlatt2DArray(X_DIMENSION, Y_DIMENSION);

   int runnsDone = 0;

   Timer timer;

   //filling the starting configuration at random
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_int_distribution<> dis(0, 1);
   for (int i = 0; i < X_DIMENSION; i++)
   {
      for (int j = 0; j < Y_DIMENSION; j++)
      {
         oldBoard[i][j] = dis(gen);
      }
   }

   //the main game loop
   while (runnsDone < AMM_RUNS)
   {
      displayGame(oldBoard, X_DIMENSION, Y_DIMENSION);
      timer.addTimeStart();

      SendToCUDA(oldBoard, newBoard);

      timer.addTimeFinish();
      //coppy new state to old board
      memcpy(oldBoard, newBoard, X_DIMENSION * Y_DIMENSION * sizeof(int));

      //clear console
      //system("cls");

      runnsDone++;
   }

   displayGame(newBoard, X_DIMENSION, Y_DIMENSION);

   cout << "average time per Run Millis: " << timer.calcTimes().count() << endl;
   cout << "average time per Run Nano: " << timer.calcTimesNano().count() << endl;
   cout << "end" << endl;
}

void displayGame(int **board, int xSize, int ySize)
{
   for (int i = 0; i < xSize; i++)
   {
      for (int k = 0; k < ySize; k++)
      {
         cout << ((board[i][k]) ? " * " : "   ");
      }
      cout << endl;
   }
}
