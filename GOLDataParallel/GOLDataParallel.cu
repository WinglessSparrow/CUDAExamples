#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>

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

#define COLLUMNS 5
#define ROWS 5
#define AMM_RUNS 4
#define ALIVE 1
#define DEAD 1

#define BLOCKSIZE_X 64
#define BLOCKSIZE_Y 64

#define MOD(x, xSize) ((x - 1) % xSize + xSize) % xSize

void displayGame(int board[COLLUMNS][ROWS], int xSize, int ySize);

__global__ void  determineNextState(int *board, int *newBoard, int xSize, int ySize, size_t pitchOld, size_t pitchNew)
{
   //getting threads
   int x = (blockIdx.x * blockDim.x) + threadIdx.x;
   int y = (blockIdx.y * blockDim.y) + threadIdx.y;

   if (x < xSize && y < ySize)
   {
      printf("Old Board X: %d Y: %d, Is: %d\n", x, y, *((int *)((char *)(board + y * pitchOld)) + x));
      printf("New Board X: %d Y: %d, Is: %d\n", x, y, *((int *)((char *)(newBoard + y * pitchNew)) + x));
      //x * xSize + y + pitch is the way of mapping 2d array on 1d plane
      int idxNew = x * xSize + y + pitchNew;
      int idxOld = x * xSize + y + pitchOld;
      int state = board[idxOld];

      int output = DEAD;

      //checking if any alive condition is met
      if (state == ALIVE)
      {
         if ((newBoard[idxNew] == 2 || newBoard[idxNew] == 3))
         {
            output = ALIVE;
         }
      }
      else
      {
         if (newBoard[idxNew] == 3)
         {
            output = ALIVE;
         }
      }

      newBoard[idxNew] = output;
   }
}

__global__ void numberAliveAround(int *board, int *newBoard, int xSize, int ySize, size_t pitchOld, size_t pitchNew)
{
   //calculating the thread we are on
   int x = (blockIdx.x * blockDim.x) + threadIdx.x;
   int y = (blockIdx.y * blockDim.y) + threadIdx.y;

   if (x < xSize && y < ySize)
   {
      //printf("X: %d Y: %d, Is: %d\n", x, y, board[x * xSize + y + pitchOld]);
      //printf("Old: X: %d Y: %d, Is: %d\n", x, y, *((int *)((char *)(board + y * pitchOld)) + x));
      printf("Old: X: %d Y: %d, Is: %d\n", x, y, board[y * xSize + x]);

      int outputNumber = 0;
      int idx = 0, xMod = 0, yMod = 0;

      //represents a MOD operator, because % operator ist not quite the same
      //((tidX - 1) % xSize + xSize) % xSize;
      //navigatin in the 1d projection
      //x * xSize + y + pitch

      //right
      xMod = (x + 1) % xSize;
      idx = xMod * xSize + y + pitchOld;
      outputNumber += board[idx];

      //left
      xMod = ((x - 1) % xSize + xSize) % xSize;
      idx = xMod * xSize + y + pitchOld;
      outputNumber += board[idx];

      //down
      yMod = ((y + 1) % ySize + ySize) % ySize;
      idx = x * xSize + yMod + pitchOld;
      outputNumber += board[idx];

      //over
      yMod = ((y - 1) % ySize + ySize) % ySize;
      idx = x * xSize + yMod + pitchOld;
      outputNumber += board[idx];

      //right down corner
      xMod = ((x + 1) % xSize + xSize) % xSize;
      yMod = ((y + 1) % ySize + ySize) % ySize;
      idx = xMod * xSize + yMod + pitchOld;
      outputNumber += board[idx];

      //left down corner
      xMod = ((x - 1) % xSize + xSize) % xSize;
      yMod = ((y + 1) % ySize + ySize) % ySize;
      idx = xMod * xSize + yMod + pitchOld;
      outputNumber += board[idx];

      //right upper corner
      xMod = ((x + 1) % xSize + xSize) % xSize;
      yMod = ((y - 1) % ySize + ySize) % ySize;
      idx = xMod * xSize + yMod + pitchOld;
      outputNumber += board[idx];

      //left upper corner
      xMod = ((x - 1) % xSize + xSize) % xSize;
      yMod = ((y - 1) % ySize + ySize) % ySize;
      idx = xMod * xSize + yMod + pitchOld;
      outputNumber += board[idx];

      newBoard[x * xSize + y + pitchNew] = outputNumber;
   }
}

int divideAndRound(int numberElements, int blockSize)
{
   return ((numberElements % blockSize) != 0) ? (numberElements / blockSize + 1) : (numberElements / blockSize);
}

void SendToCUDA(int oldBoard[COLLUMNS][ROWS], int newBoard[COLLUMNS][ROWS])
{
   //CUDA pointers
   int *d_oldBoard;
   int *d_newBoard;

   size_t pitchOld;
   size_t pitchNew;

   cudaMallocPitch(&d_oldBoard, &pitchOld, COLLUMNS * sizeof(int), ROWS);
   cudaMallocPitch(&d_newBoard, &pitchNew, COLLUMNS * sizeof(int), ROWS);

   cudaMemcpy2D(d_oldBoard, pitchOld, oldBoard, COLLUMNS * sizeof(int), COLLUMNS * sizeof(int), ROWS, cudaMemcpyHostToDevice);

   dim3 grid(divideAndRound(COLLUMNS, BLOCKSIZE_X), divideAndRound(ROWS, BLOCKSIZE_Y));
   dim3 block(BLOCKSIZE_Y, BLOCKSIZE_X);

   printf("counting \n");
   numberAliveAround << <block, grid >> > (d_oldBoard, d_newBoard, COLLUMNS, ROWS, pitchOld, pitchNew);
   cudaDeviceSynchronize();
   printf("determining \n");
   determineNextState << <block, grid >> > (d_oldBoard, d_newBoard, COLLUMNS, ROWS, pitchOld, pitchNew);
   cudaDeviceSynchronize();

   cudaMemcpy2D(newBoard, COLLUMNS * sizeof(int), d_newBoard, pitchNew, COLLUMNS * sizeof(int), ROWS, cudaMemcpyDeviceToHost);

   cudaFree(d_oldBoard);
   cudaFree(d_newBoard);
}

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
   //auto newBoard = mallocFlatt2DArray(COLLUMNS, ROWS);
   //auto oldBoard = mallocFlatt2DArray(COLLUMNS, ROWS);

   int oldBoard[COLLUMNS][ROWS] =
   {
      {0, 0, 0, 0, 0},
      {0, 0, 1, 0, 0},
      {0, 0, 1, 0, 0},
      {0, 0, 1, 0, 0},
      {0, 0, 0, 0, 0}
   };
   int newBoard[COLLUMNS][ROWS] = { 0 };

   int runnsDone = 0;

   Timer timer;

   //filling the starting configuration at random
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_int_distribution<> dis(0, 1);
   //for (int i = 0; i < COLLUMNS; i++)
   //{
   //   for (int j = 0; j < ROWS; j++)
   //   {
   //      oldBoard[i][j] = dis(gen);
   //   }
   //}

   //the main game loop
   while (runnsDone < AMM_RUNS)
   {
      displayGame(oldBoard, COLLUMNS, ROWS);
      timer.addTimeStart();

      SendToCUDA(oldBoard, newBoard);

      timer.addTimeFinish();
      //coppy new state to old board
      memcpy(oldBoard, newBoard, COLLUMNS * ROWS * sizeof(int));

      //clear console
      //system("cls");

      runnsDone++;
   }

   displayGame(newBoard, COLLUMNS, ROWS);

   cout << "average time per Run Millis: " << timer.calcTimes().count() << endl;
   cout << "average time per Run Nano: " << timer.calcTimesNano().count() << endl;
   cout << "end" << endl;
}

void displayGame(int board[COLLUMNS][ROWS], int xSize, int ySize)
{
   cout << endl;
   //cout << "========================================================================================================================" << endl;

   for (int i = 0; i < xSize; i++)
   {
      for (int k = 0; k < ySize; k++)
      {
         cout << ((board[i][k]) ? " * " : " _ ");
      }
      cout << "|" << endl;
   }
   cout << endl;
   //cout << "========================================================================================================================" << endl;
}
