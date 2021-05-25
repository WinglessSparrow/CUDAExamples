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

#define COLUMNS 5
#define ROWS 5
#define AMM_RUNS 100
#define ALIVE 1
#define DEAD 0

#define BLOCKSIZE_X 64
#define BLOCKSIZE_Y 64

void displayGame(int board[ROWS][COLUMNS], int xSize, int ySize);

__global__ void  determineNextState(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew)
{
   //getting threads
   int x = (blockIdx.x * blockDim.x) + threadIdx.x;
   int y = (blockIdx.y * blockDim.y) + threadIdx.y;

   size_t pitchOldAdjusted = pitchOld / sizeof(int);
   size_t pitchNewAdjusted = pitchNew / sizeof(int);

   if (x < rows && y < columns)
   {
      int idxNew = y * pitchNewAdjusted + x;
      int idxOld = y * pitchOldAdjusted + x;

      int state = board[idxOld];

      //printf("New Board X: %d Y: %d, Is: %d\n", x, y, newBoard[idxNew]);

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

__global__ void numberAliveAround(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew)
{
   //calculating the thread we are on
   int row = (blockIdx.x * blockDim.x) + threadIdx.x;
   int column = (blockIdx.y * blockDim.y) + threadIdx.y;

   //adjusting pitch, because it's the ammount of bytes and not integer array width
   size_t pitchOldAdjusted = pitchOld / sizeof(int);
   size_t pitchNewAdjusted = pitchNew / sizeof(int);

   if (row < rows && column < columns)
   {
      //printf("Old: X: %d Y: %d, Is: %d\n", row, column, board[column * pitchOldAdjusted + row]);

      int outputNumber = 0;
      int idx = 0, xMod = 0, yMod = 0;

      //over
      yMod = (column - 1 + columns) % columns;
      idx = yMod * pitchOldAdjusted + row;
      outputNumber += board[idx];

      //under
      yMod = (column + 1) % columns;
      idx = yMod * pitchOldAdjusted + row;
      outputNumber += board[idx];

      //right
      xMod = (row + 1) % rows;
      idx = column * pitchOldAdjusted + xMod;
      outputNumber += board[idx];

      //left
      xMod = ((row - 1) + rows) % rows;
      idx = column * pitchOldAdjusted + xMod;
      outputNumber += board[idx];

      //right bottom corner
      xMod = (row + 1) % rows;
      yMod = (column + 1) % columns;
      idx = yMod * pitchOldAdjusted + xMod;
      outputNumber += board[idx];

      //left bottom corner
      xMod = (row - 1 + rows) % rows;
      yMod = (column + 1) % columns;
      idx = yMod * pitchOldAdjusted + xMod;
      outputNumber += board[idx];

      //right upper corner
      xMod = (row + 1) % rows;
      yMod = (column - 1 + columns) % columns;
      idx = yMod * pitchOldAdjusted + xMod;
      outputNumber += board[idx];

      //left upper corner
      xMod = (row - 1 + rows) % rows;
      yMod = (column - 1 + columns) % columns;
      idx = yMod * pitchOldAdjusted + xMod;
      outputNumber += board[idx];

      newBoard[column * pitchNewAdjusted + row] = outputNumber;
   }
}

int divideAndRound(int numberElements, int blockSize)
{
   return ((numberElements % blockSize) != 0) ? (numberElements / blockSize + 1) : (numberElements / blockSize);
}

void SendToCUDA(int oldBoard[ROWS][COLUMNS], int newBoard[ROWS][COLUMNS])
{
   //CUDA pointers
   int *d_oldBoard;
   int *d_newBoard;

   size_t pitchOld;
   size_t pitchNew;

   cudaMallocPitch((void **)&d_oldBoard, (size_t *)&pitchOld, (size_t)COLUMNS * sizeof(int), (size_t)ROWS);
   cudaMallocPitch((void **)&d_newBoard, (size_t *)&pitchNew, (size_t)COLUMNS * sizeof(int), (size_t)ROWS);

   cudaMemcpy2D(d_oldBoard, pitchOld, oldBoard, COLUMNS * sizeof(int), COLUMNS * sizeof(int), ROWS, cudaMemcpyHostToDevice);

   dim3 grid(divideAndRound(ROWS, BLOCKSIZE_X), divideAndRound(COLUMNS, BLOCKSIZE_Y));
   dim3 block(BLOCKSIZE_Y, BLOCKSIZE_X);

   //printf("counting \n");
   //need to pitch / sizeof(int), because pitch is the ammount of bytes, and I need to navigate an int array
   numberAliveAround << <block, grid >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   //numberAliveAround << <block, grid >> > (d_oldBoard, d_newBoard, ROWS, COLUMNS, pitchOld, pitchNew);
   cudaDeviceSynchronize();
   //printf("determining \n");
   determineNextState << <block, grid >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   //determineNextState << <block, grid >> > (d_oldBoard, d_newBoard, ROWS, COLUMNS, pitchOld, pitchNew);
   cudaDeviceSynchronize();

   cudaMemcpy2D(newBoard, COLUMNS * sizeof(int), d_newBoard, pitchNew, COLUMNS * sizeof(int), ROWS, cudaMemcpyDeviceToHost);

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

   int oldBoard[ROWS][COLUMNS] =
   {
      {0, 0, 0, 0, 0},
      {0, 0, 1, 0, 0},
      {0, 0, 1, 0, 0},
      {0, 0, 1, 0, 0},
      {0, 0, 0, 0, 0}
   };
   int newBoard[ROWS][COLUMNS] = { 0 };

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
      system("cls");
      displayGame(oldBoard, COLUMNS, ROWS);
      timer.addTimeStart();

      SendToCUDA(oldBoard, newBoard);

      timer.addTimeFinish();
      //coppy new state to old board
      memcpy(oldBoard, newBoard, COLUMNS * ROWS * sizeof(int));

      //clear console
      sleep_for(milliseconds(500));

      runnsDone++;
   }

   displayGame(newBoard, COLUMNS, ROWS);

   cout << "average time per Run Millis: " << timer.calcTimes().count() << endl;
   cout << "average time per Run Nano: " << timer.calcTimesNano().count() << endl;
   cout << "end" << endl;
}

void displayGame(int board[ROWS][COLUMNS], int xSize, int ySize)
{
   cout << endl;
   //cout << "========================================================================================================================" << endl;

   for (int i = 0; i < ySize; i++)
   {
      for (int k = 0; k < xSize; k++)
      {
         cout << ((board[i][k]) ? " * " : "   ");
      }
      cout << endl;
   }
   cout << endl;
   //cout << "========================================================================================================================" << endl;
}
