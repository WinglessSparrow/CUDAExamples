#include "kernels.cuh"

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


#define COLUMNS 30
#define ROWS 30
#define AMM_RUNS 100
#define ALIVE 1
#define DEAD 0

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32

#if COLUMNS < 40 && ROWS < 40
#define DISPLAY true
#else
#define DISPLAY false
#endif

__global__ void  determineNextState(int *board, int *newBoard, int rows, int columns, size_t pitchOld, size_t pitchNew)
{
   //getting coordintates of the thread
   int x = (blockIdx.x * blockDim.x) + threadIdx.x;
   int y = (blockIdx.y * blockDim.y) + threadIdx.y;

   //adjusting the pitch
   size_t pitchOldAdjusted = pitchOld / sizeof(int);
   size_t pitchNewAdjusted = pitchNew / sizeof(int);

   if (x < rows && y < columns)
   {
      int idxNew = y * pitchNewAdjusted + x;
      int idxOld = y * pitchOldAdjusted + x;

      //remebering the old state
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

int divideAndRound(int numberElements, int blockSize);
void SendToCUDA(int *oldBoard, int *newBoard);
void displayGame(int *board, int xSize, int ySize);
void fillBoardRandom(int *board, int xSize, int ySize);

int main()
{
   int runnsDone = 0;

   //allocating memory
   int *oldBoard = new int[ROWS * COLUMNS];
   int *newBoard = new int[ROWS * COLUMNS];
   *newBoard = { 0 };

   Timer timer;

   fillBoardRandom(oldBoard, ROWS, COLUMNS);

   cout << "Game of Life, Data Parralel on CUDA with " << ROWS << " ROWS and " << COLUMNS << " Columns" << endl;

   //the main game loop
   while (runnsDone < AMM_RUNS)
   {
#if DISPLAY
      //clear console
      system("cls");
      displayGame(oldBoard, COLUMNS, ROWS);
#endif      

      //main calculation
      timer.addTimeStart();

      //SendToCUDA(oldBoard, newBoard);
      SendToCUDA(oldBoard, newBoard);

      timer.addTimeFinish();

#if DISPLAY
      //coppy new state to old board
      memcpy(oldBoard, newBoard, COLUMNS * ROWS * sizeof(int));

      sleep_for(milliseconds(5));
#endif 

      runnsDone++;
   }

#if DISPLAY
   //clear console
   system("cls");
   displayGame(oldBoard, COLUMNS, ROWS);
#endif   

   cout << "average time per Run Millis: " << timer.calcTimes().count() << endl;
   cout << "average time per Run Nano: " << timer.calcTimesNano().count() << endl;
   cout << "end" << endl;
}

void displayGame(int *board, int xSize, int ySize)
{
   cout << endl;

   for (int i = 0; i < ySize; i++)
   {
      for (int k = 0; k < xSize; k++)
      {
         cout << ((board[i * ySize + k]) ? " * " : "   ");
      }
      cout << endl;
   }
   cout << endl;
}

int divideAndRound(int numberElements, int blockSize)
{
   return ((numberElements % blockSize) != 0) ? (numberElements / blockSize + 1) : (numberElements / blockSize);
}

void fillBoardRandom(int *board, int xSize, int ySize)
{
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_int_distribution<> dis(0, 1);

   for (int i = 0; i < xSize; i++)
   {
      for (int j = 0; j < ySize; j++)
      {
         board[i * ySize + j] = dis(gen);
      }
   }
}

void SendToCUDA(int *oldBoard, int *newBoard)
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

   checkAbove << <block, grid, 0 >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   checkUnder << <block, grid, 1 >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   checkRight << <block, grid, 2 >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   checkLeft << <block, grid, 3 >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   checkLeftAbove << <block, grid, 4 >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   checkLeftUnder << <block, grid, 5 >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   checkRightAbove << <block, grid, 6 >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   checkRightUnder << <block, grid, 7 >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   cudaDeviceSynchronize();

   determineNextState << <block, grid >> > (d_oldBoard, d_newBoard, COLUMNS, ROWS, pitchOld, pitchNew);
   cudaDeviceSynchronize();

   cudaMemcpy2D(newBoard, COLUMNS * sizeof(int), d_newBoard, pitchNew, COLUMNS * sizeof(int), ROWS, cudaMemcpyDeviceToHost);

   cudaFree(d_oldBoard);
   cudaFree(d_newBoard);
}