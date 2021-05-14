#include <iostream>
#include <chrono>
#include <thread>
#include "../AdditionalFunctionallity/Timer.cpp"
#include <random>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using std::cout;
using std::endl;
using std::this_thread::sleep_for;
using std::chrono::milliseconds;
using std::copy;

#define X_DIMENSION 5000
#define Y_DIMENSION 5000
#define AMM_RUNS 1

enum CellState
{
   alive = 1,
   dead = 0
};

CellState determineNextState(int numberAliveCellsArround, CellState state);
int numberAliveAround(const CellState board[][Y_DIMENSION], int xSize, int ySize, int xCell, int yCell);
void displayGame(const CellState board[][Y_DIMENSION], int xSize, int ySize);

int main()
{

   auto newBoard = new CellState[X_DIMENSION][Y_DIMENSION];
   auto oldBoard = new CellState[X_DIMENSION][Y_DIMENSION];

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
         oldBoard[i][j] = (CellState)dis(gen);
      }
   }

   //the main game loop
   while (runnsDone < AMM_RUNS)
   {

      //displayGame(oldBoard, X_DIMENSION, Y_DIMENSION);
      timer.addTimeStart();
      //iterating through board and calculating next state
      for (int k = 0; k < X_DIMENSION; k++)
      {
         for (int j = 0; j < Y_DIMENSION; j++)
         {
            int aliveCellsAround = numberAliveAround(oldBoard, X_DIMENSION, Y_DIMENSION, k, j);
            newBoard[k][j] = determineNextState(aliveCellsAround, oldBoard[k][j]);
         }
      }
      timer.addTimeFinish();
      //coppy new state to old board
      memcpy(oldBoard, newBoard, X_DIMENSION * Y_DIMENSION * sizeof(CellState));

      //clear console
      runnsDone++;
      system("cls");
   }

   cout << "average time per Run Millis: " << timer.calcTimes().count() << endl;
   cout << "average time per Run Nano: " << timer.calcTimesNano().count() << endl;
   cout << "end" << endl;
}

CellState determineNextState(int numberAliveCellsArround, CellState state)
{
   CellState outputState = dead;

   switch (state)
   {
   case alive:
      if ((numberAliveCellsArround == 2 || numberAliveCellsArround == 3))
      {
         outputState = alive;
      }
      break;
   case dead:
      if ((numberAliveCellsArround == 3))
      {
         outputState = alive;
      }
      break;
   }


   return outputState;
}

int numberAliveAround(const CellState board[][Y_DIMENSION], int xSize, int ySize, int xCell, int yCell)
{
   int outputNumber = 0;
   int x = 0, y = 0;

   //represents a MOD operator, because % operator ist not quite the same
   //((xCell - 1) % xSize + xSize) % xSize;

   x = (xCell + 1) % xSize;
   y = yCell;
   outputNumber += board[x][y];
   x = ((xCell - 1) % xSize + xSize) % xSize;
   y = yCell;
   outputNumber += board[x][y];
   x = xCell;
   y = ((yCell + 1) % ySize + ySize) % ySize;
   outputNumber += board[x][y];
   x = xCell;
   y = ((yCell - 1) % ySize + ySize) % ySize;
   outputNumber += board[x][y];
   x = ((xCell + 1) % xSize + xSize) % xSize;
   y = ((yCell + 1) % ySize + ySize) % ySize;
   outputNumber += board[x][y];
   x = ((xCell - 1) % xSize + xSize) % xSize;
   y = ((yCell - 1) % ySize + ySize) % ySize;
   outputNumber += board[x][y];
   x = ((xCell + 1) % xSize + xSize) % xSize;
   y = ((yCell - 1) % ySize + ySize) % ySize;
   outputNumber += board[x][y];
   x = ((xCell - 1) % xSize + xSize) % xSize;
   y = ((yCell + 1) % ySize + ySize) % ySize;
   outputNumber += board[x][y];

   return outputNumber;
}

void displayGame(const CellState board[][Y_DIMENSION], int xSize, int ySize)
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
