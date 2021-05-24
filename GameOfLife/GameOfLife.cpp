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

#define COLLUMNS 50
#define ROWS 50
#define AMM_RUNS 10000

enum CellState
{
   alive = 1,
   dead = 0
};

CellState determineNextState(int numberAliveCellsArround, CellState state);
int numberAliveAround(const CellState board[][ROWS], int xSize, int ySize, int xCell, int yCell);
void displayGame(const CellState board[][ROWS], int xSize, int ySize);

int main()
{

   auto newBoard = new CellState[COLLUMNS][ROWS];
   auto oldBoard = new CellState[COLLUMNS][ROWS];

   int runnsDone = 0;

   Timer timer;

   //filling the starting configuration at random
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_int_distribution<> dis(0, 1);
   for (int i = 0; i < COLLUMNS; i++)
   {
      for (int j = 0; j < ROWS; j++)
      {
         oldBoard[i][j] = (CellState)dis(gen);
      }
   }

   //the main game loop
   while (runnsDone < AMM_RUNS)
   {

      displayGame(oldBoard, COLLUMNS, ROWS);
      timer.addTimeStart();
      //iterating through board and calculating next state
      for (int k = 0; k < COLLUMNS; k++)
      {
         for (int j = 0; j < ROWS; j++)
         {
            int aliveCellsAround = numberAliveAround(oldBoard, COLLUMNS, ROWS, k, j);
            newBoard[k][j] = determineNextState(aliveCellsAround, oldBoard[k][j]);
         }
      }
      timer.addTimeFinish();
      //coppy new state to old board
      memcpy(oldBoard, newBoard, COLLUMNS * ROWS * sizeof(CellState));

      //clear console
      sleep_for(milliseconds(5));
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

int numberAliveAround(const CellState board[][ROWS], int xSize, int ySize, int xCell, int yCell)
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

void displayGame(const CellState board[][ROWS], int xSize, int ySize)
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
