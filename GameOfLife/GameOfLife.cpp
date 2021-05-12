#include <iostream>
#include <chrono>
#include <thread>


using std::cout;
using std::endl;
using std::this_thread::sleep_for;
using std::chrono::milliseconds;
using std::copy;

#define X_DIMENSION_TEMPLATE 17
#define Y_DIMENSION_TEMPLATE 17

#define X_DIMENSION 50
#define Y_DIMENSION 50
#define AMM_RUNS 100

enum CellState
{
   alive = 1,
   dead = 0
};

CellState determineNextState(int numberAliveCellsArround, CellState state);
int numberAliveAround(CellState board[][Y_DIMENSION], int xSize, int ySize, int xCell, int yCell);
void displayGame(CellState board[][Y_DIMENSION], int xSize, int ySize);

int main()
{
   /*CellState oldBoard[X_DIMENSION][Y_DIMENSION] = {
      {dead, dead, dead, dead, dead, dead},
      {dead, alive, alive, dead, dead, dead},
      {dead, alive, dead, dead, dead, dead},
      {dead, dead, dead, dead, alive, dead},
      {dead, dead, dead, alive, alive, dead},
      {dead, dead, dead, dead, dead, dead},

   };*/
   /* CellState oldBoard[X_DIMENSION][Y_DIMENSION] = {
       {dead, dead, alive, dead, dead, dead},
       {alive, dead, alive, dead, dead, dead},
       {dead, alive, alive, dead, dead, dead},
       {dead, dead, dead, dead, dead, dead},
       {dead, dead, dead, dead, dead, dead},
       {dead, dead, dead, dead, dead, dead},
    };*/

   CellState startingTemplate[X_DIMENSION_TEMPLATE][Y_DIMENSION_TEMPLATE] = {
   {dead, dead, dead, dead, dead, dead, dead,   dead, dead, dead, dead, dead, dead, dead, dead, dead, dead},
   {dead, dead, dead, dead, dead, dead, dead,   dead, dead, dead, dead, dead, dead, dead, dead, dead, dead},
   {dead, dead, dead, dead, alive, alive, alive,dead, dead, dead, alive, alive, alive, dead, dead, dead, dead},
   {dead, dead, dead, dead, dead, dead, dead, dead,   dead, dead, dead, dead, dead,    dead, dead, dead, dead},
   {dead, dead, alive, dead, dead, dead, dead, alive, dead, alive, dead, dead, dead,   dead, alive, dead, dead},
   {dead, dead, alive, dead, dead, dead, dead, alive, dead, alive, dead, dead, dead,   dead, alive, dead, dead},
   {dead, dead, alive, dead, dead, dead, dead, alive, dead, alive, dead, dead, dead,   dead, alive, dead, dead},
   {dead, dead, dead, dead, alive, alive, alive,dead, dead, dead, alive, alive, alive, dead, dead, dead, dead},
   {dead, dead, dead, dead, dead, dead, dead,   dead, dead, dead, dead, dead, dead, dead, dead, dead, dead},
   {dead, dead, dead, dead, alive, alive, alive,dead, dead, dead, alive, alive, alive, dead, dead, dead, dead},
   {dead, dead, alive, dead, dead, dead, dead, alive, dead, alive, dead, dead, dead,   dead, alive, dead, dead},
   {dead, dead, alive, dead, dead, dead, dead, alive, dead, alive, dead, dead, dead,   dead, alive, dead, dead},
   {dead, dead, alive, dead, dead, dead, dead, alive, dead, alive, dead, dead, dead,   dead, alive, dead, dead},
   {dead, dead, dead, dead, dead, dead, dead, dead,   dead, dead, dead, dead, dead,    dead, dead, dead, dead},
   {dead, dead, dead, dead, alive, alive, alive,dead, dead, dead,  alive, alive, alive, dead, dead, dead, dead},
   {dead, dead, dead, dead, dead, dead, dead, dead,   dead, dead, dead, dead, dead, dead, dead, dead, dead},
   {dead, dead, dead, dead, dead, dead, dead, dead,   dead, dead, dead, dead, dead, dead, dead, dead, dead}
   };
   CellState oldBoard[X_DIMENSION][Y_DIMENSION] = { dead };

   //coppy template into board
   for (int i = 0; i < X_DIMENSION_TEMPLATE; i++)
   {
      for (int j = 0; j < Y_DIMENSION_TEMPLATE; j++)
      {
         oldBoard[i][j] = startingTemplate[i][j];
      }
   }

   CellState newBoard[X_DIMENSION][Y_DIMENSION];

   while (1)
   {
      displayGame(oldBoard, X_DIMENSION, Y_DIMENSION);
      for (int k = 0; k < X_DIMENSION; k++)
      {
         for (int j = 0; j < Y_DIMENSION; j++)
         {
            newBoard[k][j] = determineNextState(numberAliveAround(oldBoard, X_DIMENSION, Y_DIMENSION, k, j), oldBoard[k][j]);
         }
      }
      memcpy(oldBoard, newBoard, X_DIMENSION * Y_DIMENSION * sizeof(CellState));
      sleep_for(milliseconds{ 500 });
      system("cls");
   }


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

int numberAliveAround(CellState board[][Y_DIMENSION], int xSize, int ySize, int xCell, int yCell)
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

void displayGame(CellState board[][Y_DIMENSION], int xSize, int ySize)
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
