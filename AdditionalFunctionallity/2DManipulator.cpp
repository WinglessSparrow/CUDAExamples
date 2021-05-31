#include "2DManipulator.h"
using namespace std;

void ProjectedManipulator::displayGame(int *board, size_t xSize, size_t ySize)
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

void ProjectedManipulator::fillProjected2DArrayRandom(int *board, size_t xSize, size_t ySize)
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
