#include "2DManipulator.h"
using namespace std;

void ProjectedManipulator::displayGame(int *board, size_t xSize, size_t ySize)
{
   for (int i = 0; i < xSize * 3; i++)cout << "=";
   cout << endl;

   for (int i = 0; i < ySize; i++)
   {
      cout << "|";
      for (int k = 0; k < xSize; k++)
      {
         cout << ((board[i * ySize + k]) ? " * " : "   ");
      }
      cout << "|" << endl;
   }

   for (int i = 0; i < xSize * 3; i++)cout << "=";
   cout << endl;
}

void ProjectedManipulator::fillProjected2DArrayRandom(int *board, size_t xSize, size_t ySize, int from, int to)
{
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_int_distribution<> dis(from, to);

   for (int i = 0; i < xSize; i++)
   {
      for (int j = 0; j < ySize; j++)
      {
         board[i * ySize + j] = dis(gen);
      }
   }
}

int ProjectedManipulator::half(const int i, int max)
{
   if (!i) return 0;

   return (max / 2) == i;
}

void ProjectedManipulator::displayMatricess(int rows, int columns, int *matrixA, int *matrixB, int *matrixC)
{
   for (int i = 0; i < rows; i++)
   {
      for (int j = 0; j < columns; j++)
      {
         cout << matrixA[j * columns + i] << " ";
      }

      if (half(i, rows)) cout << " *\t";
      else cout << " \t";

      for (int j = 0; j < columns; j++)
      {
         cout << matrixB[j * columns + i] << " ";
      }

      if (half(i, rows)) cout << " =\t";
      else cout << " \t";

      for (int j = 0; j < columns; j++)
      {
         cout << matrixC[j * columns + i] << " ";
      }

      cout << endl;
   }
}

