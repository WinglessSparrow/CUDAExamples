// MatrixMultiplication.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <random>

#define COLUMNS 3
#define ROWS 3

using namespace std;

void fillMatrixRandom(int *matrix, int xSize, int ySize)
{
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_int_distribution<> dis(0, 15);

   for (int i = 0; i < xSize; i++)
   {
      for (int j = 0; j < ySize; j++)
      {
         matrix[i * ySize + j] = dis(gen);
      }
   }
}

void fillMatrixZero(int *matrix, int xSize, int ySize)
{

   for (int i = 0; i < xSize; i++)
   {
      for (int j = 0; j < ySize; j++)
      {
         matrix[i * ySize + j] = 0;
      }
   }
}

int half(const int i)
{
   if (!i) return 0;
   int s = 1, v = i;
   if (i < 0) s = -1, v = -i;

   return (v >> 1) * s;
}

int main()
{
   int *matrixA = new int[COLUMNS * ROWS];
   int *matrixB = new int[COLUMNS * ROWS];
   int *matrixC = new int[COLUMNS * ROWS];

   fillMatrixRandom(matrixA, ROWS, COLUMNS);
   fillMatrixRandom(matrixB, ROWS, COLUMNS);
   fillMatrixZero(matrixC, ROWS, COLUMNS);

   for (int i = 0; i < ROWS; i++)
   {
      for (int j = 0; j < COLUMNS; j++)
      {
         for (int k = 0; k < ROWS; k++)
         {
            matrixC[j * COLUMNS + i] += matrixA[k * COLUMNS + i] * matrixB[j * COLUMNS + k];
         }
      }
   }

   for (int i = 0; i < ROWS; i++)
   {
      for (int j = 0; j < COLUMNS; j++)
      {
         cout << matrixA[j * COLUMNS + i] << " ";
      }

      if (half) cout << " + ";
      else cout << "   ";

      for (int j = 0; j < COLUMNS; j++)
      {
         cout << matrixB[j * COLUMNS + i] << " ";
      }

      if (half) cout << " = ";
      else cout << "   ";

      for (int j = 0; j < COLUMNS; j++)
      {
         cout << matrixC[j * COLUMNS + i] << " ";
      }

      cout << endl;
   }
}