// MatrixMultiplication.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <random>

#define columns 3
#define rows 3

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

int half(const int i, int max)
{
   if (!i) return 0;

   return (max / 2) == i;
}

int main()
{
   int *matrixA = new int[columns * rows];
   int *matrixB = new int[columns * rows];
   int *matrixC = new int[columns * rows];

   fillMatrixRandom(matrixA, rows, columns);
   fillMatrixRandom(matrixB, rows, columns);
   fillMatrixZero(matrixC, rows, columns);

   for (int i = 0; i < rows; i++)
   {
      for (int j = 0; j < columns; j++)
      {
         for (int k = 0; k < rows; k++)
         {
            matrixC[j * columns + i] += matrixA[k * columns + i] * matrixB[j * columns + k];
         }
      }
   }

   for (int i = 0; i < rows; i++)
   {
      for (int j = 0; j < columns; j++)
      {
         cout << matrixA[j * columns + i] << " ";
      }

      if (half(i, rows)) cout << " +\t";
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