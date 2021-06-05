#ifndef MANIPULATOR_H
#define MANIPULATOR_H

#include <iostream>
#include <random>

class ProjectedManipulator
{
public:
   static void displayGame(int *board, size_t xSize, size_t ySize);
   static void fillProjected2DArrayRandom(int *board, size_t xSize, size_t ySize, int from, int to);
   static void displayMatricess(int rows, int columns, int *matrixA, int *matrixB, int *matrixC);
private:
   static int half(const int i, int max);

};

#endif