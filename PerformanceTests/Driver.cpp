#include "Driver.h"

#include "../AdditionalFunctionallity/Timer.cpp"
#include "../AdditionalFunctionallity/2DManipulator.cpp"

using namespace std;

Timer Driver::runTest(size_t rows, size_t cols, int numRuns, TestBase *test)
{
   bool display = cols < 30 && rows < 30;
   display = false;

   int runnsDone = 0;

   //allocating memory
   int *oldBoard = new int[rows * cols];
   int *newBoard = new int[rows * cols];
   *newBoard = { 0 };

   int *matrixA = new int[rows * cols];
   int *matrixB = new int[rows * cols];
   int *matrixC = new int[rows * cols];

   Timer timer;

   ProjectedManipulator::fillProjected2DArrayRandom(oldBoard, rows, cols, 0, 1);

   //the main testing loop
   while (runnsDone < numRuns)
   {
      //randomize matrix for each run
      ProjectedManipulator::fillProjected2DArrayRandom(matrixA, rows, cols, 0, 30);
      ProjectedManipulator::fillProjected2DArrayRandom(matrixB, rows, cols, 0, 30);
      ProjectedManipulator::fillProjected2DArrayRandom(matrixC, rows, cols, 0, 0);

      if (display)
      {
         system("cls");
         ProjectedManipulator::displayGame(oldBoard, cols, rows);
      }


      //main calculation
      timer.addTimeStart();

      test->executeCalculation(matrixA, matrixB, matrixC, oldBoard, newBoard, rows, cols);

      timer.addTimeFinish();

      if (display)
      {
         memcpy(oldBoard, newBoard, cols * rows * sizeof(int));

         std::this_thread::sleep_for(milliseconds(5));
      }

      runnsDone++;
   }

   if (display)
   {
      system("cls");
      ProjectedManipulator::displayGame(oldBoard, cols, rows);
      ProjectedManipulator::displayMatricess(rows, cols, matrixA, matrixB, matrixC);
   }

   //deallocating memory
   delete[] oldBoard;
   delete[] newBoard;
   delete[] matrixA;
   delete[] matrixB;
   delete[] matrixC;

   timer.setName(test->getName());

   return timer;
}
