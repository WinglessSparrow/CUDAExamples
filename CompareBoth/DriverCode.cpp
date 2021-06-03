#include "DriverCode.h"

using namespace std;

Timer DriverCode::executeTest(size_t rows, size_t columns, size_t numRuns, TestBase *test)
{
   bool display = columns < 40 && rows < 40;

   int runnsDone = 0;

   //allocating memory
   int *oldBoard = new int[rows * columns];
   int *newBoard = new int[rows * columns];
   *newBoard = { 0 };

   int *matrixA = new int[rows * columns];
   int *matrixB = new int[rows * columns];
   int *matrixC = new int[rows * columns];

   Timer timer;

   ProjectedManipulator::fillProjected2DArrayRandom(oldBoard, rows, columns, 0, 1);
   ProjectedManipulator::fillProjected2DArrayRandom(matrixA, rows, columns, 0, 30);
   ProjectedManipulator::fillProjected2DArrayRandom(matrixB, rows, columns, 0, 30);
   ProjectedManipulator::fillProjected2DArrayRandom(matrixC, rows, columns, 0, 0);

   //the main testing loop
   while (runnsDone < numRuns)
   {
      if (display)
      {
         system("cls");
         ProjectedManipulator::displayGame(oldBoard, columns, rows);
      }


      //main calculation
      timer.addTimeStart();

      test->calculateData(matrixA, matrixB, matrixC, oldBoard, newBoard, rows, columns);

      timer.addTimeFinish();

      if (display)
      {
         memcpy(oldBoard, newBoard, columns * rows * sizeof(int));

         std::this_thread::sleep_for(milliseconds(5));
      }

      runnsDone++;
   }

   if (display)
   {
      system("cls");
      ProjectedManipulator::displayGame(oldBoard, columns, rows);
   }

   //deallocating memory
   delete[] oldBoard;
   delete[] newBoard;
   delete[] matrixA;
   delete[] matrixB;
   delete[] matrixC;

   return timer;
}
