#include "GOL.cuh"


Timer GameOfLife::ExecuteGOL(size_t rows, size_t columns, size_t numRuns, void sendToCuda(int *oldBoard, int *newBoard, size_t rows, size_t columns))
{
   bool display = columns < 40 && rows < 40;

   int runnsDone = 0;

   //allocating memory
   int *oldBoard = new int[rows * columns];
   int *newBoard = new int[rows * columns];
   *newBoard = { 0 };

   Timer timer;

   ProjectedManipulator::fillProjected2DArrayRandom(oldBoard, rows, columns, 0, 1);
   cout << "Start: Data Parralel Game of Life" << endl;

   //the main game loop
   while (runnsDone < numRuns)
   {
      if (display)
      {
         system("cls");
         ProjectedManipulator::displayGame(oldBoard, columns, rows);
      }


      //main calculation
      timer.addTimeStart();

      sendToCuda(oldBoard, newBoard, rows, columns);

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

   delete[] oldBoard;
   delete[] newBoard;

   cout << "Done with: Data Parralel Game of Life" << endl;

   return timer;
}
