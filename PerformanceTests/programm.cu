#include <stdio.h>
#include "Driver.h"

#include <vector>

using namespace std;

#define ROWS 20
#define COLS 20
#define NUM_RUNS 1

void printTimers(vector<Timer> timers)
{
   for each (auto t in timers)
   {
      cout << "Test of " << t.getName() << " : Miliseconds: " << t.calcTimes().count() << "; Nanoseconds: " << t.calcTimesNano().count() << endl;
   }
}

__global__ void blankKernel()
{
}

int main()
{
   Driver driver;
   vector<Timer> timers;
   vector<TestBase *> tests;


   tests.push_back(new DataParallel);
   tests.push_back(new DataParallelNoOverlap());
   tests.push_back(new TaskParallel());

   //blank kernel, because first one always starts slower than the rest
   blankKernel << <1, 1 >> > ();
   cudaDeviceSynchronize();

   cout << "Starting testing with " << ROWS * COLS << " elements" << endl;

   for each (auto t in tests)
   {
      cout << "Test of: " << t->getName() << endl;
      timers.push_back(driver.runTest(ROWS, COLS, NUM_RUNS, t));
   }


   printTimers(timers);

   return 0;
}

