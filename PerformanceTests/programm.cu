#include <stdio.h>
#include "Driver.h"

#include <vector>

using namespace std;

#define ROWS 30
#define COLS 30
#define NUM_RUNS 100

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

   //blank kernel, because first one always strats slower than the rest
   blankKernel << <1, 1 >> > ();
   cudaDeviceSynchronize();

   cout << "Starting testing with " << ROWS * COLS << " elements" << endl;

   cout << "Test 1" << endl;
   timers.push_back(driver.runTest(ROWS, COLS, NUM_RUNS, new TaskParallel()));
   cout << "Test 2" << endl;
   //timers.push_back(driver.runTest(ROWS, COLS, NUM_RUNS, new DataParallelNoOverlap()));
   cout << "Test 3" << endl;
   //timers.push_back(driver.runTest(ROWS, COLS, NUM_RUNS, new DataParallel()));

   printTimers(timers);

   return 0;
}

