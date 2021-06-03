#include "DriverCode.h"

#define NUM_RUNS 100
#define NUM_EXAMPLES 2
#define COLUMNS 30
#define ROWS 30

using namespace std;

void outputResults(Timer *timers, int runs, int xWidth, int yWidth)
{
   cout << "CUDA Accelerated" << endl;
   cout << "Game Of Life DataParallel > Grid:" << xWidth << " by " << yWidth << " Miliseconds: " << timers[0].calcTimes().count() << " Nanoseconds: " << timers[0].calcTimesNano().count() << endl;
   cout << "Game Of Life TaskParallel > Grid:" << xWidth << " by " << yWidth << " Miliseconds: " << timers[1].calcTimes().count() << " Nanoseconds: " << timers[1].calcTimesNano().count() << endl;
}


int main()
{
   DriverCode driver;

   Timer timers[NUM_EXAMPLES];


   return 0;
}