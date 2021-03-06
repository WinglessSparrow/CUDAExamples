#ifndef TIMER_H
#define TIMER_H

#include <vector>
#include <iostream>
#include <chrono>
#include <string>

using std::vector;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using std::chrono::nanoseconds;
using std::chrono::time_point;
using std::chrono::system_clock;

class Timer
{
public:

   void addTimeStart();
   void addTimeFinish();
   milliseconds calcTimes();
   nanoseconds calcTimesNano();
   void setName(const std::string name);
   std::string getName();

private:
   std::string name;

   struct timeMeasurement
   {
      time_point<system_clock> start;
      time_point<system_clock> finish;
   };

   vector<timeMeasurement> _times;
};

#endif