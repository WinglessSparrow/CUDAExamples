#include "Timer.h"

Timer::Timer()
{
}

Timer::~Timer()
{
}

void Timer::addTimeStart()
{
   timeMeasurement temp;
   temp.start = system_clock::now();
   _times.push_back(temp);
}

void Timer::addTimeFinish()
{
   auto &temp = _times.back();
   temp.finish = system_clock::now();
}

milliseconds Timer::calcTimes()
{
   milliseconds timeInMillis = milliseconds::zero();
   for (auto &time : _times)
   {
      timeInMillis += duration_cast<milliseconds>(time.finish - time.start);
   }

   timeInMillis /= _times.size();

   return timeInMillis;
}

nanoseconds Timer::calcTimesNano()
{
   nanoseconds timeInNano = nanoseconds::zero();
   for (auto &time : _times)
   {
      timeInNano += duration_cast<milliseconds>(time.finish - time.start);
   }

   timeInNano /= _times.size();

   return timeInNano;
}

