/* Common.cpp */

#include "Common.hpp"

// Global channel instances used by the application
Channel channel1;
Channel channel2;

// Control flags for stopping acquisition and program
std::atomic<bool> stop_acquisition(false);
std::atomic<bool> stop_program(false);
