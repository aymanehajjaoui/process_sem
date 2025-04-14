/* ADC.hpp */

#pragma once

#include "rp.h"
#include "Common.hpp"

// Initializes Red Pitaya acquisition parameters and buffers
void initialize_acq();

// Releases Red Pitaya resources and disables acquisition
void cleanup();
