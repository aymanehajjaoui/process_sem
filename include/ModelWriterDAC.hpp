/* CNNWriterDAC.hpp */

#pragma once

#include "Common.hpp"
#include "DAC.hpp"

// Writes CNN inference results to the Red Pitaya DAC
void log_results_dac(Channel &channel, rp_channel_t rp_channel);
