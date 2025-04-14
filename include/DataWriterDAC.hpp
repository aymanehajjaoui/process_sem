/* DataWriterDAC.hpp */

#pragma once

#include "Common.hpp"
#include "DAC.hpp"

// Sends acquired data from the DAC queue to the specified Red Pitaya analog output channel.
// This function continuously reads from the queue and updates the DAC voltage output.
void write_data_dac(Channel &channel, rp_channel_t rp_channel);
