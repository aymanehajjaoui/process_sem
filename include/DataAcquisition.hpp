/* DataAcquisition.hpp */

#pragma once

#include "ADC.hpp"

// Starts the data acquisition process on the specified Red Pitaya channel.
// Handles trigger detection, buffer reads, and dispatch to queues.
void acquire_data(Channel &channel, rp_channel_t rp_channel);
