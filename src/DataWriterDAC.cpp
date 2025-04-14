/* DataWriter.cpp */

#include "DataWriterDAC.hpp"
#include <iostream>
#include <type_traits>

// Thread function to send acquired data to DAC
void write_data_dac(Channel &channel, rp_channel_t rp_channel)
{
    try
    {
        while (true)
        {
            // Wait until data is available or interruption occurs
            if (sem_wait(&channel.data_sem_dac) != 0)
            {
                if (errno == EINTR && stop_program.load())
                    break; // Gracefully exit if interrupted by SIGINT
                continue;
            }

            // Exit if program is stopping and no data remains
            if (stop_program.load() && channel.data_queue_dac.empty())
                break;

            // Process all queued data for DAC
            while (!channel.data_queue_dac.empty())
            {
                auto part = channel.data_queue_dac.front();
                channel.data_queue_dac.pop();

                for (size_t k = 0; k < MODEL_INPUT_DIM_0; ++k)
                {
                    float voltage = OutputToVoltage(part->data[k][0]);
                    voltage = std::clamp(voltage, -1.0f, 1.0f);
                    rp_GenAmp(rp_channel, voltage);  // Send to DAC
                }

                channel.counters->write_count_dac.fetch_add(1, std::memory_order_relaxed);
            }

            // Exit if acquisition is done and there's no more data
            if (channel.acquisition_done && channel.data_queue_dac.empty())
                break;
        }

        std::cout << "Data writing on DAC thread on channel "
                  << static_cast<int>(channel.channel_id) + 1 << " exiting..." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in write_data_dac for channel "
                  << static_cast<int>(channel.channel_id) + 1 << ": " << e.what() << std::endl;
    }
}
