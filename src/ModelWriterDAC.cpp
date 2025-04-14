/* modelWriterDAC.cpp */

#include "ModelWriterDAC.hpp"
#include <iostream>
#include <semaphore.h>

void log_results_dac(Channel &channel, rp_channel_t rp_channel)
{
    try
    {
        while (true)
        {
            // Wait for new model results for DAC
            if (sem_wait(&channel.result_sem_dac) != 0)
            {
                if (errno == EINTR && stop_program.load())
                    break;
                continue;
            }

            // Exit if stopping and nothing left to process
            if (stop_program.load() && channel.result_buffer_dac.empty())
                break;

            while (!channel.result_buffer_dac.empty())
            {
                const model_result_t &result = channel.result_buffer_dac.front();
                float voltage = std::clamp(OutputToVoltage(result.output[0]), -1.0f, 1.0f);
                rp_GenAmp(rp_channel, voltage);

                channel.result_buffer_dac.pop_front();
                channel.counters->log_count_dac.fetch_add(1, std::memory_order_relaxed);
            }

            if (channel.processing_done && channel.result_buffer_dac.empty())
                break;
        }

        std::cout << "Logging inference results on DAC thread on channel "
                  << static_cast<int>(channel.channel_id) + 1 << " exiting..." << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception in log_results_dac for channel "
                  << static_cast<int>(channel.channel_id) + 1 << ": " << e.what() << std::endl;
    }
}
