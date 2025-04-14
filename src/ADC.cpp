/* ADC.cpp */

#include "ADC.hpp"
#include <iostream>

// Initializes Red Pitaya acquisition settings
void initialize_acq()
{
    rp_AcqReset();

    // Enable split trigger mode
    if (rp_AcqSetSplitTrigger(true) != RP_OK)
        std::cerr << "rp_AcqSetSplitTrigger failed!" << std::endl;
    if (rp_AcqSetSplitTriggerPass(true) != RP_OK)
        std::cerr << "rp_AcqSetSplitTriggerPass failed!" << std::endl;

    // Get reserved AXI memory region
    uint32_t g_adc_axi_start, g_adc_axi_size;
    if (rp_AcqAxiGetMemoryRegion(&g_adc_axi_start, &g_adc_axi_size) != RP_OK)
    {
        std::cerr << "rp_AcqAxiGetMemoryRegion failed!" << std::endl;
        exit(-1);
    }

    std::cout << "Reserved memory Start 0x" << std::hex << g_adc_axi_start
              << " Size 0x" << g_adc_axi_size << std::dec << std::endl;

    // Set decimation factor for each channel
    if (rp_AcqAxiSetDecimationFactorCh(RP_CH_1, DECIMATION) != RP_OK ||
        rp_AcqAxiSetDecimationFactorCh(RP_CH_2, DECIMATION) != RP_OK)
    {
        std::cerr << "rp_AcqAxiSetDecimationFactor failed!" << std::endl;
        exit(-1);
    }

    // Print sampling rate
    float sampling_rate;
    if (rp_AcqGetSamplingRateHz(&sampling_rate) == RP_OK)
        std::cout << "Current Sampling Rate: " << sampling_rate << " Hz\n";
    else
        std::cerr << "Failed to get sampling rate\n";

    // Set trigger delay
    if (rp_AcqAxiSetTriggerDelay(RP_CH_1, 0) != RP_OK ||
        rp_AcqAxiSetTriggerDelay(RP_CH_2, 0) != RP_OK)
    {
        std::cerr << "rp_AcqAxiSetTriggerDelay failed!" << std::endl;
        exit(-1);
    }

    // Set buffer memory for both channels
    if (rp_AcqAxiSetBufferSamples(RP_CH_1, g_adc_axi_start, DATA_SIZE) != RP_OK ||
        rp_AcqAxiSetBufferSamples(RP_CH_2, g_adc_axi_start + (g_adc_axi_size / 2), DATA_SIZE) != RP_OK)
    {
        std::cerr << "rp_AcqAxiSetBuffer failed!" << std::endl;
        exit(-1);
    }

    // Enable acquisition hardware
    if (rp_AcqAxiEnable(RP_CH_1, true) != RP_OK ||
        rp_AcqAxiEnable(RP_CH_2, true) != RP_OK)
    {
        std::cerr << "rp_AcqAxiEnable failed!" << std::endl;
        exit(-1);
    }

    // Set trigger conditions
    if (rp_AcqSetTriggerLevel(RP_T_CH_1, 0) != RP_OK ||
        rp_AcqSetTriggerLevel(RP_T_CH_2, 0) != RP_OK ||
        rp_AcqSetTriggerSrcCh(RP_CH_1, RP_TRIG_SRC_CHA_PE) != RP_OK ||
        rp_AcqSetTriggerSrcCh(RP_CH_2, RP_TRIG_SRC_CHB_PE) != RP_OK)
    {
        std::cerr << "rp_AcqSetTrigger setup failed!" << std::endl;
        exit(-1);
    }

    // Start acquisition
    if (rp_AcqStartCh(RP_CH_1) != RP_OK || rp_AcqStartCh(RP_CH_2) != RP_OK)
    {
        std::cerr << "rp_AcqStart failed!" << std::endl;
        exit(-1);
    }
}

// Cleans up Red Pitaya resources after acquisition
void cleanup()
{
    std::cout << "\nReleasing resources\n";
    rp_AcqStopCh(RP_CH_1);
    rp_AcqStopCh(RP_CH_2);
    rp_AcqAxiEnable(RP_CH_1, false);
    rp_AcqAxiEnable(RP_CH_2, false);
    rp_Release();
    std::cout << "Cleanup done." << std::endl;
}
