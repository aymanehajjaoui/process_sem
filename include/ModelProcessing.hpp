/* CNNProcessing.hpp */

#pragma once

#include "Common.hpp"
#include "SystemUtils.hpp"

// CNN model inference for Red Pitaya using CMSIS-NN (enabled in model compilation)
#define WITH_CMSIS_NN 1
#define ARM_MATH_DSP 1
#define ARM_NN_TRUNCATE

// Performs CNN inference on data in the model queue for a given channel
void model_inference(Channel &channel);

// Same as model_inference, but includes sample normalization before inference
void model_inference_mod(Channel &channel);
