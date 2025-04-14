/* DAC.hpp */

#pragma once

#include "Common.hpp"
#include <type_traits>

// Initializes DAC configuration (sets waveform, enables output, etc.)
void initialize_DAC();

// Converts output values from model to voltage level acceptable by Red Pitaya DAC
template <typename T>
float OutputToVoltage(T value)
{
    if constexpr (std::is_same_v<T, int16_t>) {
        return static_cast<float>(value) / 8192.0f;  // Normalize 16-bit signed to [-1, 1]
    } else if constexpr (std::is_same_v<T, int8_t>) {
        return static_cast<float>(value) / 128.0f;   // Normalize 8-bit signed to [-1, 1]
    } else if constexpr (std::is_same_v<T, float>) {
        return value;  // Already in correct format
    } else {
        return static_cast<float>(value); // Fallback conversion
    }
}
