#pragma once

#include "types.h"
#include <vector>

namespace bc {

// Forward DCT-II: spatial domain -> frequency domain
std::vector<float> dct_ii(const Scanline& signal);

// Inverse DCT (DCT-III): frequency domain -> spatial domain
Scanline dct_iii(const std::vector<float>& coefficients);

enum class FilterType {
    LowPass,        // Zero out coefficients above cutoff fraction
    HardThreshold,  // Zero out coefficients with magnitude below threshold
    SoftThreshold,  // Shrink coefficients toward zero by threshold amount
    BandPass,       // Keep coefficients in [band_low, band_high] index range
};

struct FilterParams {
    FilterType type = FilterType::LowPass;
    float cutoff = 0.5f;      // For LowPass: fraction of coefficients to keep (0-1)
    float threshold = 0.0f;   // For Hard/Soft threshold: magnitude cutoff
    int band_low = 0;         // For BandPass: first coefficient to keep
    int band_high = 0;        // For BandPass: last coefficient to keep
};

// Apply filter in frequency domain
std::vector<float> apply_filter(const std::vector<float>& coefficients, const FilterParams& params);

// Convenience: DCT-II -> filter -> DCT-III
Scanline dct_filter(const Scanline& signal, const FilterParams& params);

// Power spectrum |X_k|^2 for analysis
std::vector<float> power_spectrum(const std::vector<float>& coefficients);

} // namespace bc
