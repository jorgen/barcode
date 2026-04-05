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
    Gaussian,       // Smooth rolloff using Gaussian envelope in frequency domain
    Wiener,         // Noise-adaptive filter: preserves strong components, attenuates weak
    WienerDeconv,   // Wiener deconvolution: inverts Gaussian blur PSF with regularization
    HighBoost,      // High-frequency emphasis: amplifies detail relative to DC
};

struct FilterParams {
    FilterType type = FilterType::LowPass;
    float cutoff = 0.5f;      // For LowPass: fraction of coefficients to keep (0-1)
    float threshold = 0.0f;   // For Hard/Soft threshold: magnitude cutoff
    int band_low = 0;         // For BandPass: first coefficient to keep
    int band_high = 0;        // For BandPass: last coefficient to keep
    float sigma = 3.0f;       // For Gaussian: spatial-domain sigma (larger = more smoothing)
    float noise_power = 10.0f; // For Wiener: estimated noise power (larger = more smoothing)
    float blur_sigma = 1.0f;   // For WienerDeconv: Gaussian PSF sigma in pixels
    float noise_ratio = 0.01f; // For WienerDeconv: regularization (noise / signal power)
    float boost = 1.0f;        // For HighBoost: amplification factor (0-5)
};

// Apply filter in frequency domain
std::vector<float> apply_filter(const std::vector<float>& coefficients, const FilterParams& params);

// Convenience: DCT-II -> filter -> DCT-III
Scanline dct_filter(const Scanline& signal, const FilterParams& params);

// Power spectrum |X_k|^2 for analysis
std::vector<float> power_spectrum(const std::vector<float>& coefficients);

// Estimate blur sigma from power spectrum rolloff.
// Returns 0 if the signal appears unblurred.
float estimate_blur_sigma(const std::vector<float>& coefficients);

} // namespace bc
