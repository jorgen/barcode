#include "dct.h"

#include <cmath>
#include <numbers>

namespace bc {

std::vector<float> dct_ii(const Scanline& signal) {
    const int N = static_cast<int>(signal.size());
    std::vector<float> coefficients(N);

    const float scale = std::sqrt(2.0f / N);
    const float s0 = 1.0f / std::sqrt(2.0f); // s(0) = 1/sqrt(2)

    for (int k = 0; k < N; ++k) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            sum += signal[n] * std::cos(std::numbers::pi_v<float> / N * (n + 0.5f) * k);
        }
        coefficients[k] = scale * (k == 0 ? s0 : 1.0f) * sum;
    }

    return coefficients;
}

Scanline dct_iii(const std::vector<float>& coefficients) {
    const int N = static_cast<int>(coefficients.size());
    Scanline signal(N);

    const float scale = std::sqrt(2.0f / N);
    const float s0 = 1.0f / std::sqrt(2.0f);

    for (int n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += (k == 0 ? s0 : 1.0f) * coefficients[k]
                   * std::cos(std::numbers::pi_v<float> / N * (n + 0.5f) * k);
        }
        signal[n] = scale * sum;
    }

    return signal;
}

std::vector<float> apply_filter(const std::vector<float>& coefficients, const FilterParams& params) {
    const int N = static_cast<int>(coefficients.size());
    std::vector<float> filtered = coefficients;

    switch (params.type) {
    case FilterType::LowPass: {
        int cutoff_idx = static_cast<int>(params.cutoff * N);
        for (int k = cutoff_idx; k < N; ++k)
            filtered[k] = 0.0f;
        break;
    }
    case FilterType::HardThreshold: {
        for (int k = 0; k < N; ++k) {
            if (std::abs(filtered[k]) < params.threshold)
                filtered[k] = 0.0f;
        }
        break;
    }
    case FilterType::SoftThreshold: {
        for (int k = 0; k < N; ++k) {
            float val = filtered[k];
            if (std::abs(val) <= params.threshold) {
                filtered[k] = 0.0f;
            } else {
                filtered[k] = std::copysign(std::abs(val) - params.threshold, val);
            }
        }
        break;
    }
    case FilterType::BandPass: {
        for (int k = 0; k < N; ++k) {
            if (k < params.band_low || k > params.band_high)
                filtered[k] = 0.0f;
        }
        break;
    }
    }

    return filtered;
}

Scanline dct_filter(const Scanline& signal, const FilterParams& params) {
    auto coeffs = dct_ii(signal);
    auto filtered = apply_filter(coeffs, params);
    return dct_iii(filtered);
}

std::vector<float> power_spectrum(const std::vector<float>& coefficients) {
    std::vector<float> spectrum(coefficients.size());
    for (size_t k = 0; k < coefficients.size(); ++k) {
        spectrum[k] = coefficients[k] * coefficients[k];
    }
    return spectrum;
}

} // namespace bc
