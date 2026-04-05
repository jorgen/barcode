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
    case FilterType::Gaussian: {
        // Convert spatial sigma to frequency-domain sigma
        // sigma_freq = N / (2 * pi * sigma_spatial)
        float sigma_freq = static_cast<float>(N) /
                           (2.0f * std::numbers::pi_v<float> * params.sigma);
        float two_sigma_sq = 2.0f * sigma_freq * sigma_freq;
        for (int k = 0; k < N; ++k) {
            float gain = std::exp(-static_cast<float>(k * k) / two_sigma_sq);
            filtered[k] *= gain;
        }
        break;
    }
    case FilterType::Wiener: {
        // Wiener filter: H(k) = |X(k)|^2 / (|X(k)|^2 + noise_power)
        for (int k = 0; k < N; ++k) {
            float power = filtered[k] * filtered[k];
            float gain = power / (power + params.noise_power);
            filtered[k] *= gain;
        }
        break;
    }
    case FilterType::WienerDeconv: {
        // Wiener deconvolution assuming Gaussian PSF
        // H(k) = exp(-pi^2 * blur_sigma^2 * k^2 / (2 * N^2))
        // gain(k) = H(k) / (H(k)^2 + noise_ratio)
        float pi2 = std::numbers::pi_v<float> * std::numbers::pi_v<float>;
        float bs2 = params.blur_sigma * params.blur_sigma;
        float N2 = static_cast<float>(N) * static_cast<float>(N);
        for (int k = 0; k < N; ++k) {
            float kf = static_cast<float>(k);
            float H = std::exp(-pi2 * bs2 * kf * kf / (2.0f * N2));
            float gain = H / (H * H + params.noise_ratio);
            filtered[k] *= gain;
        }
        break;
    }
    case FilterType::HighBoost: {
        // gain(k) = 1 + boost * (1 - exp(-k^2 / (2 * sigma_freq^2)))
        // Reuse sigma for transition width (sigma_freq)
        float sigma_freq = params.sigma;
        float two_sf2 = 2.0f * sigma_freq * sigma_freq;
        for (int k = 0; k < N; ++k) {
            float kf = static_cast<float>(k);
            float gain = 1.0f + params.boost * (1.0f - std::exp(-kf * kf / two_sf2));
            filtered[k] *= gain;
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

float estimate_blur_sigma(const std::vector<float>& coefficients) {
    const int N = static_cast<int>(coefficients.size());
    if (N < 16) return 0.0f;

    // Fit log(P(k)) vs k^2 via linear regression in range [N/8, N/2]
    // For a Gaussian-blurred signal: log(P(k)) ~ const - (pi^2 * sigma^2 / N^2) * k^2
    int k_lo = N / 8;
    int k_hi = N / 2;

    // Compute log power spectrum, skipping zero-power bins
    float sum_x = 0.0f, sum_y = 0.0f, sum_xx = 0.0f, sum_xy = 0.0f;
    int count = 0;
    for (int k = k_lo; k <= k_hi; ++k) {
        float power = coefficients[k] * coefficients[k];
        if (power <= 0.0f) continue;
        float x = static_cast<float>(k) * static_cast<float>(k);
        float y = std::log(power);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
        ++count;
    }

    if (count < 4) return 0.0f;

    float n = static_cast<float>(count);
    float denom = n * sum_xx - sum_x * sum_x;
    if (std::abs(denom) < 1e-12f) return 0.0f;

    float slope = (n * sum_xy - sum_x * sum_y) / denom;

    // slope should be negative for a blurred signal
    // slope = -pi^2 * sigma^2 / N^2  =>  sigma = N * sqrt(-slope) / pi
    if (slope >= 0.0f) return 0.0f;

    float sigma = static_cast<float>(N) * std::sqrt(-slope) / std::numbers::pi_v<float>;
    return sigma;
}

} // namespace bc
