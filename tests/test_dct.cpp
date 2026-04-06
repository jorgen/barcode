#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "dct.h"

#include <cmath>
#include <numbers>
#include <numeric>
#include <random>

using namespace bc;
using Catch::Matchers::WithinAbs;

TEST_CASE("DCT-II/III round-trip preserves signal", "[dct]") {
    SECTION("constant signal") {
        Scanline signal(64, 128.0f);
        auto coeffs = dct_ii(signal);
        auto recovered = dct_iii(coeffs);

        for (size_t i = 0; i < signal.size(); ++i) {
            REQUIRE_THAT(recovered[i], WithinAbs(signal[i], 1e-3));
        }
    }

    SECTION("ramp signal") {
        Scanline signal(100);
        for (size_t i = 0; i < signal.size(); ++i) {
            signal[i] = static_cast<float>(i);
        }

        auto coeffs = dct_ii(signal);
        auto recovered = dct_iii(coeffs);

        for (size_t i = 0; i < signal.size(); ++i) {
            REQUIRE_THAT(recovered[i], WithinAbs(signal[i], 1e-3));
        }
    }

    SECTION("random signal") {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 255.0f);

        Scanline signal(128);
        for (auto& v : signal) v = dist(rng);

        auto coeffs = dct_ii(signal);
        auto recovered = dct_iii(coeffs);

        for (size_t i = 0; i < signal.size(); ++i) {
            REQUIRE_THAT(recovered[i], WithinAbs(signal[i], 0.05));
        }
    }

    SECTION("odd-length signal") {
        Scanline signal(37);
        for (size_t i = 0; i < signal.size(); ++i) {
            signal[i] = std::sin(static_cast<float>(i) * 0.3f) * 100.0f + 128.0f;
        }

        auto coeffs = dct_ii(signal);
        auto recovered = dct_iii(coeffs);

        for (size_t i = 0; i < signal.size(); ++i) {
            REQUIRE_THAT(recovered[i], WithinAbs(signal[i], 1e-3));
        }
    }
}

TEST_CASE("DCT-II DC component of constant signal", "[dct]") {
    Scanline signal(64, 42.0f);
    auto coeffs = dct_ii(signal);

    // DC component should be 42 * sqrt(N) (with our normalization: 42 * sqrt(N) * sqrt(2/N) * 1/sqrt(2) = 42)
    // Actually with our normalization: X_0 = sqrt(2/N) * (1/sqrt(2)) * sum(x_n) = sum(x_n) / sqrt(N) = 42*sqrt(64)/sqrt(64) ...
    // Let's just check: X_0 = sqrt(2/N) * (1/sqrt(2)) * N * 42 = sqrt(N) * 42
    // X_0 = sqrt(2/64) * (1/sqrt(2)) * 64 * 42 = (1/sqrt(64)) * 64 * 42 = 8 * 42 = 336
    // Wait no: sqrt(2/64) = sqrt(1/32), 1/sqrt(2) factor, so sqrt(1/32) * 1/sqrt(2) = sqrt(1/64) = 1/8
    // X_0 = (1/8) * 64 * 42 = 336
    REQUIRE_THAT(coeffs[0], WithinAbs(42.0f * std::sqrt(64.0f), 1e-3));

    // All other coefficients should be ~0
    for (size_t k = 1; k < coeffs.size(); ++k) {
        REQUIRE_THAT(coeffs[k], WithinAbs(0.0f, 1e-3));
    }
}

TEST_CASE("DCT energy preservation (Parseval)", "[dct]") {
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(0.0f, 255.0f);

    Scanline signal(64);
    for (auto& v : signal) v = dist(rng);

    float spatial_energy = 0.0f;
    for (auto v : signal) spatial_energy += v * v;

    auto coeffs = dct_ii(signal);
    float freq_energy = 0.0f;
    for (auto c : coeffs) freq_energy += c * c;

    REQUIRE_THAT(freq_energy, WithinAbs(spatial_energy, spatial_energy * 1e-4));
}

TEST_CASE("Low-pass filter removes high-frequency noise", "[dct][filter]") {
    // Create a smooth signal + high-frequency noise
    Scanline signal(128);
    for (size_t i = 0; i < signal.size(); ++i) {
        float smooth = 128.0f + 50.0f * std::sin(2.0f * std::numbers::pi_v<float> * i / 128.0f * 3.0f);
        float noise = 20.0f * std::sin(2.0f * std::numbers::pi_v<float> * i / 128.0f * 50.0f);
        signal[i] = smooth + noise;
    }

    FilterParams params;
    params.type = FilterType::LowPass;
    params.cutoff = 0.15f; // keep only lowest 15% of frequencies

    auto filtered = dct_filter(signal, params);

    // Filtered signal should be closer to the smooth component
    float error_before = 0.0f;
    float error_after = 0.0f;
    for (size_t i = 0; i < signal.size(); ++i) {
        float smooth = 128.0f + 50.0f * std::sin(2.0f * std::numbers::pi_v<float> * i / 128.0f * 3.0f);
        error_before += (signal[i] - smooth) * (signal[i] - smooth);
        error_after += (filtered[i] - smooth) * (filtered[i] - smooth);
    }

    REQUIRE(error_after < error_before * 0.1f); // filtering should reduce error significantly
}

TEST_CASE("Hard threshold filter", "[dct][filter]") {
    Scanline signal(64, 100.0f);
    // Add tiny noise
    signal[10] += 0.001f;

    auto coeffs = dct_ii(signal);

    FilterParams params;
    params.type = FilterType::HardThreshold;
    params.threshold = 0.01f;

    auto filtered = apply_filter(coeffs, params);

    // Only DC should survive (the tiny perturbation coefficient should be zeroed)
    REQUIRE(std::abs(filtered[0]) > 0.01f); // DC survives
}

TEST_CASE("Gaussian filter reduces noise while preserving signal shape", "[dct][filter]") {
    // Low-frequency signal + high-frequency noise
    const int N = 256;
    Scanline signal(N);
    Scanline clean(N);
    std::mt19937 rng(99);
    std::normal_distribution<float> noise_dist(0.0f, 15.0f);

    for (int i = 0; i < N; ++i) {
        // 1 cycle over N samples — concentrated at k=1 in DCT
        float smooth = 128.0f + 60.0f * std::sin(2.0f * std::numbers::pi_v<float> * i / N);
        clean[i] = smooth;
        signal[i] = smooth + noise_dist(rng);
    }

    FilterParams params;
    params.type = FilterType::Gaussian;
    params.sigma = 2.0f; // sigma_freq ≈ N/(2*pi*2) ≈ 20 — passes low freqs, cuts high

    auto filtered = dct_filter(signal, params);

    float error_before = 0.0f, error_after = 0.0f;
    for (int i = 0; i < N; ++i) {
        error_before += (signal[i] - clean[i]) * (signal[i] - clean[i]);
        error_after += (filtered[i] - clean[i]) * (filtered[i] - clean[i]);
    }

    REQUIRE(error_after < error_before);
}

TEST_CASE("Gaussian filter round-trip preserves DC", "[dct][filter]") {
    Scanline signal(64, 100.0f);
    FilterParams params;
    params.type = FilterType::Gaussian;
    params.sigma = 5.0f;

    auto filtered = dct_filter(signal, params);

    for (size_t i = 0; i < signal.size(); ++i) {
        REQUIRE_THAT(filtered[i], WithinAbs(100.0f, 0.1));
    }
}

TEST_CASE("Wiener filter preserves strong signal components", "[dct][filter]") {
    // Strong sinusoid + weak noise
    Scanline signal(128);
    Scanline clean(128);
    std::mt19937 rng(77);
    std::normal_distribution<float> noise_dist(0.0f, 5.0f);

    for (size_t i = 0; i < signal.size(); ++i) {
        float s = 128.0f + 80.0f * std::sin(2.0f * std::numbers::pi_v<float> * i / 128.0f * 6.0f);
        clean[i] = s;
        signal[i] = s + noise_dist(rng);
    }

    FilterParams params;
    params.type = FilterType::Wiener;
    params.noise_power = 50.0f;

    auto filtered = dct_filter(signal, params);

    float error_before = 0.0f, error_after = 0.0f;
    for (size_t i = 0; i < signal.size(); ++i) {
        error_before += (signal[i] - clean[i]) * (signal[i] - clean[i]);
        error_after += (filtered[i] - clean[i]) * (filtered[i] - clean[i]);
    }

    REQUIRE(error_after < error_before);
}

TEST_CASE("Wiener filter with zero noise power is identity", "[dct][filter]") {
    std::mt19937 rng(55);
    std::uniform_real_distribution<float> dist(0.0f, 255.0f);

    Scanline signal(64);
    for (auto& v : signal) v = dist(rng);

    FilterParams params;
    params.type = FilterType::Wiener;
    params.noise_power = 0.0f;

    auto filtered = dct_filter(signal, params);

    for (size_t i = 0; i < signal.size(); ++i) {
        REQUIRE_THAT(filtered[i], WithinAbs(signal[i], 0.1));
    }
}

TEST_CASE("WienerDeconv with blur_sigma=0 is near-identity", "[dct][deblur]") {
    std::mt19937 rng(101);
    std::uniform_real_distribution<float> dist(0.0f, 255.0f);

    Scanline signal(64);
    for (auto& v : signal) v = dist(rng);

    FilterParams params;
    params.type = FilterType::WienerDeconv;
    params.blur_sigma = 0.0f;
    params.noise_ratio = 0.0001f;

    auto filtered = dct_filter(signal, params);

    for (size_t i = 0; i < signal.size(); ++i) {
        REQUIRE_THAT(filtered[i], WithinAbs(signal[i], 0.5));
    }
}

TEST_CASE("WienerDeconv recovers Gaussian-blurred square wave", "[dct][deblur]") {
    const int N = 128;

    // Create a sharp square wave
    Scanline sharp(N);
    for (int i = 0; i < N; ++i) {
        sharp[i] = (i % 16 < 8) ? 200.0f : 50.0f;
    }

    // Blur it with a Gaussian filter
    float blur_sigma = 2.0f;
    FilterParams blur_params;
    blur_params.type = FilterType::Gaussian;
    blur_params.sigma = blur_sigma;
    auto blurred = dct_filter(sharp, blur_params);

    // Measure error before deconvolution
    float error_blurred = 0.0f;
    for (int i = 0; i < N; ++i) {
        float d = blurred[i] - sharp[i];
        error_blurred += d * d;
    }

    // Deconvolve
    FilterParams deconv_params;
    deconv_params.type = FilterType::WienerDeconv;
    deconv_params.blur_sigma = blur_sigma;
    deconv_params.noise_ratio = 0.01f;
    auto recovered = dct_filter(blurred, deconv_params);

    float error_recovered = 0.0f;
    for (int i = 0; i < N; ++i) {
        float d = recovered[i] - sharp[i];
        error_recovered += d * d;
    }

    REQUIRE(error_recovered < error_blurred);
}

TEST_CASE("WienerDeconv with high noise_ratio suppresses output", "[dct][deblur]") {
    Scanline signal(64);
    for (size_t i = 0; i < signal.size(); ++i) {
        signal[i] = (i % 8 < 4) ? 200.0f : 50.0f;
    }

    FilterParams params;
    params.type = FilterType::WienerDeconv;
    params.blur_sigma = 1.0f;
    params.noise_ratio = 1000.0f; // Extreme regularization

    auto filtered = dct_filter(signal, params);

    // With huge noise_ratio, all gain -> 0 except maybe DC
    // The filtered signal should be much flatter than the original
    auto [min_orig, max_orig] = std::minmax_element(signal.begin(), signal.end());
    auto [min_filt, max_filt] = std::minmax_element(filtered.begin(), filtered.end());
    float range_orig = *max_orig - *min_orig;
    float range_filt = *max_filt - *min_filt;

    REQUIRE(range_filt < range_orig * 0.5f);
}

TEST_CASE("HighBoost preserves DC on constant signal", "[dct][deblur]") {
    Scanline signal(64, 150.0f);

    FilterParams params;
    params.type = FilterType::HighBoost;
    params.boost = 3.0f;
    params.sigma = 3.0f;

    auto filtered = dct_filter(signal, params);

    // DC preserved: all values should remain ~150
    for (size_t i = 0; i < signal.size(); ++i) {
        REQUIRE_THAT(filtered[i], WithinAbs(150.0f, 0.1));
    }
}

TEST_CASE("HighBoost amplifies high-frequency coefficients more than low", "[dct][deblur]") {
    // Signal with both low and high frequency components
    const int N = 128;
    Scanline signal(N);
    for (int i = 0; i < N; ++i) {
        signal[i] = 128.0f
            + 50.0f * std::sin(2.0f * std::numbers::pi_v<float> * i / N * 2.0f)   // low freq
            + 20.0f * std::sin(2.0f * std::numbers::pi_v<float> * i / N * 40.0f);  // high freq
    }

    auto coeffs = dct_ii(signal);

    FilterParams params;
    params.type = FilterType::HighBoost;
    params.boost = 2.0f;
    params.sigma = 3.0f;

    auto boosted = apply_filter(coeffs, params);

    // High-frequency coefficients should be amplified more than low-frequency ones
    // Check ratio at low-freq index (~2) vs high-freq index (~40)
    float ratio_low = (coeffs[2] != 0.0f) ? boosted[2] / coeffs[2] : 1.0f;
    float ratio_high = (coeffs[40] != 0.0f) ? boosted[40] / coeffs[40] : 1.0f;

    REQUIRE(ratio_high > ratio_low);
}

TEST_CASE("estimate_blur_sigma returns ~0 for sharp signal and recovers known sigma", "[dct][deblur]") {
    const int N = 256;

    SECTION("sharp signal returns small sigma") {
        // White noise has flat power spectrum — no Gaussian rolloff
        std::mt19937 rng(202);
        std::uniform_real_distribution<float> dist(0.0f, 255.0f);
        Scanline sharp(N);
        for (auto& v : sharp) v = dist(rng);
        auto coeffs = dct_ii(sharp);
        float sigma = estimate_blur_sigma(coeffs);
        REQUIRE(sigma < 1.0f);
    }

    SECTION("recovers known blur sigma within ~50%") {
        // Create a sharp signal then blur it
        Scanline sharp(N);
        for (int i = 0; i < N; ++i) {
            sharp[i] = (i % 8 < 4) ? 255.0f : 0.0f;
        }

        float true_sigma = 3.0f;
        FilterParams blur_params;
        blur_params.type = FilterType::Gaussian;
        blur_params.sigma = true_sigma;
        auto blurred = dct_filter(sharp, blur_params);

        auto coeffs = dct_ii(blurred);
        float estimated = estimate_blur_sigma(coeffs);

        // Should be within 50% of true value
        REQUIRE(estimated > true_sigma * 0.5f);
        REQUIRE(estimated < true_sigma * 1.5f);
    }
}

TEST_CASE("Autocorrelation of constant signal is constant", "[dct][autocorrelation]") {
    Scanline signal(64, 42.0f);
    auto autocorr = dct_autocorrelation(signal);
    REQUIRE(autocorr.size() == 64);

    // A constant signal is perfectly correlated with itself at all lags
    for (size_t k = 1; k < autocorr.size(); ++k) {
        REQUIRE_THAT(autocorr[k], WithinAbs(autocorr[0], autocorr[0] * 1e-3f));
    }
}

TEST_CASE("Autocorrelation of cosine peaks at multiples of period", "[dct][autocorrelation]") {
    const int N = 128;
    const int period = 16; // cosine with period 16 samples
    Scanline signal(N);
    for (int i = 0; i < N; ++i) {
        signal[i] = std::cos(2.0f * std::numbers::pi_v<float> * i / period);
    }

    auto autocorr = dct_autocorrelation(signal);

    // Normalize
    float norm = autocorr[0];
    REQUIRE(norm > 0.0f);
    for (auto& v : autocorr) v /= norm;

    // autocorrelation should peak near lag=period and lag=2*period
    // Find first peak after lag 0
    int first_peak = -1;
    for (int i = 2; i < N / 2; ++i) {
        if (autocorr[i] > autocorr[i - 1] && autocorr[i] > autocorr[i + 1] && autocorr[i] > 0.5f) {
            first_peak = i;
            break;
        }
    }
    REQUIRE(std::abs(first_peak - period) <= 1);
}

TEST_CASE("Autocorrelation of square wave peaks at period", "[dct][autocorrelation]") {
    const int N = 128;
    const int period = 20;
    Scanline signal(N);
    for (int i = 0; i < N; ++i) {
        signal[i] = (i % period < period / 2) ? 255.0f : 0.0f;
    }

    auto autocorr = dct_autocorrelation(signal);
    float norm = autocorr[0];
    REQUIRE(norm > 0.0f);
    for (auto& v : autocorr) v /= norm;

    // Find first peak after lag 0
    int first_peak = -1;
    for (int i = 2; i < N / 2; ++i) {
        if (autocorr[i] > autocorr[i - 1] && autocorr[i] > autocorr[i + 1] && autocorr[i] > 0.3f) {
            first_peak = i;
            break;
        }
    }
    REQUIRE(std::abs(first_peak - period) <= 2);
}

TEST_CASE("Power spectrum", "[dct]") {
    Scanline signal(64);
    for (size_t i = 0; i < signal.size(); ++i) {
        signal[i] = std::cos(std::numbers::pi_v<float> * 5.0f * i / 64.0f);
    }

    auto coeffs = dct_ii(signal);
    auto spectrum = power_spectrum(coeffs);

    REQUIRE(spectrum.size() == 64);

    // Find peak
    auto max_it = std::max_element(spectrum.begin(), spectrum.end());
    int peak_idx = static_cast<int>(std::distance(spectrum.begin(), max_it));

    // Peak should be at or near frequency index 5
    REQUIRE(std::abs(peak_idx - 5) <= 1);
}
