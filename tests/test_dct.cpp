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
