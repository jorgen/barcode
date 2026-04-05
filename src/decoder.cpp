#include "decoder.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

namespace bc {

// EAN-13 L-code patterns: {space, bar, space, bar} widths in modules
static constexpr std::array<std::array<int, 4>, 10> L_PATTERNS = {{
    {3, 2, 1, 1}, // 0
    {2, 2, 2, 1}, // 1
    {2, 1, 2, 2}, // 2
    {1, 4, 1, 1}, // 3
    {1, 1, 3, 2}, // 4
    {1, 2, 3, 1}, // 5
    {1, 1, 1, 4}, // 6
    {1, 3, 1, 2}, // 7
    {1, 2, 1, 3}, // 8
    {3, 1, 1, 2}, // 9
}};

// First-digit parity encoding (0 = L, 1 = G)
static constexpr std::array<std::array<int, 6>, 10> PARITY_PATTERNS = {{
    {0, 0, 0, 0, 0, 0}, // 0
    {0, 0, 1, 0, 1, 1}, // 1
    {0, 0, 1, 1, 0, 1}, // 2
    {0, 0, 1, 1, 1, 0}, // 3
    {0, 1, 0, 0, 1, 1}, // 4
    {0, 1, 1, 0, 0, 1}, // 5
    {0, 1, 1, 1, 0, 0}, // 6
    {0, 1, 0, 1, 0, 1}, // 7
    {0, 1, 0, 1, 1, 0}, // 8
    {0, 1, 1, 0, 1, 0}, // 9
}};

// Match a set of 4 normalized widths against digit patterns.
// Returns {digit, is_g_code, error} or {-1, false, inf} if no match.
struct DigitMatch {
    int digit = -1;
    bool is_g = false;
    float error = 1e9f;
};

static DigitMatch match_digit(const float* widths, float module_width, bool is_right_half) {
    // Normalize widths to module counts
    std::array<float, 4> normalized;
    for (int i = 0; i < 4; ++i) {
        normalized[i] = widths[i] / module_width;
    }

    DigitMatch best;

    for (int d = 0; d < 10; ++d) {
        // Try L-code
        float err_l = 0.0f;
        for (int i = 0; i < 4; ++i) {
            float diff = normalized[i] - L_PATTERNS[d][i];
            err_l += diff * diff;
        }

        if (!is_right_half && err_l < best.error) {
            best = {d, false, err_l};
        }

        // Try G-code (L reversed)
        float err_g = 0.0f;
        for (int i = 0; i < 4; ++i) {
            float diff = normalized[i] - L_PATTERNS[d][3 - i];
            err_g += diff * diff;
        }

        if (!is_right_half && err_g < best.error) {
            best = {d, true, err_g};
        }

        // Try R-code (same pattern structure as L, but bars and spaces are swapped
        // in the actual signal — since we measure widths the same way, R-codes
        // have the same width pattern as L-codes but start with a bar instead of space)
        if (is_right_half && err_l < best.error) {
            best = {d, false, err_l};
        }
    }

    return best;
}

std::vector<Edge> detect_edges(const Scanline& signal, float threshold) {
    if (signal.size() < 2) return {};

    // Auto-threshold: midpoint between min and max
    if (threshold == 0.0f) {
        auto [min_it, max_it] = std::minmax_element(signal.begin(), signal.end());
        threshold = (*min_it + *max_it) * 0.5f;
    }

    std::vector<Edge> edges;
    bool prev_above = signal[0] >= threshold;

    for (size_t i = 1; i < signal.size(); ++i) {
        bool curr_above = signal[i] >= threshold;
        if (curr_above != prev_above) {
            // Linear interpolation of crossing position
            float t = (threshold - signal[i - 1]) / (signal[i] - signal[i - 1]);
            float pos = (i - 1) + t;
            edges.push_back({pos, curr_above}); // rising = going above threshold (dark to light)
        }
        prev_above = curr_above;
    }

    return edges;
}

std::vector<Edge> detect_edges_gradient(const Scanline& signal, float min_gradient) {
    if (signal.size() < 3) return {};

    const int N = static_cast<int>(signal.size());

    // Compute first derivative
    std::vector<float> deriv(N - 1);
    for (int i = 0; i < N - 1; ++i) {
        deriv[i] = signal[i + 1] - signal[i];
    }

    // Auto min_gradient: use median absolute derivative
    if (min_gradient == 0.0f) {
        std::vector<float> abs_deriv(deriv.size());
        for (size_t i = 0; i < deriv.size(); ++i) {
            abs_deriv[i] = std::abs(deriv[i]);
        }
        std::sort(abs_deriv.begin(), abs_deriv.end());
        float median = abs_deriv[abs_deriv.size() / 2];
        // Use 2x median as threshold — filters flat regions while keeping real edges
        min_gradient = std::max(median * 2.0f, 1.0f);
    }

    // Find local extrema in derivative that exceed the threshold
    std::vector<Edge> edges;
    for (int i = 1; i < static_cast<int>(deriv.size()) - 1; ++i) {
        bool is_max = deriv[i] > deriv[i - 1] && deriv[i] > deriv[i + 1];
        bool is_min = deriv[i] < deriv[i - 1] && deriv[i] < deriv[i + 1];

        if (!is_max && !is_min) continue;
        if (std::abs(deriv[i]) < min_gradient) continue;

        // Sub-pixel interpolation via parabolic fit on derivative peak
        float a = deriv[i - 1];
        float b = deriv[i];
        float c = deriv[i + 1];
        float offset = 0.5f * (a - c) / (a - 2.0f * b + c);

        // Edge position: derivative index i is between signal[i] and signal[i+1]
        // So the edge center is at signal index i + 0.5 + offset
        float pos = static_cast<float>(i) + 0.5f + offset;

        bool rising = deriv[i] > 0; // positive derivative = value increasing = dark-to-light
        edges.push_back({pos, rising});
    }

    return edges;
}

float otsu_threshold(const Scanline& signal) {
    if (signal.empty()) return 0.0f;

    // Build histogram (256 bins for [0, 255] signal range)
    constexpr int BINS = 256;
    std::array<int, BINS> hist{};
    for (float v : signal) {
        int bin = std::clamp(static_cast<int>(v), 0, BINS - 1);
        hist[bin]++;
    }

    int total = static_cast<int>(signal.size());
    float sum_all = 0.0f;
    for (int i = 0; i < BINS; ++i) {
        sum_all += static_cast<float>(i) * hist[i];
    }

    float sum_bg = 0.0f;
    int weight_bg = 0;
    float max_variance = 0.0f;
    int best_threshold = 0;

    for (int t = 0; t < BINS; ++t) {
        weight_bg += hist[t];
        if (weight_bg == 0) continue;

        int weight_fg = total - weight_bg;
        if (weight_fg == 0) break;

        sum_bg += static_cast<float>(t) * hist[t];
        float mean_bg = sum_bg / weight_bg;
        float mean_fg = (sum_all - sum_bg) / weight_fg;

        float diff = mean_bg - mean_fg;
        float variance = static_cast<float>(weight_bg) * static_cast<float>(weight_fg) * diff * diff;

        if (variance > max_variance) {
            max_variance = variance;
            best_threshold = t;
        }
    }

    return static_cast<float>(best_threshold);
}

std::vector<float> measure_widths(const std::vector<Edge>& edges) {
    std::vector<float> widths;
    if (edges.size() < 2) return widths;

    widths.reserve(edges.size() - 1);
    for (size_t i = 1; i < edges.size(); ++i) {
        widths.push_back(edges[i].position - edges[i - 1].position);
    }

    return widths;
}

DecodeResult decode_ean13(const std::vector<float>& widths) {
    // EAN-13 structure in terms of width measurements:
    // Guard bars create edges, digits create edges.
    // We need at least: 3 (left guard) + 6*4 (left digits) + 5 (center) + 6*4 (right digits) + 3 (right guard)
    // = 3 + 24 + 5 + 24 + 3 = 59 width measurements
    // But widths are between edges, so we need 59 widths = 60 edges.
    // Actually: guards produce edges too. Let's count differently.
    //
    // The full barcode has 95 modules. The widths array should represent
    // the alternating bar/space widths across the entire barcode.
    // Start guard (bar-space-bar): 3 widths (1-1-1 modules)
    // 6 left digits: 6 * 4 = 24 widths
    // Center guard (space-bar-space-bar-space): 5 widths (1-1-1-1-1 modules)
    // 6 right digits: 6 * 4 = 24 widths
    // End guard (bar-space-bar): 3 widths (1-1-1 modules)
    // Total: 3 + 24 + 5 + 24 + 3 = 59 widths

    if (widths.size() < 59) {
        return {false, "", 0.0f, "EAN-13"};
    }

    // Estimate module width from total width / 95 modules
    float total_width = 0.0f;
    for (size_t i = 0; i < 59; ++i) {
        total_width += widths[i];
    }
    float module_width = total_width / 95.0f;

    // Verify guard patterns
    auto check_guard = [&](int offset, int expected_modules, int count) -> float {
        float err = 0.0f;
        for (int i = 0; i < count; ++i) {
            float normalized = widths[offset + i] / module_width;
            err += (normalized - expected_modules) * (normalized - expected_modules);
        }
        return err;
    };

    // Left guard: 3 bars/spaces of 1 module each
    float guard_err = check_guard(0, 1, 3);
    // Center guard: 5 bars/spaces of 1 module each
    guard_err += check_guard(27, 1, 5);
    // Right guard: 3 bars/spaces of 1 module each
    guard_err += check_guard(56, 1, 3);

    if (guard_err > 5.0f) {
        return {false, "", 0.0f, "EAN-13"};
    }

    // Decode left 6 digits (starting at width index 3)
    std::array<int, 13> digits{};
    std::array<int, 6> parity{};
    float total_error = guard_err;

    for (int i = 0; i < 6; ++i) {
        auto match = match_digit(&widths[3 + i * 4], module_width, false);
        if (match.digit < 0) {
            return {false, "", 0.0f, "EAN-13"};
        }
        digits[i + 1] = match.digit;
        parity[i] = match.is_g ? 1 : 0;
        total_error += match.error;
    }

    // Decode right 6 digits (starting at width index 32)
    for (int i = 0; i < 6; ++i) {
        auto match = match_digit(&widths[32 + i * 4], module_width, true);
        if (match.digit < 0) {
            return {false, "", 0.0f, "EAN-13"};
        }
        digits[i + 7] = match.digit;
        total_error += match.error;
    }

    // Determine first digit from parity pattern
    int first_digit = -1;
    for (int d = 0; d < 10; ++d) {
        bool match = true;
        for (int i = 0; i < 6; ++i) {
            if (PARITY_PATTERNS[d][i] != parity[i]) {
                match = false;
                break;
            }
        }
        if (match) {
            first_digit = d;
            break;
        }
    }

    if (first_digit < 0) {
        return {false, "", 0.0f, "EAN-13"};
    }
    digits[0] = first_digit;

    // Verify check digit
    int check_sum = 0;
    for (int i = 0; i < 12; ++i) {
        check_sum += digits[i] * (i % 2 == 0 ? 1 : 3);
    }
    int expected_check = (10 - (check_sum % 10)) % 10;

    if (digits[12] != expected_check) {
        return {false, "", 0.0f, "EAN-13"};
    }

    // Build result string
    std::string text;
    text.reserve(13);
    for (int d : digits) {
        text += static_cast<char>('0' + d);
    }

    // Confidence: inverse of total matching error, normalized
    float confidence = std::max(0.0f, 1.0f - total_error / 20.0f);

    return {true, text, confidence, "EAN-13"};
}

DecodeResult decode_scanline(const Scanline& signal, EdgeMethod method) {
    std::vector<Edge> edges;
    switch (method) {
    case EdgeMethod::Threshold:
        edges = detect_edges(signal);
        break;
    case EdgeMethod::Gradient:
        edges = detect_edges_gradient(signal);
        break;
    }

    if (edges.empty()) {
        return {false, "", 0.0f, ""};
    }

    auto widths = measure_widths(edges);
    return decode_ean13(widths);
}

} // namespace bc
