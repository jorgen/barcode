#include "decoder.h"
#include "dct.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>

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

// --- Correlation-based decoder ---

float estimate_module_width(const Scanline& signal) {
    const int N = static_cast<int>(signal.size());
    if (N < 20) return 0.0f;

    // Mean-subtract to focus on AC structure
    float mean = 0.0f;
    for (float v : signal) mean += v;
    mean /= static_cast<float>(N);

    Scanline centered(N);
    for (int i = 0; i < N; ++i) {
        centered[i] = signal[i] - mean;
    }

    auto autocorr = dct_autocorrelation(centered);

    // Normalize by autocorr[0]
    if (std::abs(autocorr[0]) < 1e-9f) return 0.0f;
    float norm = autocorr[0];
    for (auto& v : autocorr) v /= norm;

    // Find first local minimum after lag 0 (anti-correlation from bar/space alternation).
    // Shifting by one module width turns bars into spaces → maximum anti-correlation.
    // So the first minimum lag ≈ module_width.
    int first_min = -1;
    for (int i = 2; i < N / 2; ++i) {
        if (autocorr[i] < autocorr[i - 1] && autocorr[i] <= autocorr[i + 1]) {
            first_min = i;
            break;
        }
    }
    if (first_min < 0) return 0.0f;

    // Parabolic refinement for sub-sample precision
    float a = autocorr[first_min - 1];
    float b = autocorr[first_min];
    float c = autocorr[first_min + 1];
    float denom = a - 2.0f * b + c;
    float offset = 0.0f;
    if (std::abs(denom) > 1e-9f) {
        offset = 0.5f * (a - c) / denom;
    }
    float mw = static_cast<float>(first_min) + offset;

    // Sanity check
    float max_mw = static_cast<float>(N) / 20.0f;
    if (mw < 1.0f || mw > max_mw) return 0.0f;

    return mw;
}

Scanline make_digit_template(int digit, float module_width, char code_type) {
    auto pattern = L_PATTERNS[digit];

    // G-code: reverse the pattern
    std::array<int, 4> p;
    if (code_type == 'G') {
        for (int i = 0; i < 4; ++i) p[i] = pattern[3 - i];
    } else {
        p = pattern;
    }

    int total_samples = static_cast<int>(std::round(7.0f * module_width));
    Scanline templ(total_samples);

    // L/G-code: starts with space (255), alternates space/bar/space/bar
    // R-code: invert polarity — starts with bar (0), alternates bar/space/bar/space
    float bar_val = (code_type == 'R') ? 255.0f : 0.0f;
    float space_val = (code_type == 'R') ? 0.0f : 255.0f;

    int pos = 0;
    for (int i = 0; i < 4; ++i) {
        float val = (i % 2 == 0) ? space_val : bar_val;
        int width = static_cast<int>(std::round((pos + p[i]) * module_width)) -
                    static_cast<int>(std::round(pos * module_width));
        for (int j = 0; j < width && (static_cast<int>(std::round(pos * module_width)) + j) < total_samples; ++j) {
            templ[static_cast<int>(std::round(pos * module_width)) + j] = val;
        }
        pos += p[i];
    }

    return templ;
}

Scanline make_guard_template(const std::string& guard_type, float module_width) {
    if (guard_type == "center") {
        // space-bar-space-bar-space (5 modules)
        int total = static_cast<int>(std::round(5.0f * module_width));
        Scanline templ(total);
        for (int i = 0; i < 5; ++i) {
            float val = (i % 2 == 0) ? 255.0f : 0.0f;
            int start = static_cast<int>(std::round(i * module_width));
            int end = static_cast<int>(std::round((i + 1) * module_width));
            for (int j = start; j < end && j < total; ++j) {
                templ[j] = val;
            }
        }
        return templ;
    }

    // left/right guard: bar-space-bar (3 modules)
    int total = static_cast<int>(std::round(3.0f * module_width));
    Scanline templ(total);
    for (int i = 0; i < 3; ++i) {
        float val = (i % 2 == 0) ? 0.0f : 255.0f;
        int start = static_cast<int>(std::round(i * module_width));
        int end = static_cast<int>(std::round((i + 1) * module_width));
        for (int j = start; j < end && j < total; ++j) {
            templ[j] = val;
        }
    }
    return templ;
}

float normalized_cross_correlation(const Scanline& a, const Scanline& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    const int N = static_cast<int>(a.size());

    float mean_a = 0.0f, mean_b = 0.0f;
    for (int i = 0; i < N; ++i) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= N;
    mean_b /= N;

    float dot = 0.0f, var_a = 0.0f, var_b = 0.0f;
    for (int i = 0; i < N; ++i) {
        float da = a[i] - mean_a;
        float db = b[i] - mean_b;
        dot += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    float denom = std::sqrt(var_a * var_b);
    if (denom < 1e-9f) return 0.0f;
    return dot / denom;
}

CorrelationMatch slide_correlate(const Scanline& signal, const Scanline& templ) {
    const int sig_n = static_cast<int>(signal.size());
    const int tmpl_n = static_cast<int>(templ.size());
    if (tmpl_n > sig_n || tmpl_n == 0) return {};

    CorrelationMatch best;
    for (int pos = 0; pos <= sig_n - tmpl_n; ++pos) {
        Scanline window(signal.begin() + pos, signal.begin() + pos + tmpl_n);
        float ncc = normalized_cross_correlation(window, templ);
        if (ncc > best.correlation) {
            best.correlation = ncc;
            best.position = pos;
        }
    }
    return best;
}

// Compute NCC between template and signal at given offset, without allocating.
static float ncc_at(const Scanline& signal, const Scanline& templ, int offset) {
    const int N = static_cast<int>(templ.size());
    if (offset < 0 || offset + N > static_cast<int>(signal.size())) return -2.0f;

    float mean_s = 0.0f, mean_t = 0.0f;
    for (int i = 0; i < N; ++i) {
        mean_s += signal[offset + i];
        mean_t += templ[i];
    }
    mean_s /= static_cast<float>(N);
    mean_t /= static_cast<float>(N);

    float dot = 0.0f, var_s = 0.0f, var_t = 0.0f;
    for (int i = 0; i < N; ++i) {
        float ds = signal[offset + i] - mean_s;
        float dt = templ[i] - mean_t;
        dot += ds * dt;
        var_s += ds * ds;
        var_t += dt * dt;
    }

    float denom = std::sqrt(var_s * var_t);
    if (denom < 1e-9f) return 0.0f;
    return dot / denom;
}

// Attempt correlation decode with a given module width estimate.
// Returns success only if checksum validates.
static DecodeResult try_correlation_decode(const Scanline& signal, float trial_mw) {
    const int N = static_cast<int>(signal.size());
    float mw = trial_mw;

    // Barcode must fit in signal
    int barcode_modules = static_cast<int>(std::round(95.0f * mw));
    if (barcode_modules > N || barcode_modules < 20) return {false, "", 0.0f, "EAN-13"};

    // Use extended guard templates (guard + adjacent quiet zone) for distinctiveness.
    // Short 3-module guards match many interior bar-space-bar patterns; adding quiet
    // zone context makes the templates unique to actual guard positions.
    constexpr int QUIET_MODS = 3;
    int ext_len = static_cast<int>(std::round((QUIET_MODS + 3) * mw));

    // Extended left guard: quiet(3mw) + bar-space-bar(3mw)
    Scanline ext_lg(ext_len, 255.0f);
    int lg_quiet_end = static_cast<int>(std::round(QUIET_MODS * mw));
    for (int m = 0; m < 3; ++m) {
        float val = (m % 2 == 0) ? 0.0f : 255.0f;
        int s = lg_quiet_end + static_cast<int>(std::round(m * mw));
        int e = lg_quiet_end + static_cast<int>(std::round((m + 1) * mw));
        for (int j = s; j < e && j < ext_len; ++j) ext_lg[j] = val;
    }

    // Extended right guard: bar-space-bar(3mw) + quiet(3mw)
    Scanline ext_rg(ext_len, 255.0f);
    for (int m = 0; m < 3; ++m) {
        float val = (m % 2 == 0) ? 0.0f : 255.0f;
        int s = static_cast<int>(std::round(m * mw));
        int e = static_cast<int>(std::round((m + 1) * mw));
        for (int j = s; j < e && j < ext_len; ++j) ext_rg[j] = val;
    }

    // Two-guard search: find LG local maxima, validate each with RG.
    // Distance from ext_lg match pos to ext_rg match pos = 95 modules.
    int barcode_start = -1;
    float best_combined = -1.0f;
    int max_start = N - static_cast<int>(std::round((95 + QUIET_MODS) * mw));
    if (max_start < 0) return {false, "", 0.0f, "EAN-13"};

    float prev_ncc = -2.0f, prev_prev_ncc = -2.0f;
    for (int pos = 0; pos <= max_start + 1; ++pos) {
        float ncc = (pos <= max_start) ? ncc_at(signal, ext_lg, pos) : -2.0f;

        if (pos >= 2 && prev_ncc > 0.3f &&
            prev_ncc >= prev_prev_ncc && prev_ncc >= ncc) {
            int lg_match = pos - 1;

            // Search for right guard near expected position
            int expected_rg = lg_match + static_cast<int>(std::round(95.0f * mw));
            int rg_margin = static_cast<int>(5.0f * mw);
            int rg_start = std::max(lg_match + ext_len, expected_rg - rg_margin);
            int rg_end = std::min(N - ext_len, expected_rg + rg_margin);

            float best_rg_ncc = -2.0f;
            int best_rg_match = -1;
            for (int rp = rg_start; rp <= rg_end; ++rp) {
                float rg_ncc = ncc_at(signal, ext_rg, rp);
                if (rg_ncc > best_rg_ncc) {
                    best_rg_ncc = rg_ncc;
                    best_rg_match = rp;
                }
            }

            if (best_rg_ncc > 0.3f) {
                float combined = prev_ncc + best_rg_ncc;
                if (combined > best_combined) {
                    best_combined = combined;
                    // Refine mw from guard pair distance
                    float refined = static_cast<float>(best_rg_match - lg_match) / 95.0f;
                    if (refined >= 1.0f) {
                        mw = refined;
                        barcode_start = lg_match + static_cast<int>(std::round(QUIET_MODS * mw));
                    }
                }
            }
        }

        prev_prev_ncc = prev_ncc;
        prev_ncc = ncc;
    }

    if (barcode_start < 0) return {false, "", 0.0f, "EAN-13"};

    // Decode left 6 digits
    int left_data_start = barcode_start + static_cast<int>(std::round(3.0f * mw));
    std::array<int, 13> digits{};
    std::array<int, 6> parity{};
    float total_corr = 0.0f;

    for (int d = 0; d < 6; ++d) {
        int nominal = left_data_start + static_cast<int>(std::round(d * 7.0f * mw));
        float best_corr = -2.0f;
        int best_digit = -1;
        bool best_is_g = false;

        for (int digit = 0; digit < 10; ++digit) {
            for (char code : {'L', 'G'}) {
                auto templ = make_digit_template(digit, mw, code);
                for (int off = -4; off <= 4; ++off) {
                    float ncc = ncc_at(signal, templ, nominal + off);
                    if (ncc > best_corr) {
                        best_corr = ncc;
                        best_digit = digit;
                        best_is_g = (code == 'G');
                    }
                }
            }
        }

        if (best_digit < 0) return {false, "", 0.0f, "EAN-13"};
        digits[d + 1] = best_digit;
        parity[d] = best_is_g ? 1 : 0;
        total_corr += best_corr;
    }

    // Decode right 6 digits
    int right_data_start = left_data_start + static_cast<int>(std::round(47.0f * mw));

    for (int d = 0; d < 6; ++d) {
        int nominal = right_data_start + static_cast<int>(std::round(d * 7.0f * mw));
        float best_corr = -2.0f;
        int best_digit = -1;

        for (int digit = 0; digit < 10; ++digit) {
            auto templ = make_digit_template(digit, mw, 'R');
            for (int off = -4; off <= 4; ++off) {
                float ncc = ncc_at(signal, templ, nominal + off);
                if (ncc > best_corr) {
                    best_corr = ncc;
                    best_digit = digit;
                }
            }
        }

        if (best_digit < 0) return {false, "", 0.0f, "EAN-13"};
        digits[d + 7] = best_digit;
        total_corr += best_corr;
    }

    // First digit from parity pattern
    int first_digit = -1;
    for (int d = 0; d < 10; ++d) {
        bool match = true;
        for (int i = 0; i < 6; ++i) {
            if (PARITY_PATTERNS[d][i] != parity[i]) { match = false; break; }
        }
        if (match) { first_digit = d; break; }
    }
    if (first_digit < 0) return {false, "", 0.0f, "EAN-13"};
    digits[0] = first_digit;

    // Verify check digit
    int check_sum = 0;
    for (int i = 0; i < 12; ++i) {
        check_sum += digits[i] * (i % 2 == 0 ? 1 : 3);
    }
    int expected_check = (10 - (check_sum % 10)) % 10;
    if (digits[12] != expected_check) return {false, "", 0.0f, "EAN-13"};

    std::string text;
    text.reserve(13);
    for (int d : digits) text += static_cast<char>('0' + d);

    float confidence = total_corr / 12.0f;
    // Reject low-confidence decodes that likely passed checksum by coincidence
    if (confidence < 0.84f) return {false, "", 0.0f, "EAN-13"};
    return {true, text, confidence, "EAN-13"};
}

DecodeResult decode_ean13_correlation(const Scanline& signal) {
    const int N = static_cast<int>(signal.size());
    if (N < 30) return {false, "", 0.0f, "EAN-13"};

    DecodeResult best{false, "", 0.0f, "EAN-13"};

    auto try_mw = [&](float trial) {
        auto result = try_correlation_decode(signal, trial);
        if (result.success && result.confidence > best.confidence) {
            best = result;
        }
    };

    // Fast path: try autocorrelation-based module width estimate
    float acorr_mw = estimate_module_width(signal);
    if (acorr_mw > 0.0f) {
        try_mw(acorr_mw);
        if (best.confidence > 0.9f) return best; // high confidence, skip search
    }

    // Multi-scale search: try a range of module widths, keep highest confidence
    for (float trial = 1.5f; trial <= 12.0f; trial += 0.25f) {
        try_mw(trial);
    }

    return best;
}

DecodeResult decode_scanline(const Scanline& signal, DecodeMethod method) {
    switch (method) {
    case DecodeMethod::EdgeThreshold:
        return decode_scanline(signal, EdgeMethod::Threshold);
    case DecodeMethod::EdgeGradient:
        return decode_scanline(signal, EdgeMethod::Gradient);
    case DecodeMethod::Correlation:
        return decode_ean13_correlation(signal);
    }
    return {false, "", 0.0f, ""};
}

} // namespace bc
