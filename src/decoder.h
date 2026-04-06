#pragma once

#include "types.h"

#include <vector>

namespace bc {

struct Edge {
    float position; // Sub-sample index of the edge
    bool rising;    // true = dark-to-light (value increasing), false = light-to-dark
};

enum class EdgeMethod {
    Threshold,  // Classic threshold-crossing detection
    Gradient,   // Derivative peak detection
};

// Detect edges (bar/space transitions) in a scanline signal.
// If threshold == 0, uses the midpoint between signal min and max.
std::vector<Edge> detect_edges(const Scanline& signal, float threshold = 0.0f);

// Detect edges via peaks in the first derivative.
// min_gradient: minimum |derivative| to count as edge. 0 = auto (median-based).
std::vector<Edge> detect_edges_gradient(const Scanline& signal, float min_gradient = 0.0f);

// Compute optimal binarization threshold via Otsu's method.
// Returns the threshold that maximizes inter-class variance for a bimodal signal.
[[nodiscard]] float otsu_threshold(const Scanline& signal);

// Measure alternating bar/space widths from edge positions.
std::vector<float> measure_widths(const std::vector<Edge>& edges);

// Decode EAN-13 from measured bar/space widths.
DecodeResult decode_ean13(const std::vector<float>& widths);

// Full pipeline: scanline -> edges -> widths -> decode.
// method selects edge detection strategy.
DecodeResult decode_scanline(const Scanline& signal, EdgeMethod method = EdgeMethod::Threshold);

// --- Correlation-based decoder ---

// Estimate module width from autocorrelation of the signal.
[[nodiscard]] float estimate_module_width(const Scanline& signal);

// Generate a waveform template for an EAN-13 digit.
// code_type: 'L', 'G', or 'R'
[[nodiscard]] Scanline make_digit_template(int digit, float module_width, char code_type);

// Generate a guard pattern template. guard_type: "left", "right", or "center"
[[nodiscard]] Scanline make_guard_template(const std::string& guard_type, float module_width);

// Normalized cross-correlation between two equal-length signals. Returns [-1, 1].
[[nodiscard]] float normalized_cross_correlation(const Scanline& a, const Scanline& b);

struct CorrelationMatch {
    int position = -1;
    float correlation = -1.0f;
};

// Slide template across signal, return position and correlation of best match.
[[nodiscard]] CorrelationMatch slide_correlate(const Scanline& signal, const Scanline& templ);

// Full correlation-based EAN-13 decode pipeline.
[[nodiscard]] DecodeResult decode_ean13_correlation(const Scanline& signal);

enum class DecodeMethod {
    EdgeThreshold,
    EdgeGradient,
    Correlation,
};

// Unified decode entry point with method selection.
[[nodiscard]] DecodeResult decode_scanline(const Scanline& signal, DecodeMethod method);

} // namespace bc
