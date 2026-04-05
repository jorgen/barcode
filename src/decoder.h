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

} // namespace bc
