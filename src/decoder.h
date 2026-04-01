#pragma once

#include "types.h"

#include <vector>

namespace bc {

struct Edge {
    float position; // Sub-sample index of the edge
    bool rising;    // true = dark-to-light (value increasing), false = light-to-dark
};

// Detect edges (bar/space transitions) in a scanline signal.
// If threshold == 0, uses the midpoint between signal min and max.
std::vector<Edge> detect_edges(const Scanline& signal, float threshold = 0.0f);

// Measure alternating bar/space widths from edge positions.
std::vector<float> measure_widths(const std::vector<Edge>& edges);

// Decode EAN-13 from measured bar/space widths.
DecodeResult decode_ean13(const std::vector<float>& widths);

// Full pipeline: scanline -> edges -> widths -> decode.
// Tries the midpoint threshold, then adjusts if needed.
DecodeResult decode_scanline(const Scanline& signal);

} // namespace bc
