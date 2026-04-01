#pragma once

#include "types.h"
#include "image.h"

#include <vector>

namespace bc {

struct ExtractionParams {
    int num_scanlines = 5;         // Number of parallel scanlines to sample
    float scanline_spacing = 2.0f; // Pixels between parallel scanlines
    int samples_per_scanline = 0;  // 0 = auto (use region width along direction)
    float sample_step = 1.0f;      // Pixels between samples along scan direction
};

// Extract scanlines from image within the given region.
// Samples along region.direction, spaced across the perpendicular axis.
std::vector<Scanline> extract_scanlines(
    const Image& img,
    const BarcodeRegion& region,
    const ExtractionParams& params = {}
);

// Average multiple scanlines into one (noise reduction).
Scanline average_scanlines(const std::vector<Scanline>& scanlines);

} // namespace bc
