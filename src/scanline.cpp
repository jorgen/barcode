#include "scanline.h"

#include <cmath>

namespace bc {

std::vector<Scanline> extract_scanlines(
    const Image& img,
    const BarcodeRegion& region,
    const ExtractionParams& params)
{
    Vec2f dir = region.direction.normalized();
    Vec2f perp = dir.perpendicular();

    // Region center
    float cx = region.x + region.w * 0.5f;
    float cy = region.y + region.h * 0.5f;

    // Determine scanline length in samples
    // Project region dimensions onto scan direction to get length
    float scan_length = std::abs(region.w * dir.x) + std::abs(region.h * dir.y);
    int num_samples = params.samples_per_scanline > 0
        ? params.samples_per_scanline
        : static_cast<int>(scan_length / params.sample_step);

    if (num_samples < 1) num_samples = 1;

    std::vector<Scanline> scanlines;
    scanlines.reserve(params.num_scanlines);

    // Offset so scanlines are centered around the region center
    float perp_offset_start = -params.scanline_spacing * (params.num_scanlines - 1) * 0.5f;

    for (int s = 0; s < params.num_scanlines; ++s) {
        float perp_offset = perp_offset_start + s * params.scanline_spacing;

        // Start point: center - half the scanline length along direction + perpendicular offset
        float start_x = cx - dir.x * scan_length * 0.5f + perp.x * perp_offset;
        float start_y = cy - dir.y * scan_length * 0.5f + perp.y * perp_offset;

        Scanline line(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            float fx = start_x + dir.x * params.sample_step * i;
            float fy = start_y + dir.y * params.sample_step * i;
            line[i] = img.sample(fx, fy);
        }
        scanlines.push_back(std::move(line));
    }

    return scanlines;
}

Scanline average_scanlines(const std::vector<Scanline>& scanlines) {
    if (scanlines.empty()) return {};

    size_t len = scanlines[0].size();
    Scanline avg(len, 0.0f);

    for (const auto& line : scanlines) {
        for (size_t i = 0; i < len && i < line.size(); ++i) {
            avg[i] += line[i];
        }
    }

    float n = static_cast<float>(scanlines.size());
    for (auto& v : avg) {
        v /= n;
    }

    return avg;
}

} // namespace bc
