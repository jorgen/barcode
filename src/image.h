#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace bc {

struct Image {
    std::vector<uint8_t> pixels; // row-major grayscale
    int width = 0;
    int height = 0;

    [[nodiscard]] uint8_t at(int x, int y) const {
        return pixels[y * width + x];
    }

    // Bilinear interpolation at sub-pixel coordinates. Clamps to image bounds.
    [[nodiscard]] float sample(float fx, float fy) const {
        float cx = std::clamp(fx, 0.0f, static_cast<float>(width - 1));
        float cy = std::clamp(fy, 0.0f, static_cast<float>(height - 1));

        int x0 = static_cast<int>(cx);
        int y0 = static_cast<int>(cy);
        int x1 = std::min(x0 + 1, width - 1);
        int y1 = std::min(y0 + 1, height - 1);

        float dx = cx - x0;
        float dy = cy - y0;

        float top    = at(x0, y0) * (1 - dx) + at(x1, y0) * dx;
        float bottom = at(x0, y1) * (1 - dx) + at(x1, y1) * dx;

        return top * (1 - dy) + bottom * dy;
    }
};

// Load any image (PNG, JPG, BMP, PGM, ...) as grayscale via stb_image.
Image load_image(const std::string& path);

// Generate a synthetic EAN-13 barcode image.
// module_width: pixels per thinnest element. height: image height in pixels.
// digits: 13-character string of the EAN-13 code (including check digit).
Image make_ean13_image(const std::string& digits, int module_width, int height);

} // namespace bc
