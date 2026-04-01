#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "image.h"

#include <array>
#include <stdexcept>

namespace bc {

Image load_image(const std::string& path) {
    int w, h, channels;
    uint8_t* data = stbi_load(path.c_str(), &w, &h, &channels, 1); // force grayscale
    if (!data) {
        throw std::runtime_error("Failed to load image: " + path + " (" + stbi_failure_reason() + ")");
    }

    Image img;
    img.width = w;
    img.height = h;
    img.pixels.assign(data, data + w * h);
    stbi_image_free(data);
    return img;
}

// EAN-13 encoding tables
// L-codes: patterns for left-half digits with odd parity
// Each entry is {bar_widths...} as space-bar-space-bar (4 elements summing to 7)
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

// First digit parity encoding (0 = L-code, 1 = G-code)
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

Image make_ean13_image(const std::string& digits, int module_width, int height) {
    if (digits.size() != 13) {
        throw std::runtime_error("EAN-13 requires exactly 13 digits");
    }

    // Build the module pattern (95 modules total)
    // 0 = space (white), 1 = bar (black)
    std::vector<int> modules;
    modules.reserve(95);

    auto emit = [&](int value, int count) {
        for (int i = 0; i < count; ++i)
            modules.push_back(value);
    };

    // Helper to emit a digit pattern
    auto emit_digit = [&](const std::array<int, 4>& pattern, bool starts_with_space) {
        // pattern is {w1, w2, w3, w4} alternating space/bar or bar/space
        int val = starts_with_space ? 0 : 1;
        for (int w : pattern) {
            emit(val, w);
            val = 1 - val;
        }
    };

    int first_digit = digits[0] - '0';

    // Left guard: bar-space-bar
    emit(1, 1); emit(0, 1); emit(1, 1);

    // Left 6 digits
    for (int i = 0; i < 6; ++i) {
        int d = digits[i + 1] - '0';
        bool use_g = PARITY_PATTERNS[first_digit][i] == 1;

        if (!use_g) {
            // L-code: starts with space
            emit_digit(L_PATTERNS[d], true);
        } else {
            // G-code: L-code reversed, starts with space
            auto pat = L_PATTERNS[d];
            std::array<int, 4> reversed = {pat[3], pat[2], pat[1], pat[0]};
            emit_digit(reversed, true);
        }
    }

    // Center guard: space-bar-space-bar-space
    emit(0, 1); emit(1, 1); emit(0, 1); emit(1, 1); emit(0, 1);

    // Right 6 digits (R-code: L-code bit-inverted, starts with bar)
    for (int i = 0; i < 6; ++i) {
        int d = digits[i + 7] - '0';
        emit_digit(L_PATTERNS[d], false);
    }

    // Right guard: bar-space-bar
    emit(1, 1); emit(0, 1); emit(1, 1);

    // Build image: add quiet zone (10 modules white on each side)
    int quiet_zone = 10;
    int total_modules = static_cast<int>(modules.size()) + 2 * quiet_zone;
    int img_width = total_modules * module_width;

    Image img;
    img.width = img_width;
    img.height = height;
    img.pixels.resize(img_width * height, 255); // white background

    for (int row = 0; row < height; ++row) {
        for (int m = 0; m < static_cast<int>(modules.size()); ++m) {
            uint8_t val = modules[m] ? 0 : 255; // bar = black, space = white
            int px_start = (quiet_zone + m) * module_width;
            for (int px = 0; px < module_width; ++px) {
                img.pixels[row * img_width + px_start + px] = val;
            }
        }
    }

    return img;
}

} // namespace bc
