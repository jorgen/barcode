#pragma once

#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

namespace bc {

struct Vec2f {
    float x = 0.0f;
    float y = 0.0f;

    [[nodiscard]] float length() const { return std::sqrt(x * x + y * y); }

    [[nodiscard]] Vec2f normalized() const {
        float len = length();
        return {x / len, y / len};
    }

    [[nodiscard]] Vec2f perpendicular() const { return {-y, x}; }

    Vec2f operator+(Vec2f o) const { return {x + o.x, y + o.y}; }
    Vec2f operator-(Vec2f o) const { return {x - o.x, y - o.y}; }
    Vec2f operator*(float s) const { return {x * s, y * s}; }
};

struct BarcodeRegion {
    float x = 0, y = 0;    // Top-left of AABB
    float w = 0, h = 0;    // Width/height of AABB
    Vec2f direction;        // Unit vector along scan direction (perpendicular to bars)
};

using Scanline = std::vector<float>;

struct DecodeResult {
    bool success = false;
    std::string text;
    float confidence = 0.0f;
    std::string format;
};

} // namespace bc
