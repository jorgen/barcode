#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "scanline.h"

using namespace bc;
using Catch::Matchers::WithinAbs;

TEST_CASE("Extract horizontal scanline from uniform image", "[scanline]") {
    Image img;
    img.width = 20;
    img.height = 10;
    img.pixels.assign(200, 128);

    BarcodeRegion region{0, 0, 20, 10, {1.0f, 0.0f}};
    ExtractionParams params;
    params.num_scanlines = 1;
    params.sample_step = 1.0f;

    auto lines = extract_scanlines(img, region, params);
    REQUIRE(lines.size() == 1);

    for (auto v : lines[0]) {
        REQUIRE_THAT(v, WithinAbs(128.0f, 0.01));
    }
}

TEST_CASE("Extract scanline from gradient image", "[scanline]") {
    // Create a horizontal gradient: pixel value = x
    Image img;
    img.width = 100;
    img.height = 10;
    img.pixels.resize(1000);
    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 100; ++x) {
            img.pixels[y * 100 + x] = static_cast<uint8_t>(x);
        }
    }

    BarcodeRegion region{0, 0, 100, 10, {1.0f, 0.0f}};
    ExtractionParams params;
    params.num_scanlines = 1;
    params.sample_step = 1.0f;

    auto lines = extract_scanlines(img, region, params);
    REQUIRE(lines.size() == 1);
    REQUIRE(lines[0].size() > 50);

    // Values should increase along the scanline
    for (size_t i = 1; i < lines[0].size(); ++i) {
        REQUIRE(lines[0][i] >= lines[0][i - 1] - 1.0f); // allow tiny tolerance
    }
}

TEST_CASE("Multiple scanlines are extracted in parallel", "[scanline]") {
    Image img;
    img.width = 50;
    img.height = 20;
    img.pixels.assign(1000, 100);

    BarcodeRegion region{0, 0, 50, 20, {1.0f, 0.0f}};
    ExtractionParams params;
    params.num_scanlines = 5;
    params.scanline_spacing = 2.0f;

    auto lines = extract_scanlines(img, region, params);
    REQUIRE(lines.size() == 5);
}

TEST_CASE("Average scanlines reduces noise", "[scanline]") {
    // Two scanlines: one high, one low
    Scanline a(10, 200.0f);
    Scanline b(10, 100.0f);

    auto avg = average_scanlines({a, b});
    REQUIRE(avg.size() == 10);

    for (auto v : avg) {
        REQUIRE_THAT(v, WithinAbs(150.0f, 0.01));
    }
}

TEST_CASE("Bilinear sampling interpolates correctly", "[scanline]") {
    Image img;
    img.width = 2;
    img.height = 2;
    img.pixels = {0, 100, 0, 100};

    // Sample at center: should be average of all 4 pixels
    REQUIRE_THAT(img.sample(0.5f, 0.5f), WithinAbs(50.0f, 0.01));

    // Sample at corners
    REQUIRE_THAT(img.sample(0.0f, 0.0f), WithinAbs(0.0f, 0.01));
    REQUIRE_THAT(img.sample(1.0f, 0.0f), WithinAbs(100.0f, 0.01));
}
