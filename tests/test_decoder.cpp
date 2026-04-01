#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "decoder.h"
#include "image.h"

using namespace bc;
using Catch::Matchers::WithinAbs;

// Helper: generate a perfect EAN-13 scanline from module pattern
static Scanline make_perfect_scanline(const std::string& digits, float module_width) {
    auto img = make_ean13_image(digits, static_cast<int>(module_width), 1);
    Scanline line(img.width);
    for (int x = 0; x < img.width; ++x) {
        line[x] = static_cast<float>(img.at(x, 0));
    }
    return line;
}

TEST_CASE("Edge detection on square wave", "[decoder][edges]") {
    // Create a simple square wave: low-high-low-high
    Scanline signal = {0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0};

    auto edges = detect_edges(signal);

    // Should find 4 edges
    REQUIRE(edges.size() == 4);

    // First edge: rising (0 -> 255) around index 2-3
    REQUIRE(edges[0].rising == true);
    REQUIRE(edges[0].position > 2.0f);
    REQUIRE(edges[0].position < 3.0f);

    // Second edge: falling (255 -> 0) around index 5-6
    REQUIRE(edges[1].rising == false);
}

TEST_CASE("Measure widths from edges", "[decoder][widths]") {
    std::vector<Edge> edges = {
        {1.0f, true},
        {4.0f, false},
        {6.0f, true},
        {9.0f, false},
    };

    auto widths = measure_widths(edges);
    REQUIRE(widths.size() == 3);
    REQUIRE_THAT(widths[0], WithinAbs(3.0f, 0.01));
    REQUIRE_THAT(widths[1], WithinAbs(2.0f, 0.01));
    REQUIRE_THAT(widths[2], WithinAbs(3.0f, 0.01));
}

TEST_CASE("Decode perfect synthetic EAN-13", "[decoder][ean13]") {
    // Standard test barcode: 5901234123457
    std::string code = "5901234123457";

    SECTION("module width 3") {
        auto scanline = make_perfect_scanline(code, 3.0f);
        auto result = decode_scanline(scanline);
        REQUIRE(result.success);
        REQUIRE(result.text == code);
        REQUIRE(result.format == "EAN-13");
    }

    SECTION("module width 2") {
        auto scanline = make_perfect_scanline(code, 2.0f);
        auto result = decode_scanline(scanline);
        REQUIRE(result.success);
        REQUIRE(result.text == code);
    }

    SECTION("module width 5") {
        auto scanline = make_perfect_scanline(code, 5.0f);
        auto result = decode_scanline(scanline);
        REQUIRE(result.success);
        REQUIRE(result.text == code);
    }
}

TEST_CASE("Decode various EAN-13 codes", "[decoder][ean13]") {
    std::vector<std::string> codes = {
        "4006381333931",
        "0012345678905",
        "9780201379624",  // ISBN
        "8413000065504",
    };

    for (const auto& code : codes) {
        CAPTURE(code);
        auto scanline = make_perfect_scanline(code, 3.0f);
        auto result = decode_scanline(scanline);
        REQUIRE(result.success);
        REQUIRE(result.text == code);
    }
}

TEST_CASE("Decode fails on garbage data", "[decoder][ean13]") {
    Scanline noise(200, 128.0f);
    auto result = decode_scanline(noise);
    REQUIRE_FALSE(result.success);
}
