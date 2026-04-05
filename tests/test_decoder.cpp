#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "decoder.h"
#include "image.h"

#include <algorithm>
#include <random>

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

TEST_CASE("Gradient edge detection on square wave", "[decoder][edges][gradient]") {
    Scanline signal = {0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0};

    auto edges = detect_edges_gradient(signal, 50.0f);

    // Should find 4 edges (same as threshold method)
    REQUIRE(edges.size() == 4);

    // First edge: rising
    REQUIRE(edges[0].rising == true);
    REQUIRE(edges[0].position > 2.0f);
    REQUIRE(edges[0].position < 3.5f);

    // Second edge: falling
    REQUIRE(edges[1].rising == false);
}

TEST_CASE("Gradient edge detection on noisy signal", "[decoder][edges][gradient][noise]") {
    // Create a square wave with noise — gradient should still find the right edges
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 10.0f);

    Scanline signal(100);
    for (int i = 0; i < 100; ++i) {
        float base = ((i / 10) % 2 == 0) ? 20.0f : 230.0f;
        signal[i] = std::clamp(base + noise(rng), 0.0f, 255.0f);
    }

    auto edges_thresh = detect_edges(signal);
    auto edges_grad = detect_edges_gradient(signal);

    // Both methods should find transitions in a 10-segment square wave
    REQUIRE(edges_grad.size() >= 5);
    REQUIRE(edges_grad.size() <= 30);
}

TEST_CASE("Otsu threshold on bimodal signal", "[decoder][otsu]") {
    // Bimodal: half the values near 50, half near 200
    Scanline signal;
    for (int i = 0; i < 50; ++i) signal.push_back(50.0f);
    for (int i = 0; i < 50; ++i) signal.push_back(200.0f);

    float thresh = otsu_threshold(signal);

    // Otsu should find threshold between the two modes
    REQUIRE(thresh >= 50.0f);
    REQUIRE(thresh <= 200.0f);
}

TEST_CASE("Otsu threshold is better than midpoint for asymmetric signal", "[decoder][otsu]") {
    // 80% low values, 20% high values — midpoint will be wrong
    Scanline signal;
    for (int i = 0; i < 80; ++i) signal.push_back(30.0f);
    for (int i = 0; i < 20; ++i) signal.push_back(220.0f);

    float midpoint = (30.0f + 220.0f) / 2.0f; // 125 — too high for this distribution
    float otsu = otsu_threshold(signal);

    // Otsu should be lower than midpoint, closer to the boundary
    REQUIRE(otsu < midpoint);
    REQUIRE(otsu >= 30.0f);
}

TEST_CASE("Decode synthetic EAN-13 with gradient method", "[decoder][ean13][gradient]") {
    std::string code = "5901234123457";
    auto scanline = make_perfect_scanline(code, 3.0f);
    auto result = decode_scanline(scanline, EdgeMethod::Gradient);
    REQUIRE(result.success);
    REQUIRE(result.text == code);
}

TEST_CASE("Decode fails on garbage data", "[decoder][ean13]") {
    Scanline noise(200, 128.0f);
    auto result = decode_scanline(noise);
    REQUIRE_FALSE(result.success);
}
