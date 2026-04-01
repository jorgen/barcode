#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "image.h"
#include "scanline.h"
#include "dct.h"
#include "decoder.h"

#include <random>

using namespace bc;

TEST_CASE("Full pipeline: synthetic barcode image to decoded string", "[integration]") {
    std::string code = "5901234123457";
    int module_width = 3;
    int height = 50;

    auto img = make_ean13_image(code, module_width, height);
    REQUIRE(img.width > 0);
    REQUIRE(img.height == height);

    // Define region covering the barcode (with quiet zones)
    BarcodeRegion region;
    region.x = 0;
    region.y = 0;
    region.w = static_cast<float>(img.width);
    region.h = static_cast<float>(img.height);
    region.direction = {1.0f, 0.0f};

    ExtractionParams params;
    params.num_scanlines = 1;
    params.sample_step = 1.0f;

    auto lines = extract_scanlines(img, region, params);
    REQUIRE(lines.size() == 1);

    auto result = decode_scanline(lines[0]);
    REQUIRE(result.success);
    REQUIRE(result.text == code);
}

TEST_CASE("Pipeline with DCT filtering on clean signal", "[integration][dct]") {
    std::string code = "5901234123457";
    auto img = make_ean13_image(code, 3, 50);

    BarcodeRegion region{0, 0, static_cast<float>(img.width), static_cast<float>(img.height), {1.0f, 0.0f}};
    ExtractionParams params;
    params.num_scanlines = 1;

    auto lines = extract_scanlines(img, region, params);
    auto& scanline = lines[0];

    // Apply low-pass DCT filter (should preserve the signal since it's clean)
    FilterParams filter;
    filter.type = FilterType::LowPass;
    filter.cutoff = 0.5f;

    auto filtered = dct_filter(scanline, filter);
    auto result = decode_scanline(filtered);
    REQUIRE(result.success);
    REQUIRE(result.text == code);
}

TEST_CASE("Pipeline with noisy signal and DCT denoising", "[integration][dct][noise]") {
    std::string code = "5901234123457";
    int module_width = 3;
    auto img = make_ean13_image(code, module_width, 50);

    BarcodeRegion region{0, 0, static_cast<float>(img.width), static_cast<float>(img.height), {1.0f, 0.0f}};
    ExtractionParams params;
    params.num_scanlines = 1;

    auto lines = extract_scanlines(img, region, params);
    auto scanline = lines[0];

    // Add noise
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 30.0f);
    for (auto& v : scanline) {
        v = std::clamp(v + noise(rng), 0.0f, 255.0f);
    }

    // Try decoding without filtering
    auto result_noisy = decode_scanline(scanline);

    // Apply DCT low-pass filter
    FilterParams filter;
    filter.type = FilterType::LowPass;
    filter.cutoff = 0.3f;

    auto filtered = dct_filter(scanline, filter);
    auto result_filtered = decode_scanline(filtered);

    // The filtered result should decode successfully (the noisy one may or may not)
    INFO("Noisy decode: " << (result_noisy.success ? result_noisy.text : "FAILED"));
    INFO("Filtered decode: " << (result_filtered.success ? result_filtered.text : "FAILED"));

    // At module_width=3 with stddev=30 noise, DCT filtering should help
    REQUIRE(result_filtered.success);
    REQUIRE(result_filtered.text == code);
}

TEST_CASE("Multiple scanline averaging + DCT", "[integration][averaging]") {
    std::string code = "9780201379624";
    auto img = make_ean13_image(code, 3, 50);

    BarcodeRegion region{0, 0, static_cast<float>(img.width), static_cast<float>(img.height), {1.0f, 0.0f}};
    ExtractionParams params;
    params.num_scanlines = 5;
    params.scanline_spacing = 3.0f;

    auto lines = extract_scanlines(img, region, params);
    auto averaged = average_scanlines(lines);

    auto result = decode_scanline(averaged);
    REQUIRE(result.success);
    REQUIRE(result.text == code);
}
