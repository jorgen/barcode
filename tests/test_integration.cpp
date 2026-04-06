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

TEST_CASE("Blur + WienerDeconv recovers EAN-13 decode", "[integration][deblur]") {
    std::string code = "5901234123457";
    auto img = make_ean13_image(code, 5, 50);

    BarcodeRegion region{0, 0, static_cast<float>(img.width), static_cast<float>(img.height), {1.0f, 0.0f}};
    ExtractionParams params;
    params.num_scanlines = 1;

    auto lines = extract_scanlines(img, region, params);
    auto scanline = lines[0];

    // Mild blur (sigma=1.0 relative to module_width=5)
    float blur_sigma = 1.0f;
    FilterParams blur_params;
    blur_params.type = FilterType::Gaussian;
    blur_params.sigma = blur_sigma;
    auto blurred = dct_filter(scanline, blur_params);

    // Deconvolve
    FilterParams deconv_params;
    deconv_params.type = FilterType::WienerDeconv;
    deconv_params.blur_sigma = blur_sigma;
    deconv_params.noise_ratio = 0.01f;
    auto recovered = dct_filter(blurred, deconv_params);

    auto result_recovered = decode_scanline(recovered);
    REQUIRE(result_recovered.success);
    REQUIRE(result_recovered.text == code);
}

TEST_CASE("Blur + HighBoost recovers EAN-13 decode", "[integration][deblur]") {
    std::string code = "5901234123457";
    auto img = make_ean13_image(code, 5, 50);

    BarcodeRegion region{0, 0, static_cast<float>(img.width), static_cast<float>(img.height), {1.0f, 0.0f}};
    ExtractionParams params;
    params.num_scanlines = 1;

    auto lines = extract_scanlines(img, region, params);
    auto scanline = lines[0];

    // Mild blur
    FilterParams blur_params;
    blur_params.type = FilterType::Gaussian;
    blur_params.sigma = 1.0f;
    auto blurred = dct_filter(scanline, blur_params);

    // High-boost to sharpen
    FilterParams boost_params;
    boost_params.type = FilterType::HighBoost;
    boost_params.boost = 2.0f;
    boost_params.sigma = 3.0f;
    auto sharpened = dct_filter(blurred, boost_params);

    auto result = decode_scanline(sharpened);
    REQUIRE(result.success);
    REQUIRE(result.text == code);
}

TEST_CASE("Auto blur estimation + WienerDeconv recovers EAN-13 decode", "[integration][deblur]") {
    std::string code = "5901234123457";
    auto img = make_ean13_image(code, 5, 50);

    BarcodeRegion region{0, 0, static_cast<float>(img.width), static_cast<float>(img.height), {1.0f, 0.0f}};
    ExtractionParams params;
    params.num_scanlines = 1;

    auto lines = extract_scanlines(img, region, params);
    auto scanline = lines[0];

    // Blur the scanline
    float true_sigma = 1.5f;
    FilterParams blur_params;
    blur_params.type = FilterType::Gaussian;
    blur_params.sigma = true_sigma;
    auto blurred = dct_filter(scanline, blur_params);

    // Estimate blur from the blurred signal
    auto coeffs = dct_ii(blurred);
    float estimated_sigma = estimate_blur_sigma(coeffs);
    REQUIRE(estimated_sigma > 0.0f);

    // Use estimated sigma for deconvolution
    FilterParams deconv_params;
    deconv_params.type = FilterType::WienerDeconv;
    deconv_params.blur_sigma = estimated_sigma;
    deconv_params.noise_ratio = 0.01f;
    auto recovered = dct_filter(blurred, deconv_params);

    auto result = decode_scanline(recovered);
    REQUIRE(result.success);
    REQUIRE(result.text == code);
}

TEST_CASE("Full pipeline with correlation decode", "[integration][correlation]") {
    std::string code = "5901234123457";
    int module_width = 3;
    int height = 50;

    auto img = make_ean13_image(code, module_width, height);
    BarcodeRegion region{0, 0, static_cast<float>(img.width), static_cast<float>(img.height), {1.0f, 0.0f}};
    ExtractionParams params;
    params.num_scanlines = 1;
    params.sample_step = 1.0f;

    auto lines = extract_scanlines(img, region, params);
    REQUIRE(lines.size() == 1);

    auto result = decode_scanline(lines[0], DecodeMethod::Correlation);
    REQUIRE(result.success);
    REQUIRE(result.text == code);
}

TEST_CASE("Correlation decode handles noise that breaks edge detection", "[integration][correlation]") {
    std::string code = "5901234123457";
    // Module width 5 with noise — templates are long enough for robust matching
    auto img = make_ean13_image(code, 5, 50);

    BarcodeRegion region{0, 0, static_cast<float>(img.width), static_cast<float>(img.height), {1.0f, 0.0f}};
    ExtractionParams params;
    params.num_scanlines = 1;

    auto lines = extract_scanlines(img, region, params);
    auto scanline = lines[0];

    // Add noise that is significant relative to module width
    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0.0f, 30.0f);
    for (auto& v : scanline) {
        v = std::clamp(v + noise(rng), 0.0f, 255.0f);
    }

    // Wiener filter to clean up noise (preserves signal structure)
    FilterParams filter;
    filter.type = FilterType::Wiener;
    filter.noise_power = 900.0f;
    auto filtered = dct_filter(scanline, filter);

    // Correlation decode on filtered signal
    auto result = decode_scanline(filtered, DecodeMethod::Correlation);
    REQUIRE(result.success);
    REQUIRE(result.text == code);
}

TEST_CASE("Noisy scanline with DCT filter + correlation decode", "[integration][correlation][noise]") {
    std::string code = "5901234123457";
    // Larger module width tolerates noise better
    auto img = make_ean13_image(code, 5, 50);

    BarcodeRegion region{0, 0, static_cast<float>(img.width), static_cast<float>(img.height), {1.0f, 0.0f}};
    ExtractionParams params;
    params.num_scanlines = 1;

    auto lines = extract_scanlines(img, region, params);
    auto scanline = lines[0];

    // Add moderate noise
    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 20.0f);
    for (auto& v : scanline) {
        v = std::clamp(v + noise(rng), 0.0f, 255.0f);
    }

    // Apply Wiener filter (preserves signal structure better than low-pass)
    FilterParams filter;
    filter.type = FilterType::Wiener;
    filter.noise_power = 400.0f; // ~noise_stddev^2

    auto filtered = dct_filter(scanline, filter);

    auto result = decode_scanline(filtered, DecodeMethod::Correlation);
    REQUIRE(result.success);
    REQUIRE(result.text == code);
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
