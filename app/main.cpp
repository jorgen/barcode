#include "image.h"
#include "scanline.h"
#include "dct.h"
#include "decoder.h"

#include <cstdlib>
#include <iostream>
#include <string>

static void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <image_path> [options]\n"
              << "\n"
              << "Options:\n"
              << "  --region x,y,w,h     Barcode region AABB (default: full image)\n"
              << "  --direction dx,dy    Scan direction (default: 1,0 = horizontal)\n"
              << "  --filter <type>      Filter type: none, lowpass, hard, soft (default: lowpass)\n"
              << "  --cutoff <float>     Filter cutoff (default: 0.3)\n"
              << "  --scanlines <int>    Number of scanlines (default: 5)\n"
              << "\n";
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string image_path = argv[1];

    // Defaults
    float rx = -1, ry = -1, rw = -1, rh = -1;
    float dx = 1.0f, dy = 0.0f;
    std::string filter_type = "lowpass";
    float cutoff = 0.3f;
    int num_scanlines = 5;

    // Parse arguments
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--region" && i + 1 < argc) {
            std::sscanf(argv[++i], "%f,%f,%f,%f", &rx, &ry, &rw, &rh);
        } else if (arg == "--direction" && i + 1 < argc) {
            std::sscanf(argv[++i], "%f,%f", &dx, &dy);
        } else if (arg == "--filter" && i + 1 < argc) {
            filter_type = argv[++i];
        } else if (arg == "--cutoff" && i + 1 < argc) {
            cutoff = std::stof(argv[++i]);
        } else if (arg == "--scanlines" && i + 1 < argc) {
            num_scanlines = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        auto img = bc::load_image(image_path);
        std::cerr << "Loaded image: " << img.width << "x" << img.height << "\n";

        // Default region: full image
        bc::BarcodeRegion region;
        if (rx >= 0) {
            region = {rx, ry, rw, rh, bc::Vec2f{dx, dy}.normalized()};
        } else {
            region = {0, 0, static_cast<float>(img.width), static_cast<float>(img.height),
                      bc::Vec2f{dx, dy}.normalized()};
        }

        // Extract scanlines
        bc::ExtractionParams ep;
        ep.num_scanlines = num_scanlines;
        ep.scanline_spacing = 2.0f;

        auto lines = bc::extract_scanlines(img, region, ep);
        auto scanline = bc::average_scanlines(lines);

        std::cerr << "Extracted " << lines.size() << " scanlines, "
                  << scanline.size() << " samples each\n";

        // Apply filter
        bc::Scanline filtered;
        if (filter_type == "none") {
            filtered = scanline;
        } else {
            bc::FilterParams fp;
            fp.cutoff = cutoff;
            if (filter_type == "lowpass") {
                fp.type = bc::FilterType::LowPass;
            } else if (filter_type == "hard") {
                fp.type = bc::FilterType::HardThreshold;
                fp.threshold = cutoff;
            } else if (filter_type == "soft") {
                fp.type = bc::FilterType::SoftThreshold;
                fp.threshold = cutoff;
            }
            filtered = bc::dct_filter(scanline, fp);
        }

        // Decode
        auto result = bc::decode_scanline(filtered);

        if (result.success) {
            std::cout << result.text << "\n";
            std::cerr << "Format: " << result.format
                      << " | Confidence: " << result.confidence << "\n";
            return 0;
        } else {
            std::cerr << "Failed to decode barcode\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
