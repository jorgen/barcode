# barcode_dct

A C++20 barcode decoder that uses Discrete Cosine Transform (DCT) filtering to decode 1D barcodes from noisy or low-resolution images.

The primary goal of this project is to explore whether frequency-domain filtering (DCT) can improve barcode decoding when the thinnest bar is close to 1 pixel wide. At these scales, traditional spatial-domain binarization breaks down — a single pixel might straddle a bar/space boundary, and sensor noise becomes comparable to the signal. By transforming the scanline into the frequency domain, we can separate the structured barcode signal from random noise before attempting to decode.

## Architecture

Detection is assumed to be done externally. The decoder takes a grayscale image, a bounding box (AABB), and an orientation vector describing the scan direction:

```
Image + BarcodeRegion(AABB + direction)
        │
        ▼
 Scanline Extraction ── bilinear interpolation along orientation,
        │                multiple parallel lines, optional averaging
        ▼
    DCT Filtering ───── DCT-II → frequency thresholding → DCT-III
        │                (low-pass, hard/soft threshold, band-pass)
        ▼
      Decoding ──────── edge detection → width measurement → EAN-13 pattern match
        │
        ▼
    DecodeResult { text, confidence, format }
```

### Modules

| File | Purpose |
|------|---------|
| `src/types.h` | `Vec2f`, `BarcodeRegion`, `Scanline`, `DecodeResult` |
| `src/image.h/cpp` | `Image` struct with bilinear sampling, `load_image()` (PNG/JPG/BMP via stb), `make_ean13_image()` synthetic generator |
| `src/scanline.h/cpp` | Multi-scanline extraction along arbitrary orientation, averaging |
| `src/dct.h/cpp` | DCT-II/III transforms, four filtering strategies, power spectrum analysis |
| `src/decoder.h/cpp` | Edge detection, width measurement, EAN-13 decoding |
| `app/main.cpp` | CLI tool |

## Building

Requires CMake 3.20+ and a C++20 compiler. All dependencies are fetched automatically.

```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build
```

### Dependencies (fetched via CMake FetchContent)

- **[stb_image](https://github.com/nothings/stb)** — single-header image loader (public domain)
- **[Catch2 v3](https://github.com/catchorg/Catch2)** — testing framework (test-only)
- **[zxing test data](https://github.com/zxing/zxing)** — real barcode images for integration testing

## Running Tests

```bash
ctest --test-dir build -C Debug --output-on-failure
```

20 tests covering:
- DCT round-trip correctness and Parseval's energy preservation
- Scanline extraction geometry and bilinear interpolation
- EAN-13 decoding of perfect synthetic barcodes at various module widths
- Full pipeline with Gaussian noise + DCT denoising

## CLI Usage

```bash
barcode_decode <image_path> [options]
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--region x,y,w,h` | Barcode bounding box | Full image |
| `--direction dx,dy` | Scan direction vector | `1,0` (horizontal) |
| `--filter <type>` | `none`, `lowpass`, `hard`, `soft` | `lowpass` |
| `--cutoff <float>` | Filter cutoff / threshold | `0.3` |
| `--scanlines <int>` | Number of parallel scanlines | `5` |

**Example:**
```bash
barcode_decode photo.jpg --region 100,200,300,80 --filter lowpass --cutoff 0.25
# Output: 5901234123457
```

## DCT Filtering

The core research idea: a barcode scanline is a band-limited signal (its highest meaningful frequency is determined by the module width). Noise is broadband. A well-chosen low-pass filter in the DCT domain should separate the two.

Four filtering strategies are implemented:

- **LowPass** — keep the lowest N% of frequency coefficients, zero the rest
- **HardThreshold** — zero any coefficient with magnitude below a threshold
- **SoftThreshold** — shrink all coefficients toward zero by the threshold amount (better for preserving edges)
- **BandPass** — keep coefficients in a specific frequency index range

The DCT-II/III are implemented directly from the formulas (O(N^2)) with no external FFT library. At typical scanline lengths of 100-400 samples this is trivial for modern CPUs.

## Supported Barcode Formats

- **EAN-13** — fully implemented with checksum validation

Planned:
- Code 128
- UPC-A
- Code 39
