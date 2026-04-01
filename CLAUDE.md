# CLAUDE.md

## Project Overview

C++20 barcode decoder using DCT-based noise filtering. Focused on 1D barcodes where the thinnest bar is close to 1 pixel. Detection is out of scope — we receive a bounding box + orientation vector.

## Build & Test

```bash
cmake -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build -C Debug --output-on-failure
```

On Windows with MSVC, the default generator is Visual Studio. Pass `-C Debug` or `-C Release` to ctest/cmake build as needed.

## Code Layout

- `src/` — static library `barcode_lib`. All public API is in the `bc` namespace.
  - `types.h` — foundation types (`Vec2f`, `BarcodeRegion`, `Scanline`, `DecodeResult`)
  - `image.h/cpp` — `Image` struct, `load_image()` (stb_image), `make_ean13_image()` synthetic generator
  - `scanline.h/cpp` — scanline extraction with bilinear interpolation
  - `dct.h/cpp` — DCT-II, DCT-III, filtering strategies, power spectrum
  - `decoder.h/cpp` — edge detection, width measurement, EAN-13 pattern matching
- `tests/` — Catch2 v3 tests. Tags: `[dct]`, `[scanline]`, `[decoder]`, `[ean13]`, `[integration]`, `[filter]`, `[noise]`
- `app/main.cpp` — CLI entry point

## Conventions

- C++20 standard. Use `std::numbers`, structured bindings, `[[nodiscard]]`, etc.
- No external dependencies in `src/` except stb_image (fetched via FetchContent, included only in `image.cpp`)
- Everything is in namespace `bc`
- `Scanline = std::vector<float>` with values in [0, 255]
- `Image` stores grayscale `uint8_t` pixels in row-major order
- Orientation is represented as `Vec2f` unit vector, not angles — perpendicular is `(-y, x)`
- DCT is O(N^2) naive implementation — intentional, scanlines are short (~100-400 samples)

## Test Data

- Synthetic barcodes generated via `make_ean13_image()` — used for most tests
- Real images from zxing/zxing repo fetched via FetchContent, available at `TEST_DATA_DIR` compile definition (points to `core/src/test/resources/blackbox/`)
- Test image directories: `ean13-1/` through `ean13-5/`, `code128-1/` through `code128-3/`, etc.
- Each test image has a `.txt` companion with the expected decoded value

## Adding a New Barcode Format

1. Add a `decode_<format>(const std::vector<float>& widths) -> DecodeResult` function in `decoder.h/cpp`
2. Update `decode_scanline()` to try the new format (after EAN-13 fails, or based on width count heuristics)
3. Add encoding tables and a synthetic image generator in `image.cpp` if needed
4. Add tests in `test_decoder.cpp`

## Key Design Decisions

- The DCT research is the core goal — keep the filtering pipeline easy to experiment with
- `BarcodeRegion` uses `float` for position/size to support sub-pixel regions from detection
- Filter strategies in `dct.h` are designed to be composable and easy to benchmark against each other
- stb_image is isolated to `image.cpp` — the rest of the codebase only sees `Image` and `load_image()`
