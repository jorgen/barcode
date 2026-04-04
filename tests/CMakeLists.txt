add_executable(barcode_tests
    test_dct.cpp
    test_scanline.cpp
    test_decoder.cpp
    test_integration.cpp
)

target_link_libraries(barcode_tests PRIVATE barcode_lib Catch2::Catch2WithMain)

target_compile_definitions(barcode_tests PRIVATE
    TEST_DATA_DIR="${zxing_testdata_SOURCE_DIR}/core/src/test/resources/blackbox"
)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(barcode_tests)
