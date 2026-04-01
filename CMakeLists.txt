cmake_minimum_required(VERSION 3.20)
project(barcode_dct VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

include(FetchContent)

# stb - single-header image loading (public domain)
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(stb)

# zxing (Java) - fetched for barcode test images in core/src/test/resources/blackbox/
# Contains EAN-13, Code128, UPC-A, etc. test images with expected results
# SOURCE_SUBDIR points to a data-only dir so CMake won't try to build the Java project
FetchContent_Declare(
    zxing_testdata
    GIT_REPOSITORY https://github.com/zxing/zxing.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE
    SOURCE_SUBDIR  core/src/test/resources
)
FetchContent_MakeAvailable(zxing_testdata)

add_subdirectory(src)
add_subdirectory(app)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
