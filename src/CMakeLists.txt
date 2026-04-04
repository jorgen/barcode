add_library(barcode_lib STATIC
    image.cpp
    scanline.cpp
    dct.cpp
    decoder.cpp
)

target_include_directories(barcode_lib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(barcode_lib SYSTEM PRIVATE ${stbimage_SOURCE_DIR})
