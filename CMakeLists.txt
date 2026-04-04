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
FetchContent_Declare(cmake-dep
    GIT_REPOSITORY https://github.com/jorgen/cmake-dep.git
    GIT_TAG        f61ffba
)
FetchContent_MakeAvailable(cmake-dep)
list(APPEND CMAKE_MODULE_PATH "${cmake-dep_SOURCE_DIR}/cmake")
include(CmDepMain)

set(CMDEP_PACKAGES_FILE "${CMAKE_CURRENT_SOURCE_DIR}/CMake/3rdPartyPackages.cmake")
CmDepFetch()

include(CMake/Build3rdParty.cmake)
Build3rdParty()

add_subdirectory(src)
add_subdirectory(app)
add_subdirectory(visualizer)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
