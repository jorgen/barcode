macro(Build3rdParty)
    add_subdirectory(${catch2_SOURCE_DIR} ${CMAKE_BINARY_DIR}/catch2 SYSTEM)

    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    add_subdirectory(${glfw_SOURCE_DIR} ${CMAKE_BINARY_DIR}/glfw SYSTEM)
endmacro()
