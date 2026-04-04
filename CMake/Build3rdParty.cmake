macro(Build3rdParty)
    add_subdirectory(${catch2_SOURCE_DIR} ${CMAKE_BINARY_DIR}/catch2 SYSTEM)
endmacro()
