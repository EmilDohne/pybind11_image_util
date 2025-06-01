cmake_minimum_required(VERSION 3.19)

set (CMAKE_CXX_STANDARD 20)
project (PyImageUtilBuild)

set(MAIN_PROJECT OFF)
if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(MAIN_PROJECT ON)
endif()

if (MAIN_PROJECT)
    set(PY_IMAGE_UTIL_EXTENDED_WARNINGS ON)
    set(PY_IMAGE_UTIL_BUILD_TESTS ON)
endif()

option(_PY_IMAGE_UTIL_IMAGE_SANITIZE_FLAGS ON "Internal flag used by the github CI, can be ignored.")

option(PY_IMAGE_UTIL_EXTENDED_WARNINGS OFF "Whether to compile py_img_util with extended warnings such as /Wall /Werror")
option(PY_IMAGE_UTIL_BUILD_TESTS OFF "Whether to build the test suite of py_img_util")

# Add thirdparty libraries
# --------------------------------------------------------------------------

add_subdirectory(thirdparty/pybind11)

if (PY_IMAGE_UTIL_BUILD_TESTS)
    add_library(doctest INTERFACE)
    target_include_directories(doctest SYSTEM INTERFACE thirdparty/doctest/doctest)
endif()

# Projects
# --------------------------------------------------------------------------
add_subdirectory(pybind11_image_util)

if (PY_IMAGE_UTIL_BUILD_TESTS)
    add_subdirectory(test)
endif()