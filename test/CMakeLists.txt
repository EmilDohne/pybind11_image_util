file(GLOB_RECURSE MY_SOURCES CONFIGURE_DEPENDS "src/*.cpp")

# Enable ctest
enable_testing()
add_executable(py_img_util_test ${MY_SOURCES} "main.cpp")

if(MSVC)
    target_compile_options(py_img_util_test PRIVATE /MP /utf-8)
endif()
target_link_libraries(py_img_util_test PRIVATE py_image_util pybind11::pybind11 pybind11::embed pybind11::headers doctest)

include(CTest)
add_test(test_py_img_util py_img_util_test)