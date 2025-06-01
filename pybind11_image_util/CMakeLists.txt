project(PyImgUtil)

add_library(py_image_util INTERFACE)
target_include_directories(py_image_util INTERFACE "include")
target_link_libraries(py_image_util INTERFACE pybind11::pybind11 pybind11::headers)

if (MSVC)
	target_compile_options(py_image_util INTERFACE /utf-8 /MP /DNOMINMAX)
endif()


# Crank up warning levels on both MSVC, Clang and GCC
if (PY_IMAGE_UTIL_EXTENDED_WARNINGS)
	if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
	target_compile_options(
		py_image_util 
		INTERFACE 
		-Wall 
		-Werror 
		-Wextra
	)
	elseif(PY_IMAGE_UTIL_EXTENDED_WARNINGS MATCHES "MSVC")
	target_compile_options(
		py_image_util 
		INTERFACE 
		/W4 
		/WX 
		/w44062 
		/w44464 
		/w45264
	)
	endif()
endif()

# Enable sanitizers unless on macOS (not supported) or Windows (github runners run out of memory). 
# These are for our CI runs only and should be ignored by users.
if (_PY_IMAGE_UTIL_IMAGE_SANITIZE_FLAGS AND NOT APPLE)
	if (CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
		target_compile_options(py_image_util INTERFACE -fsanitize=address,leak,undefined)
		target_link_options(py_image_util INTERFACE -fsanitize=address,leak,undefined)
	endif()
endif()