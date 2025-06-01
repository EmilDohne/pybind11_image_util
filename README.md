# Pybind11 Image Util

This is a standalone, header-only, C++20 library intended to ease swapping between `py::array_t` and `std::vector`/`std::span`
handling them as multidimensional image arrays.

This library handles both validation, casting and conversion to easily and safely convert your image buffers between
python and c++.

## About

This library was initially created as part of the [PhotoshopAPI](https://github.com/EmilDohne/PhotoshopAPI) to simplify 
the conversion of numpy arrays into c++ buffers (and vice versa) ensuring things such as that the shape is correct, that the arrays
are in c-style ordering etc.

It has since become its own standalone library to make it more generally available especially as I have found myself 
needing this more often.

Currently, the library does not handle modifying the python data directly, but instead passes the buffers back and forth.
If this is something you need, feel free to open a ticket or pr.

## Building and linking

Since the `pybind11_image_util` library is header-only you can either copy the `pybind11_image_util/include` directory
into your build tree, or, if you are using CMake, link against the `INTERFACE` target `py_image_util`

A minimal CMakeLists.txt would look like this then:

```CMake
# This is your target
add_library(my_target ...)

add_subdirectory(<path_to_pybind11_image_util>)
target_link_libraries(my_target PRIVATE py_image_util)
```

The library is constantly validated via github actions and compiles under `-Wall -Werror -Wextra` on GCC/Clang and 
`/W4 /WX /w44062 /w44464 /w45264` on MSVC. 


## Usage

### Converting python buffers to std::vector

To get started with doing a conversion from e.g. py->c++ you can use the `py_img_util::from_py_array` function 

```cpp
#include <py_img_util/image.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

auto main() -> int
{
	py::array<uint16_t> my_py_array = ...;
	std::vector<uint16_t> as_cpp_array = py_img_util::from_py_array(py_img_util::tag::vector{}, my_py_array);
	// This will have taken care of:
	// - Ensuring the shape is correct (1 or 2 dims)
	// - Ensuring the py::array is c-style contiguous (will forcecast otherwise)

	// If we already know what we want from the cpp side we can additionally specify the expected size and width
	// and the code will check that this is correct before giving us a flat vector.

	std::vector<uint16_t> as_cpp_array = py_img_util::from_py_array(
		py_img_util::tag::vector{}, 
		my_py_array,
		64, // expected width
		32  // expected height
		);
	
	// This ways we now know for certain that the incoming data was correct in its width and height to catch 
	// errors early.
}
```

### Converting std::vector to py::array

Similarly, you can use the `py_img_util::to_py_array` functions to send data from cpp back to python.

```cpp
#include <py_img_util/image.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

auto main() -> int
{
	std::vector<float> my_cpp_vector = ...;
	size_t image_width = 64;
	size_t image_height = 32;

	// We can now take this and convert it into a py::array using a couple of different overloads:
	// These functions convert a flat 1-dimensional vector into a 2d numpy vector.

	// Convert using a span, copies the data inside into the py::array
	py::array<float> py_array = py_img_util::to_py_array(
		std::span<const T>(my_cpp_vector.begin(), my_cpp_vector.end()),
		image_width,
		image_height
		);

	// Convert using a vector, copies the data inside into the py::array
	py::array<float> py_array = py_img_util::to_py_array(
		my_cpp_vector,
		image_width,
		image_height
		);

	// Convert, moving the data, this does not copy anything and gives python full 
	// control of the data.
	py::array<float> py_array = py_img_util::to_py_array(
		std::move(my_cpp_vector),
		image_width,
		image_height
		);
}
```

### Validation, Utility etc.

If you wish to be more verbose, we expose the `py_img_util::detail` namespace for utility functions and quick validation.

Some of these methods are:

- `py_img_util::detail::shape_from_py_array`
- `py_img_util::detail::strides_from_shape`
- `py_img_util::detail::check_shape`
- `py_img_util::detail::check_c_style_contiguous`
