// Copyright Contributors to the pybind11_image_util project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/EmilDohne/pybind11_image_util

#pragma once

#include <format>
#include <vector>
#include <unordered_map>
#include <string>
#include <span>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "macros.h"


namespace NAMESPACE_PY_IMAGE_UTIL
{
	namespace py = pybind11;

	namespace detail
	{

		/// Generate a shape array from the py::array_t checking at runtime whether the shape fits into the allowed dims and matches total_size
		///
		/// \param data The data to extract the shape information from
		/// \param allowed_dims The number of dimensions that are allowed. Could e.g. be {1, 2} to allow one and two dimensional arrays
		/// \param total_size The total size of the expected data, if the dimensions do not hold this amount of data we throw a value_error
		/// 
		/// \return The shape as a cpp array
		template <typename T>
		std::vector<size_t> shape_from_py_array(const py::array_t<T>& data, const std::vector<size_t> allowed_dims, size_t total_size)
		{
			std::vector<size_t> shape;
			size_t sum = 1;
			for (py::ssize_t i = 0; i < data.ndim(); ++i)
			{
				shape.push_back(data.shape(i));
				sum *= data.shape(i);
			}

			// Check that the shape is within the allowed dimensions
			if (std::find(allowed_dims.begin(), allowed_dims.end(), shape.size()) == allowed_dims.end())
			{
				std::string error_msg = "Invalid number of dimensions received, array must have one of the following number of dimensions: { ";
				for (size_t i = 0; i < allowed_dims.size() - 1; ++i)
				{
					error_msg += std::to_string(allowed_dims[i]) + ", ";
				}
				error_msg += std::to_string(allowed_dims.back());
				error_msg += " }. Instead got: " + std::to_string(shape.size());
				throw py::value_error(error_msg);
			}

			if (sum != total_size)
			{
				throw py::value_error(
					std::format(
						"Invalid array size received, expected {:L} but instead got {:L}. This should match the layers'"
						" height and width and if passing multiple channels also the number of channels.",
						total_size, sum
					)
				);
			}
			return shape;
		}

		/// Generate C-style strides from a shape vector for a given data type.
		/// 
		/// \tparam T The type of the data the strides apply to.
		/// \param shape The shape vector for which to calculate strides.
		/// \return A vector of strides in bytes corresponding to each dimension.
		template <typename T>
		std::vector<size_t> strides_from_shape(std::vector<size_t> shape)
		{
			std::vector<size_t> strides(shape.size());

			if (shape.empty())
			{
				return strides; // Return empty strides if shape is empty
			}

			size_t stride = sizeof(T);  // Start with the size of the element in bytes
			for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i)
			{
				strides[i] = stride;
				stride *= shape[i];  // Move to the next dimension stride in terms of element count
			}
			return strides;
		}

		/// Validate that a 1D shape vector matches the expected total number of elements (height * width).
		/// 
		/// \param shape A shape vector expected to have one dimension.
		/// \param expected_width The expected width of the image.
		/// \param expected_height The expected height of the image.
		/// \throws py::value_error if the shape does not match the expected size.
		inline void check_shape_1d(std::vector<size_t> shape, size_t expected_width, size_t expected_height)
		{
			assert(shape.size() == 1);
			if (shape[0] != expected_height * expected_width)
			{
				throw py::value_error(
					std::format(
						"Invalid 1st dimension size encounted, expected {:L} but instead got {:L}",
						expected_height * expected_width, shape[0]
					)
				);
			}
		}

		/// Validate that a 2D shape vector matches the expected height and width.
		/// 
		/// \param shape A shape vector expected to have two dimensions: {height, width}.
		/// \param expected_width The expected width of the image.
		/// \param expected_height The expected height of the image.
		/// \throws py::value_error if the shape does not match the expected height and width.
		inline void check_shape_2d(std::vector<size_t> shape, size_t expected_width, size_t expected_height)
		{
			assert(shape.size() == 2);
			if (shape[0] != expected_height)
			{
				throw py::value_error(
					std::format(
						"Invalid 1st dimension size encounted, expected {:L} but instead got {:L}."
						" This number should represent the images' height",
						expected_height, shape[0]
					)
				);
			}
			if (shape[1] != expected_width)
			{
				throw py::value_error(
					std::format(
						"Invalid 2nd dimension size encounted, expected {:L} but instead got {:L}."
						" This number should represent the images' width",
						expected_width, shape[1]
					)
				);
			}
		}

		/// Validate that a 3D shape vector matches the expected number of channels, height, and width.
		/// 
		/// \param shape A shape vector expected to have three dimensions: {channels, height, width}.
		/// \param expected_channels The expected number of image channels.
		/// \param expected_width The expected width of the image.
		/// \param expected_height The expected height of the image.
		/// \throws py::value_error if any dimension does not match its expected value.
		inline void check_shape_3d(std::vector<size_t> shape, size_t expected_channels, size_t expected_width, size_t expected_height)
		{
			assert(shape.size() == 3);
			if (shape[0] != expected_channels)
			{
				throw py::value_error(
					std::format(
						"Invalid 1st dimension size encounted, expected {:L} but instead got {:L}."
						" This number should represent the images' number of channels",
						expected_channels, shape[0]
					)
				);
			}
			if (shape[1] != expected_height)
			{
				throw py::value_error(
					std::format(
						"Invalid 2nd dimension size encounted, expected {:L} but instead got {:L}."
						" This number should represent the images' height",
						expected_height, shape[1]
					)
				);
			}
			if (shape[2] != expected_width)
			{
				throw py::value_error(
					std::format(
						"Invalid 3rd dimension size encounted, expected {:L} but instead got {:L}."
						" This number should represent the images' width",
						expected_width, shape[2]
					)
				);
			}
		}

		/// Check that a shape vector matches one of the supported formats (1D, 2D, or 3D) and that its dimensions match expectations.
		/// 
		/// \param shape The shape vector to validate.
		/// \param expected_width The expected image width.
		/// \param expected_height The expected image height.
		/// \param expected_channels The expected number of channels (default is 1).
		/// \throws py::value_error if shape does not conform to expected dimensions or sizes.
		inline void check_shape(std::vector<size_t> shape, size_t expected_width, size_t expected_height, size_t expected_channels = 1)
		{
			if (shape.size() == 1)
			{
				check_shape_1d(shape, expected_width, expected_height);
			}
			else if (shape.size() == 2)
			{
				check_shape_2d(shape, expected_width, expected_height);
			}
			else if (shape.size() == 3)
			{
				check_shape_3d(shape, expected_channels, expected_width, expected_height);
			}
			else
			{
				throw py::value_error(
					std::format(
						"Invalid number of array dimensions encountered, expected 1, 2 or 3 but instead got {}", shape.size()
					)
				);
			}
		}

		/// Ensure the provided Python array is C-contiguous in memory. If not, convert it in-place.
		/// 
		/// \tparam T The data type stored in the array.
		/// \param data The Python array to check and potentially convert. This operation modifies the input array.
		template <typename T>
		void check_c_style_contiguous(py::array_t<T>& data)
		{
			if (py::detail::npy_api::constants::NPY_ARRAY_C_CONTIGUOUS_ != (data.flags() & py::detail::npy_api::constants::NPY_ARRAY_C_CONTIGUOUS_))
			{
				data = data.template cast<py::array_t<T, py::array::c_style | py::array::forcecast>>();
			}
		}

		/// Validate that the given Python array is not null.
		/// 
		/// \tparam T The data type stored in the array.
		/// \param data The Python array to check.
		/// \throws py::value_error if the array is null.
		template <typename T>
		void check_not_null(const py::array_t<T>& data)
		{
			if (data.data() == nullptr)
			{
				throw py::value_error(
					"Python numpy array passed to function resolves to nullptr. If you believe this to be a mistake" \
					" please open a ticket on the projects' github page."
				);
			}
		}

		/// Validate that the size of a C++ span matches the total size implied by the shape vector.
		/// 
		/// \tparam T The data type of the span.
		/// \param data The C++ span to check.
		/// \param shape The shape vector whose product must equal the span's size.
		/// \throws py::value_error if the span size does not match the shape's product.
		template <typename T>
		void check_cpp_span_matches_shape(const std::span<const T> data, std::vector<size_t> shape)
		{
			if (shape.empty())
			{
				throw py::value_error("Passed an empty shape to check_cpp_span_matches_shape");
			}

			size_t sum = 1;
			for (const auto item : shape)
			{
				sum *= item;
			}

			if (sum != data.size())
			{
				std::string error_msg = "Invalid array dimension received: { ";
				for (size_t i = 0; i < shape.size() - 1; ++i)
				{
					error_msg += std::to_string(shape[i]) + ", ";
				}
				error_msg += std::to_string(shape.back());
				error_msg += " }. Expected these to sum up to " + std::format("{:L}", data.size()) + " but they instead sum up to ";
				error_msg += std::format("{:L}", sum) + ". This could be due to the layers' width and height not matching the channel data";
				throw py::value_error(error_msg);
			}
		}

		/// Validate that the size of a C++ vector matches the total size implied by the shape vector.
		/// 
		/// \tparam T The data type of the vector.
		/// \param data The C++ vector to check.
		/// \param shape The shape vector whose product must equal the vector's size.
		/// \throws py::value_error if the vector size does not match the shape's product.
		template <typename T>
		void check_cpp_vec_matches_shape(const std::vector<T>& data, std::vector<size_t> shape)
		{
			std::span<const T> data_span(data.data(), data.size());
			check_cpp_span_matches_shape(data_span, shape);
		}
	
	} // detail

} // NAMESPACE_PY_IMAGE_UTIL