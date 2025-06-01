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
#include "validation.h"


namespace NAMESPACE_PY_IMAGE_UTIL
{

	namespace py = pybind11;

	namespace detail
	{

		namespace from_py
		{
			/// Generate a vector from the python np array copying the data into the new container
			/// Generates a flat vector over a 1 or 2d input array. If the incoming data is not contiguous we forcecast
			/// to c-style ordering as well as asserting that the data matches expected_size
			template <typename T>
			std::vector<T> vector(py::array_t<T>& data, size_t expected_width, size_t expected_height)
			{
				size_t expected_size = expected_height * expected_width;
				// This checks that the size matches so we can safely construct assume expected_size
				// is the actual size from this point onwards
				auto shape = detail::shape_from_py_array(data, { 1, 2 }, expected_size);
				detail::check_shape(shape, expected_width, expected_height);
				detail::check_c_style_contiguous(data);
				detail::check_not_null(data);

				// Finally convert the channel to a cpp vector and return
				std::vector<T> data_vec(expected_size);
				std::memcpy(data_vec.data(), data.data(), expected_size * sizeof(T));
				return data_vec;
			}

			/// Generate a view over the data from the python array. The span should only be used
			/// for immediate construction as memory management is not guaranteed. Generates a flat 
			/// view over a 1 or 2d input array. If the incoming data is not contiguous we forcecast
			/// to c-style ordering as well as asserting that the data matches expected_size
			/// 
			/// \param data The python numpy based array we want to create a view over
			/// \param expected_size The expected size in number of elements, NOT bytes.
			template <typename T>
			const std::span<const T> view(py::array_t<T>& data, size_t expected_width, size_t expected_height)
			{
				size_t expected_size = expected_height * expected_width;
				// This checks that the size matches so we can safely construct assume expected_size
				// is the actual size from this point onwards
				auto shape = detail::shape_from_py_array(data, { 1, 2 }, expected_size);
				detail::check_shape(shape, expected_width, expected_height);
				detail::check_c_style_contiguous(data);
				detail::check_not_null(data);

				// Finally convert the channel to a cpp span and return
				std::span<const T> data_span(data.data(), expected_size);
				return data_span;
			}

		} // from_py

		namespace to_py
		{

			/// Generate a py::array_t from std::vector copying the data into 
			/// its internal buffer. This will create a copy of the cpp data.
			/// 
			/// \param data The vector to copy the data from
			/// \param shape The shape to assign to the output container
			template <typename T>
			py::array_t<T> from_vector(const std::vector<T>& data, std::vector<size_t> shape)
			{
				detail::check_cpp_vec_matches_shape(data, shape);
				return py::array_t<T>(shape, data.data());
			}

			/// Generate a py::array_t from std::vector move constructing the data. Will let the python object
			/// take ownership of the data.
			/// 
			/// \param data The vector to copy the data from
			/// \param shape The shape to assign to the output container
			template <typename T>
			py::array_t<T> from_vector(std::vector<T>&& data, std::vector<size_t> shape)
			{
				detail::check_cpp_vec_matches_shape(data, shape);
				auto strides = detail::strides_from_shape<T>(shape);

				// We generate a temporary unique_ptr to assign to the capsule
				// so that the array_t can take ownership over our data
				auto data_raw_ptr = data.data();
				auto data_ptr = std::make_unique<std::vector<T>>(std::move(data));
				auto capsule = py::capsule(data_ptr.get(), [](void* p)
					{
						std::unique_ptr<std::vector<T>>(reinterpret_cast<decltype(data_ptr)::element_type*>(p));
					});
				data_ptr.release();
				// Implicitly convert from py::array to py::array_t as they inherit from one another
				return py::array(shape, strides, data_raw_ptr, capsule);
			}

			/// Generate a py::array_t from std::vector copying the data into 
			/// its internal buffer. This will create a copy of the cpp data.
			/// 
			/// \param data The span to copy the data from
			/// \param shape The shape to assign to the output container
			template <typename T>
			py::array_t<T> from_view(const std::span<const T> data, std::vector<size_t> shape)
			{
				detail::check_cpp_span_matches_shape(data, shape);
				return py::array_t<T>(shape, data.data());
			}

		} // to_py

	} // detail

} // NAMESPACE_PY_IMAGE_UTIL