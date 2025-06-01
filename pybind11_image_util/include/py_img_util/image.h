// Copyright Contributors to the pybind11_image_util project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/EmilDohne/pybind11_image_util

#pragma once

#include <vector>
#include <unordered_map>
#include <span>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "macros.h"
#include "detail.h"


namespace NAMESPACE_PY_IMAGE_UTIL
{

	namespace py = pybind11;

	/// Keys for tag dispatching
	namespace tag
	{
		struct mapping {};
		struct view {};
		struct vector {};
	}


	/// \brief Generate a view over the py::array without copying.
	///
	/// \note Only use this function when the array data is guaranteed to outlive the view.
	/// It is ideal for one-off computations, and the span should not be retained.
	///
	/// The input array must be one- or two-dimensional:
	/// - If 1D: the size must be `expected_width * expected_height`
	/// - If 2D: shape must be `[expected_height, expected_width]`
	///
	/// \tparam T Type of array element
	/// \param _ Tag for view dispatch
	/// \param data Python array to view; converted to C-contiguous layout if needed
	/// \param expected_width Expected width (number of columns)
	/// \param expected_height Expected height (number of rows)
	/// \return A const span over the flattened data
	template <typename T>
	const std::span<const T> from_py_array(
		[[maybe_unused]] tag::view _,
		py::array_t<T>& data,
		size_t expected_width,
		size_t expected_height
	)
	{
		return detail::from_py::view(data, expected_width, expected_height);
	}

	/// \brief Generate a view over the py::array without copying.
	///
	/// \note Only use this function when the array data is guaranteed to outlive the view.
	/// It is ideal for one-off computations, and the span should not be retained.
	///
	/// \tparam T Type of array element
	/// \param _ Tag for view dispatch
	/// \param data Python array to view; converted to C-contiguous layout if needed
	/// \return A const span over the flattened data
	template <typename T>
	const std::span<const T> from_py_array(
		[[maybe_unused]] tag::view _,
		py::array_t<T>& data
	)
	{
		size_t expected_width = 0;
		auto shape = detail::shape_from_py_array(data, { 1, 2 }, data.size());
		size_t expected_height = shape[0];
		if (shape.size() == 1)
		{
			expected_width = 1;
		}
		else
		{
			expected_width = shape[1];
		}

		return detail::from_py::view(data, expected_width, expected_height);
	}


	/// \brief Convert a py::array into a std::vector with shape validation.
	///
	/// The input array must be one- or two-dimensional:
	/// - If 1D: the size must be `expected_width * expected_height`
	/// - If 2D: shape must be `[expected_height, expected_width]`
	///
	/// \tparam T Type of array element
	/// \param _ Tag for vector dispatch
	/// \param data Input array to convert; will ensure C-contiguity
	/// \param expected_width Width to validate (columns)
	/// \param expected_height Height to validate (rows)
	/// \return Flattened std::vector<T> with row-major order
	template <typename T>
	std::vector<T> from_py_array(
		[[maybe_unused]] tag::vector _,
		py::array_t<T>& data,
		size_t expected_width,
		size_t expected_height)
	{
		return detail::from_py::vector(data, expected_width, expected_height);
	}

	/// \brief Convert a py::array into a std::vector with shape validation.
	///
	/// The input array must be one- or two-dimensional.
	///
	/// \tparam T Type of array element
	/// \param _ Tag for vector dispatch
	/// \param data Input array to convert; will ensure C-contiguity
	/// \return Flattened std::vector<T> with row-major order
	template <typename T>
	std::vector<T> from_py_array(
		[[maybe_unused]] tag::vector _,
		py::array_t<T>& data
	)
	{
		size_t expected_width = 0;
		auto shape = detail::shape_from_py_array(data, { 1, 2 }, data.size());
		size_t expected_height = shape[0];
		if (shape.size() == 1)
		{
			expected_width = 1;
		}
		else
		{
			expected_width = shape[1];
		}

		return detail::from_py::vector(data, expected_width, expected_height);
	}


	/// \brief Convert a span to a 2D numpy array (py::array_t).
	///
	/// The output array will have shape `[height, width]`.
	/// The span must be in row-major order.
	///
	/// \tparam T Type of the data
	/// \param data Input span to copy into numpy array
	/// \param width Number of columns in the output
	/// \param height Number of rows in the output
	/// \return New py::array_t<T> with copied data
	template <typename T>
	py::array_t<T> to_py_array(const std::span<const T> data, size_t width, size_t height)
	{
		std::vector<size_t> shape{ height, width };
		return detail::to_py::from_view(data, shape);
	}


	/// \brief Convert a std::vector<T> to a 2D py::array_t with shape [height, width].
	///
	/// \tparam T Data type
	/// \param data Vector containing the row-major data
	/// \param width Number of columns
	/// \param height Number of rows
	/// \return py::array_t<T> sharing a copy of the vector data
	template <typename T>
	py::array_t<T> to_py_array(const std::vector<T>& data, size_t width, size_t height)
	{
		std::vector<size_t> shape{ height, width };
		return detail::to_py::from_vector(data, shape);
	}

	/// \brief Move a std::vector<T> into a new py::array_t<T> with shape [height, width].
	///
	/// \tparam T Data type
	/// \param data Vector (rvalue) to move into the array
	/// \param width Number of columns
	/// \param height Number of rows
	/// \return py::array_t<T> taking ownership of the data
	template <typename T>
	py::array_t<T> to_py_array(std::vector<T>&& data, size_t width, size_t height)
	{
		std::vector<size_t> shape{ height, width };
		return detail::to_py::from_vector(std::move(data), shape);
	}

} // NAMESPACE_PY_IMAGE_UTIL