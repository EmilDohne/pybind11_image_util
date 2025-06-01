#include "doctest.h"

#include <vector>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include "py_img_util/detail.h"
#include "py_img_util/image.h"

namespace py = pybind11;
using namespace NAMESPACE_PY_IMAGE_UTIL;


// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("from_py_array::view returns span with correct values") 
{
    py::scoped_interpreter guard{};
    std::vector<int> buffer{ 1, 2, 3, 4, 5, 6 };
    py::array_t<int> arr({ 2, 3 }, buffer.data());

    auto span = from_py_array<int>(tag::view{}, arr, 3, 2);

    CHECK(span.size() == 6);
    CHECK(span.front() == 1);
    CHECK(span.back() == 6);
}


// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("from_py_array::vector copies numpy data correctly") 
{
    py::scoped_interpreter guard{};
    std::vector<int> buffer{ 1, 2, 3, 4, 5, 6 };
    py::array_t<int> arr({ 2, 3 }, buffer.data());

    auto vec = from_py_array<int>(tag::vector{}, arr, 3, 2);

    CHECK(vec == buffer);
}


// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("from_py_array::view returns span with correct values, no expected dims")
{
    py::scoped_interpreter guard{};
    std::vector<int> buffer{ 1, 2, 3, 4, 5, 6 };
    py::array_t<int> arr({ 2, 3 }, buffer.data());

    auto span = from_py_array<int>(tag::view{}, arr);

    CHECK(span.size() == 6);
    CHECK(span.front() == 1);
    CHECK(span.back() == 6);
}


// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("from_py_array::vector copies numpy data correctly, no expected dims")
{
    py::scoped_interpreter guard{};
    std::vector<int> buffer{ 1, 2, 3, 4, 5, 6 };
    py::array_t<int> arr({ 2, 3 }, buffer.data());

    auto vec = from_py_array<int>(tag::vector{}, arr);

    CHECK(vec == buffer);
}



// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("to_py_array from std::span yields correct shape and values") 
{
    py::scoped_interpreter guard{};
    std::vector<int> buffer{ 10, 20, 30, 40, 50, 60 };
    std::span<const int> span(buffer);
    auto arr = to_py_array(span, 3, 2);

    CHECK(arr.ndim() == 2);
    CHECK(arr.shape(0) == 2);
    CHECK(arr.shape(1) == 3);
    auto r = arr.unchecked<2>();
    CHECK(r(0, 0) == 10);
    CHECK(r(1, 2) == 60);
}


// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("to_py_array from std::vector yields identical data") 
{
    py::scoped_interpreter guard{};
    std::vector<int> vec{ 100, 200, 300, 400 };
    auto arr = to_py_array(vec, 2, 2);

    CHECK(arr.shape(0) == 2);
    CHECK(arr.shape(1) == 2);
    auto r = arr.unchecked<2>();
    CHECK(r(0, 0) == 100);
    CHECK(r(1, 1) == 400);
}


// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("to_py_array from moved std::vector behaves correctly") 
{
    py::scoped_interpreter guard{};
    std::vector<int> vec{ 9, 8, 7, 6 };
    auto arr = to_py_array(std::move(vec), 2, 2);

    CHECK(arr.shape(0) == 2);
    CHECK(arr.shape(1) == 2);
    auto r = arr.unchecked<2>();
    CHECK(r(0, 0) == 9);
    CHECK(r(1, 1) == 6);
}