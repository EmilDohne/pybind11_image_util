#include "doctest.h"

#include <vector>
#include <map>
#include <unordered_map>
#include <span>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/detail/common.h>

#include "py_img_util/validation.h"

#include "test_utils.h"

namespace py = pybind11;
using namespace NAMESPACE_PY_IMAGE_UTIL::detail;


// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("shape_from_py_array passes for 1D array")
{
    test_utils::with_python([]()
        {
            // 1D array of length 6
            py::array_t<float> arr(static_cast<py::ssize_t>(6));
            auto shape = shape_from_py_array<float>(arr, { 1, 2, 3 }, 6);
            CHECK(shape == std::vector<size_t>{6});
        });
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("shape_from_py_array passes for 3D array")
{
    test_utils::with_python([]()
        {
            // 3 channels, 2x2 image
            py::array_t<float> arr({ 3, 2, 2 });
            auto shape = shape_from_py_array<float>(arr, { 1, 2, 3 }, 12);
            CHECK(shape == std::vector<size_t>{3, 2, 2});
        });
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("shape_from_py_array throws when total_size mismatches for 3D")
{
    test_utils::with_python([]()
        {
            // 3 channels, 2x2 image, but expect total_size = 10
            py::array_t<float> arr({ 3, 2, 2 });
            CHECK_THROWS_AS(shape_from_py_array<float>(arr, { 1, 2, 3 }, 10), py::value_error);
        });
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("shape_from_py_array throws for unsupported dims (4D)")
{
    test_utils::with_python([]()
        {
            // 4D: unsupported by allowed_dims = {1,2,3}
            py::array_t<double> arr({ 2, 2, 2, 2 });
            CHECK_THROWS_AS(shape_from_py_array<double>(arr, { 1, 2, 3 }, 16), py::value_error);
        });
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("strides_from_shape computes correct strides for 1D")
{
    std::vector<size_t> shape = { 5 };
    auto strides = strides_from_shape<double>(shape);
    CHECK(strides == std::vector<size_t>{sizeof(double)});
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("strides_from_shape computes correct strides for 2D")
{
    std::vector<size_t> shape = { 4, 3 };
    // expected: stride[1] = sizeof(float), stride[0] = 3 * sizeof(float)
    auto strides = strides_from_shape<float>(shape);
    CHECK(strides == std::vector<size_t>{3 * sizeof(float), sizeof(float)});
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("check_shape_1d passes for correct 1D length")
{
    std::vector<size_t> shape = { 8 };
    // expected_width * expected_height = 2 * 4 = 8
    CHECK_NOTHROW(check_shape_1d(shape, 2, 4));
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("check_shape_1d throws for incorrect 1D length")
{
    std::vector<size_t> shape = { 9 };
    // expected_width * expected_height = 2 * 4 = 8
    CHECK_THROWS_AS(check_shape_1d(shape, 2, 4), py::value_error);
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("check_shape_2d passes for correct 2D dims")
{
    std::vector<size_t> shape = { 5, 7 };
    CHECK_NOTHROW(check_shape_2d(shape, 7, 5));
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("check_shape_2d throws for incorrect first dim")
{
    std::vector<size_t> shape = { 4, 7 };
    // expected first dim = 5
    CHECK_THROWS_AS(check_shape_2d(shape, 7, 5), py::value_error);
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("check_shape_2d throws for incorrect second dim")
{
    std::vector<size_t> shape = { 5, 6 };
    // expected second dim = 7
    CHECK_THROWS_AS(check_shape_2d(shape, 7, 5), py::value_error);
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("check_shape_3d passes for correct 3D dims")
{
    std::vector<size_t> shape = { 3, 5, 7 };
    CHECK_NOTHROW(check_shape_3d(shape, 3, 7, 5));
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("check_shape_3d throws for incorrect channels")
{
    std::vector<size_t> shape = { 2, 5, 7 };
    // expected_channels = 3
    CHECK_THROWS_AS(check_shape_3d(shape, 3, 7, 5), py::value_error);
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("check_shape_3d throws for incorrect height")
{
    std::vector<size_t> shape = { 3, 4, 7 };
    // expected_height = 5
    CHECK_THROWS_AS(check_shape_3d(shape, 3, 7, 5), py::value_error);
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("check_shape_3d throws for incorrect width")
{
    std::vector<size_t> shape = { 3, 5, 6 };
    // expected_width = 7
    CHECK_THROWS_AS(check_shape_3d(shape, 3, 7, 5), py::value_error);
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("check_shape throws for dimension count > 3")
{
    std::vector<size_t> shape = { 1, 2, 3, 4 };
    CHECK_THROWS_AS(check_shape(shape, 4, 1, 1), py::value_error);
}


// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("check_c_style_contiguous converts a transposed array to C order")
{
    test_utils::with_python([]()
        {

            // 1) Create a C-contiguous 3×4 float array:
            py::array_t<float> base({ 3, 4 });
            for (py::ssize_t i = 0; i < 3; ++i)
            {
                for (py::ssize_t j = 0; j < 4; ++j)
                {
                    base.mutable_at(i, j) = static_cast<float>(i * 4 + j);
                }
            }

            // 2) Transpose it via the .attr("T") call;
            //    the resulting array is *not* C-contiguous:
            py::array_t<float> transposed = base.attr("T").cast<py::array_t<float>>();

            // Verify that it is indeed not C-contiguous (so that our test is valid):
            auto info_before = transposed.request();
            bool is_c_before = (transposed.flags() & py::array::c_style) != 0;
            REQUIRE_FALSE(is_c_before);

            // 3) Call check_c_style_contiguous, which should replace 'transposed' with a new C-contiguous array:
            check_c_style_contiguous(transposed);

            // 4) After conversion, the array must be C-contiguous:
            bool is_c_after = (transposed.flags() & py::array::c_style) != 0;
            CHECK(is_c_after);

            // 5) As a final sanity check: in true C-order, the last‐dimension stride (stride for index 1) = sizeof(float).
            auto info_after = transposed.request();
            size_t stride_last = static_cast<size_t>(info_after.strides[1]); // in bytes
            CHECK(stride_last == sizeof(float));
        });
}


// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("check_cpp_span_matches_shape passes for matching span")
{
    std::vector<double> vec(12, 1.0);
    std::span<const double> span(vec.data(), vec.size());
    std::vector<size_t> shape = { 3, 4 };
    CHECK_NOTHROW(check_cpp_span_matches_shape(span, shape));
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("check_cpp_span_matches_shape throws for mismatched span")
{
    std::vector<double> vec(11, 1.0);
    std::span<const double> span(vec.data(), vec.size());
    std::vector<size_t> shape = { 3, 4 };
    CHECK_THROWS_AS(check_cpp_span_matches_shape(span, shape), py::value_error);
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("check_cpp_vec_matches_shape passes for matching vector")
{
    std::vector<int> data(8, 0);
    std::vector<size_t> shape = { 2, 2, 2 };
    CHECK_NOTHROW(check_cpp_vec_matches_shape(data, shape));
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("check_cpp_vec_matches_shape throws for zero-dimension shape vector")
{
    std::vector<int> data; // size 0
    std::vector<size_t> shape = {};
    CHECK_THROWS_AS(check_cpp_vec_matches_shape(data, shape), py::value_error);
}