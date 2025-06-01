#include "doctest.h"

#include <vector>
#include <map>
#include <unordered_map>
#include <span>

#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include "py_img_util/detail.h"
#include "test_utils.h"

namespace py = pybind11;
using namespace NAMESPACE_PY_IMAGE_UTIL::detail;


// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("from_py::vector handles 1D data correctly")
{
    test_utils::with_python([]()
        {
            // Create a C-contiguous 1D array of length 6
            py::array_t<int> arr(std::vector<int>{ 6 });
            for (py::ssize_t i = 0; i < 6; ++i)
            {
                arr.mutable_at(i) = static_cast<int>(i + 1);
            }

            auto vec = from_py::vector<int>(arr, 6, 1);
            CHECK(vec == std::vector<int>{1, 2, 3, 4, 5, 6});
        });
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("from_py::vector handles 2D data correctly")
{
    test_utils::with_python([]()
        {
            // Create a 2×3 array:
            py::array_t<float> arr({ 2, 3 });
            float counter = 1.0f;
            for (py::ssize_t i = 0; i < 2; ++i)
            {
                for (py::ssize_t j = 0; j < 3; ++j)
                {
                    arr.mutable_at(i, j) = counter++;
                }
            }

            auto vec = from_py::vector<float>(arr, 3, 2);
            // Flat copy should be row-major: [1,2,3,4,5,6]
            CHECK(vec == std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
        });
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("from_py::vector enforces C-contiguity (transposed input)")
{
    test_utils::with_python([]()
        {
            // Build a C-contiguous 2×2 array, then transpose it so it's Fortran-ordered
            py::array_t<double> base({ 2, 2 });
            base.mutable_at(0, 0) = 1.0;
            base.mutable_at(0, 1) = 2.0;
            base.mutable_at(1, 0) = 3.0;
            base.mutable_at(1, 1) = 4.0;
            // Transpose → non-C-contiguous
            py::array_t<double> transposed = base.attr("T").cast<py::array_t<double>>();

            // Check that 'transposed' is not C-contiguous initially
            auto info = transposed.request();
            bool is_c = (transposed.flags() & py::array::c_style) != 0;
            REQUIRE_FALSE(is_c);

            // Now call from_py::vector; it must forcecast to C-order and succeed
            auto vec = from_py::vector<double>(transposed, 2, 2);
            CHECK(vec == std::vector<double>{1.0, 3.0, 2.0, 4.0});
        });
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("from_py::vector throws on size mismatch")
{
    test_utils::with_python([]()
        {
            // 2×2 array but expected dimensions imply size 6
            py::array_t<int> arr({ 2, 2 });
            CHECK_THROWS_AS(from_py::vector<int>(arr, 3, 2), py::value_error);
        });
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("from_py::view handles 2D data and returns a valid span")
{
    test_utils::with_python([]()
        {
            std::vector<int> data = { 10, 20, 30, 40, 50, 60 };
            py::array_t<int> arr({ 2, 3 }, data.data());

            auto span = from_py::view<int>(arr, 3, 2);
            CHECK(span.size() == 6);
            CHECK(span[0] == 10);
            CHECK(span[3] == 40);
        });
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("from_py::view enforces C-contiguity for non-contiguous input")
{
    test_utils::with_python([]()
        {
            // Create a 3×2 array and transpose it
            py::array_t<int> base({ 3, 2 });
            int count = 1;
            for (py::ssize_t i = 0; i < 3; ++i)
            {
                for (py::ssize_t j = 0; j < 2; ++j)
                {
                    base.mutable_at(i, j) = count++;
                }
            }

            py::array_t<int> transposed = base.attr("T").cast<py::array_t<int>>();
            auto info = transposed.request();
            bool is_c = (transposed.flags() & py::array::c_style) != 0;
            REQUIRE_FALSE(is_c);

            auto span = from_py::view<int>(transposed, 3, 2);
            // After forcing C-order, the span should reflect row-major flattening:
            // Original 'base' was [[1,2],[3,4],[5,6]]; transpose gives [[1,3,5],[2,4,6]]
            // Forcing C-order flattens as [1,3,5,2,4,6]
            CHECK(span[0] == 1);
            CHECK(span[5] == 6);
        });
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("from_py::view throws on dimension mismatch")
{
    test_utils::with_python([]()
        {
            py::array_t<float> arr({ 4, 4 });
            // expected dims correspond to size 12
            CHECK_THROWS_AS(from_py::view<float>(arr, 3, 4), py::value_error);
        });
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("to_py::from_vector (copy) creates a new NumPy array")
{
    test_utils::with_python([]()
        {
            std::vector<double> data = { 2.2, 3.3, 4.4, 5.5 };
            std::vector<size_t> shape = { 2, 2 };
            auto arr = to_py::from_vector<double>(data, shape);

            CHECK(arr.ndim() == 2);
            CHECK(arr.shape(0) == 2);
            CHECK(arr.shape(1) == 2);
            // Ensure contents match
            auto ptr = static_cast<double*>(arr.mutable_data());
            CHECK(ptr[0] == doctest::Approx(2.2));
            CHECK(ptr[3] == doctest::Approx(5.5));

            // Modifying the original vector should NOT affect the array (copy semantic)
            data[0] = -1.0;
            CHECK(ptr[0] == doctest::Approx(2.2));
        });
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("to_py::from_vector (move) transfers ownership correctly")
{
    test_utils::with_python([]()
        {
            std::vector<int> data = { 7, 8, 9, 10, 11, 12 };
            std::vector<size_t> shape = { 3, 2 };
            // Move the vector into from_vector
            auto arr = to_py::from_vector<int>(std::move(data), shape);

            CHECK(arr.ndim() == 2);
            CHECK(arr.shape(0) == 3);
            CHECK(arr.shape(1) == 2);
            // Data should still be intact in the NumPy array
            auto ptr = static_cast<int*>(arr.mutable_data());
            CHECK(ptr[0] == 7);
            CHECK(ptr[5] == 12);

            // The original 'data' vector must now be empty (moved-from)
            CHECK(data.empty());
        });
}

// -----------------------------------------------------------------------------------
// -----------------------------------------------------------------------------------
TEST_CASE("to_py::from_view creates correct NumPy array from span")
{
    test_utils::with_python([]()
        {
            std::array<float, 6> buffer = { 0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f };
            std::span<const float> span(buffer.data(), buffer.size());
            std::vector<size_t> shape = { 2, 3 };
            auto arr = to_py::from_view<float>(span, shape);

            CHECK(arr.ndim() == 2);
            CHECK(arr.shape(0) == 2);
            CHECK(arr.shape(1) == 3);
            auto ptr = static_cast<float*>(arr.mutable_data());
            CHECK(ptr[2] == doctest::Approx(2.5f));
            CHECK(ptr[5] == doctest::Approx(5.5f));
        });
}