#pragma once

#include <pybind11/embed.h>
#include <doctest.h>
#include <format>

namespace py = pybind11;

namespace test_utils 
{

    /// Wrapper to keep the scoped interpreter alive during stack unwinding, this is because doctest calls e.what()
    /// during exception handling, however this is not supported with py::scoped_interpreter as it needs the python
    /// interpreter alive while calling e.what(). 
    /// \param fn The function to emplace.
    inline void with_python(const std::function<void()>& fn) 
    {
        py::scoped_interpreter guard{};
        try 
        {
            fn();
        }
        catch (const std::exception& e) 
        {
            FAIL_CHECK(std::format("Exception caught in test: {}", e.what()));
        }
    }

}