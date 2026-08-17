#include <cstddef>
#include <cstring>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "fft.hpp"

namespace py = pybind11;

void fft_wrapper(py::array_t<double> real_input, py::array_t<double> imag_input, py::array_t<double> real_output, py::array_t<double> imag_output) {
    
    auto real_input_buffer = real_input.request();
    auto imag_input_buffer = imag_input.request();
    auto real_output_buffer = real_output.request();
    auto imag_output_buffer = imag_output.request();

    auto* real_input_data = static_cast<double*>(real_input_buffer.ptr);
    auto* imag_input_data = static_cast<double*>(imag_input_buffer.ptr);
    auto* real_output_data = static_cast<double*>(real_output_buffer.ptr);
    auto* imag_output_data = static_cast<double*>(imag_output_buffer.ptr);

    std::size_t n = real_input_buffer.size;

    fft(
        real_input_data,
        imag_input_data,
        real_output_data,
        imag_output_data,
        n
    );
}

py::tuple fft_analysis_wrapper(py::array_t<double> real_input, py::array_t<double> imag_input, double dt, double threshold) {
    auto real_input_buffer = real_input.request();
    auto imag_input_buffer = imag_input.request();

    std::size_t n = real_input_buffer.size;

    auto* real_input_data = static_cast<double*>(real_input_buffer.ptr);

    auto* imag_input_data = static_cast<double*>(imag_input_buffer.ptr);

    AnalysisResult result = fftAnalysis(
        real_input_data,
        imag_input_data,
        n,
        dt,
        threshold
    );

    py::array_t<double> frequencies(result.frequencies.size());
    py::array_t<double> amplitudes(result.amplitudes.size());
    py::array_t<double> phases(result.phases.size());

    std::memcpy(
        frequencies.mutable_data(),
        result.frequencies.data(),
        result.frequencies.size() * sizeof(double)
    );

    std::memcpy(
        amplitudes.mutable_data(),
        result.amplitudes.data(),
        result.amplitudes.size() * sizeof(double)
    );

    std::memcpy(
        phases.mutable_data(),
        result.phases.data(),
        result.phases.size() * sizeof(double)
    );

    return py::make_tuple(
        frequencies,
        amplitudes,
        phases
    );
}

PYBIND11_MODULE(fft_cpp, m)
{
    m.def("fft", &fft_wrapper);
    m.def("fft_analysis", &fft_analysis_wrapper);
}
