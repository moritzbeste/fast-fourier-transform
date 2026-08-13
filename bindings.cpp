#include <cstddef>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "fft.hpp"

namespace py = pybind11;

void fft_wrapper(
    py::array_t<double> real_input,
    py::array_t<double> imag_input,
    py::array_t<double> real_output,
    py::array_t<double> imag_output) {
    auto real_input_buffer = real_input.request();
    auto imag_input_buffer = imag_input.request();
    auto real_output_buffer = real_output.request();
    auto imag_output_buffer = imag_output.request();

    auto* real_input_data =
        static_cast<double*>(real_input_buffer.ptr);
    
    auto* imag_input_data =
        static_cast<double*>(imag_input_buffer.ptr);

    auto* real_output_data =
        static_cast<double*>(real_output_buffer.ptr);

    auto* imag_output_data =
        static_cast<double*>(imag_output_buffer.ptr);

    std::size_t n = real_input_buffer.size;

    fft(
        real_input_data,
        imag_input_data,
        real_output_data,
        imag_output_data,
        n
    );
}

PYBIND11_MODULE(fft_cpp, m)
{
    m.def("fft", &fft_wrapper);
}



// #include <cstddef>

// #include <pybind11/pybind11.h>
// #include <pybind11/numpy.h>

// #include "fft.hpp"

// namespace py = pybind11;

// py::array_t<double> fft_wrapper(
//     py::array_t<double> input,
//     py::array_t<double> real_output,
//     py::array_t<double> imag_output,
//     double sample_rate
// ) {
//     auto input_buffer = input.request();
//     auto real_buffer = real_output.request();
//     auto imag_buffer = imag_output.request();

//     auto* input_data =
//         static_cast<double*>(input_buffer.ptr);

//     auto* real_data =
//         static_cast<double*>(real_buffer.ptr);

//     auto* imag_data =
//         static_cast<double*>(imag_buffer.ptr);

//     std::size_t n = input_buffer.size;

//     fft(
//         input_data,
//         real_data,
//         imag_data,
//         n
//     );

//     std::vector<double> frequencies =
//         find_frequencies(
//             real_data,
//             imag_data,
//             n,
//             sample_rate
//         );

//     py::array_t<double> result(frequencies.size());

//     auto result_buffer = result.request();

//     auto* result_data =
//         static_cast<double*>(result_buffer.ptr);

//     std::copy(
//         frequencies.begin(),
//         frequencies.end(),
//         result_data
//     );

//     return result;
// }