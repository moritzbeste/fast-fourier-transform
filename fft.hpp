#pragma once

#include <cstddef>
#include <vector>

struct ComplexBuffer {
    std::vector<double> re;
    std::vector<double> im;

    explicit ComplexBuffer(std::size_t n): re(n), im(n) {}
};

struct AnalysisResult {
    std::vector<double> frequencies;
    std::vector<double> amplitudes;
    std::vector<double> phases;
};

void fft(const double* real_input, const double* imag_input, double* real_output, double* imag_output, std::size_t n);

AnalysisResult fftAnalysis(const double* real_input, const double* imag_input, std::size_t n, double dt, double threshold);