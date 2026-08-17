#include "fft.hpp"
#include <cmath>
#include <cstddef>
#include <omp.h>

void fft(const double* real_input, const double* imag_input, double* real_output, double* imag_output, std::size_t n) {
    ComplexBuffer ping(n);
    ComplexBuffer pong(n);

    #pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        ping.re[i] = real_input[i];
        ping.im[i] = imag_input[i];
    }

    std::size_t j = 0;
    for (std::size_t i = 1; i < n; ++i) {
        std::size_t bit = n >> 1;

        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }

        j ^= bit;

        if (i < j) {
            std::swap(ping.re[i], ping.re[j]);
            std::swap(ping.im[i], ping.im[j]);
        }
    }

    std::size_t p = omp_get_max_threads();

    std::size_t k = 2;

    ComplexBuffer* in  = &ping;
    ComplexBuffer* out = &pong;

    #pragma omp parallel
    while (k <= n) {
        if (k < p) { omp_set_schedule(omp_sched_static, n / p); }
        else       { omp_set_schedule(omp_sched_static, 1); }

        #pragma omp for schedule(runtime)
        for (std::size_t i = 0; i < n / 2; ++i) {
            std::size_t j = i % (k / 2);
            std::size_t group = (i / (k / 2)) * k;

            std::size_t a = group + j;
            std::size_t b = a + k / 2;

            double angle = -2.0 * M_PI * j / k;
            double wr = std::cos(angle);
            double wi = std::sin(angle);

            double br = in->re[b];
            double bi = in->im[b];

            double tr = wr * br - wi * bi;
            double ti = wr * bi + wi * br;

            double ar = in->re[a];
            double ai = in->im[a];

            out->re[a] = ar + tr;
            out->im[a] = ai + ti;

            out->re[b] = ar - tr;
            out->im[b] = ai - ti;
        }

        #pragma omp single
        {
            std::swap(in, out);
            k *= 2;
        }
    }

    #pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        real_output[i] = in->re[i];
        imag_output[i] = in->im[i];
    }
}

AnalysisResult fftAnalysis(const double* real_input, const double* imag_input, std::size_t n, double dt, double threshold) {
    AnalysisResult result;

    #pragma omp parallel for
    for (std::size_t i = 0; i <= n / 2; ++i) {
        const double re = real_input[i];
        const double im = imag_input[i];

        const double magnitude = std::sqrt(re * re + im * im);

        double amplitude = magnitude / static_cast<double>(n);

        // Convert the two-sided FFT into a one sided
        // amplitude spectrum.
        if (i != 0 && i != n / 2) {
            amplitude *= 2.0;
        }

        const double frequency = static_cast<double>(i) / (static_cast<double>(n) * dt);

        const double phase = std::atan2(im, re);

        if (amplitude > threshold) {
            #pragma omp critical
            {
                result.frequencies.push_back(frequency);
                result.amplitudes.push_back(amplitude);
                result.phases.push_back(phase);
            }
        }
    }

    return result;
}