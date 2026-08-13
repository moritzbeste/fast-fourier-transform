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
        #pragma omp for
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

// std::vector<double> find_frequencies(...);