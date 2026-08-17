import numpy as np
import matplotlib.pyplot as plt
import fft_cpp


def round_higher_power_2(v):
    # Source - https://stackoverflow.com/a/4398845
    # Posted by Jason S, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-08-12, License - CC BY-SA 4.0
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    v += 1
    return v


def generate_sequence(
    frequency_range = (1.0, 32.0), 
    amplitude_range = (0.5, 1.0), 
    phase_range = (0.0, 2.0 * np.pi), 
    n_frequencies=8, 
    n_samples=1024):

    n_samples = round_higher_power_2(n_samples)

    f   = np.random.uniform(frequency_range[0], frequency_range[1], n_frequencies)
    A   = np.random.uniform(amplitude_range[0], amplitude_range[1], n_frequencies)
    phi = np.random.uniform(phase_range[0],     phase_range[1],     n_frequencies)

    t = np.arange(n_samples) / n_samples
    seq = np.sum(
        A[:, None] * np.cos(2 * np.pi * f[:, None] * t + phi[:, None]),
        axis=0,
        dtype=np.float64,
    )
    
    return seq, f, A, phi


def generate_test_sequence():
    f   = np.array([3.0, 7.0, 13.0])
    A   = np.array([0.2, 0.5, 0.1])
    phi = np.array([0.3, 1.0, 2.0])
    n_samples = 64

    t = np.arange(n_samples) / n_samples
    seq = np.sum(
        A[:, None] * np.cos(2 * np.pi * f[:, None] * t + phi[:, None]),
        axis=0,
        dtype=np.float64,
    )
    return seq, f, A, phi


def plot_seq(seq, x_range = (0, 1)):
    x = np.linspace(x_range[0], x_range[1], len(seq))
    plt.plot(x, seq)
    plt.xlim(x_range[0], x_range[1])
    plt.yscale("log")
    plt.xlabel("x (linear scale)")
    plt.ylabel("y (log scale)")
    plt.show()


def fft(seq_real, seq_imag, size):
    real = np.empty(size, dtype=np.float64)
    imag = np.empty(size, dtype=np.float64)

    fft_cpp.fft(seq_real, seq_imag, real, imag)

    return real, imag


def ifft(real, imag, size):
    seq_real, seq_imag = fft(real, -imag, size)
    seq_imag *= -1

    return seq_real, seq_imag


if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)

    seq_real, f, A, phi = generate_sequence()
    seq_imag = np.zeros(len(seq_real))

    real, imag = fft(seq_real=seq_real, seq_imag=seq_imag, size=len(seq_real))

    result = fft_cpp.fft_analysis(real, imag, 1.0 / len(seq_real), 0.05)

    print(result)
