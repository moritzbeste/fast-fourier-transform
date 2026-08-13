build using:

cmake -S . -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DPython_EXECUTABLE="$(which python)" -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" -DOpenMP_ROOT="$(brew --prefix libomp)"
cmake --build build

run using:

PYTHONPATH=build python fourier.py