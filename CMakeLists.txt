cmake_minimum_required(VERSION 3.18)

project(fft_cpp LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP REQUIRED COMPONENTS CXX)

pybind11_add_module(fft_cpp
    bindings.cpp
    fft.cpp
)

target_link_libraries(fft_cpp PRIVATE OpenMP::OpenMP_CXX)

target_compile_features(fft_cpp PRIVATE cxx_std_17)