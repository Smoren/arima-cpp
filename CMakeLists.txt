cmake_minimum_required(VERSION 4.0)
project(arima)

set(CMAKE_CXX_STANDARD 20)

set(EIGEN3_INCLUDE_DIR "libs/eigen-3.4.0")
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(arima main.cpp
        src/arima.h
        src/arima.cpp
        src/utils.h
        src/utils.cpp
)
