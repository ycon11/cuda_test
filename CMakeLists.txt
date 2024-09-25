cmake_minimum_required(VERSION 3.10)
project(cuda_test LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB_RECURSE src_file 
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/src/*.cu")
set(CUDA_INC_DIR /usr/local/cuda-11.4/include)
set(CUDA_LIB_DIR /usr/local/cuda-11.4/lib64)

message(STATUS ">>>>>>>src_file: ${src_file}")
find_package(CUDA REQUIRED)

add_executable(${PROJECT_NAME} ${src_file})
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INC_DIR})
target_link_directories(${PROJECT_NAME} PUBLIC  ${CUDA_LIB_DIR})

target_link_libraries(${PROJECT_NAME} cuda)