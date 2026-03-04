cmake_minimum_required(VERSION 3.16)
project(llama_inference_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LLAMA_INFERENCE_ENABLE_OPENMP "Enable OpenMP for CPU parallel loops" ON)

add_library(llama_engine
    src/tokenizer.cpp
    src/engine_lifecycle.cpp
    src/engine_math_kernels.cpp
    src/engine_forward.cpp
    src/engine_sampling.cpp
)
target_include_directories(llama_engine PUBLIC include)

if(LLAMA_INFERENCE_ENABLE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(llama_engine PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(llama_engine PUBLIC LLAMA_OPENMP=1)
    endif()
endif()

add_executable(llama_infer_cpp src/main.cpp)
target_link_libraries(llama_infer_cpp PRIVATE llama_engine)

add_executable(tokenizer_test tests/test_tokenizer.cpp)
target_link_libraries(tokenizer_test PRIVATE llama_engine)

add_executable(generation_test tests/test_generation.cpp)
target_link_libraries(generation_test PRIVATE llama_engine)
