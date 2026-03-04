cmake_minimum_required(VERSION 3.16)
project(llama_inference_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LLAMA_INFERENCE_ENABLE_OPENMP "Enable OpenMP for CPU parallel loops" ON)
option(LLAMA_INFERENCE_BUILD_CUDA_KERNELS "Build modular CUDA kernel library" ON)

add_library(llama_engine src/llama_engine.cpp)
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

if(LLAMA_INFERENCE_BUILD_CUDA_KERNELS)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)

        add_library(llama_cuda_kernels
            src/cuda_kernels/rmsnorm_kernel.cu
            src/cuda_kernels/matvec_kernel.cu
            src/cuda_kernels/rope_kernel.cu
            src/cuda_kernels/attention_scores_kernel.cu
            src/cuda_kernels/softmax_kernel.cu
            src/cuda_kernels/attention_weighted_sum_kernel.cu
            src/cuda_kernels/swiglu_kernel.cu
        )
        target_include_directories(llama_cuda_kernels PUBLIC include)
        set_target_properties(llama_cuda_kernels PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            POSITION_INDEPENDENT_CODE ON
        )
    else()
        message(WARNING "CUDA compiler not found. Skipping llama_cuda_kernels target.")
    endif()
endif()
