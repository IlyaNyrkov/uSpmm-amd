cmake_minimum_required(VERSION 3.16)
project(uSpmmProject LANGUAGES CXX HIP)

# 1. C++ Standard: Upgraded to C++20 for <bit> and std::popcount
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Hardware Target (MI210)
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx90a")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Aggressive CPU optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# Aggressive GPU optimization
set(CMAKE_HIP_FLAGS_RELEASE "-O3 -ffast-math -munsafe-fp-atomics")

# 3. Find ROCm dependencies
find_package(ROCM REQUIRED)
# rocWMMA is header-only, so we find its include path
find_path(ROCWMMA_INCLUDE_DIR NAMES rocwmma/rocwmma.hpp)

find_package(hipsparse REQUIRED)

# 4. Create the main library
add_library(uspmm_lib STATIC
        src/uspmm/spmm_rocwmma.hip
)

# 5. Configure Includes
target_include_directories(uspmm_lib PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${ROCWMMA_INCLUDE_DIR}
)
target_compile_features(uspmm_lib PUBLIC cxx_std_20)

target_link_libraries(uspmm_lib PUBLIC roc::hipsparse)

# 6. Testing Module
enable_testing()
add_subdirectory(tests)

add_subdirectory(examples)
add_subdirectory(benchmarks)