cmake_minimum_required(VERSION 3.16)
project(uSpmmProject LANGUAGES CXX HIP)

# 1. C++ Standard: Upgraded to C++20 for <bit> and std::popcount
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. Hardware Target (MI210)
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(CMAKE_HIP_ARCHITECTURES "gfx90a")
endif()

# 3. Find ROCm dependencies
find_package(ROCM REQUIRED)
# rocWMMA is header-only, so we find its include path
find_path(ROCWMMA_INCLUDE_DIR NAMES rocwmma/rocwmma.hpp)

# 4. Create the main library
# (Ensure you have a dummy or actual src/uspmm/spmm_rocwmma.hip.cpp file,
# otherwise CMake will complain about missing source files)
add_library(uspmm_lib STATIC
        src/uspmm/spmm_rocwmma.hip.cpp
)

# 5. Configure Includes
target_include_directories(uspmm_lib PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        ${ROCWMMA_INCLUDE_DIR}
)
target_compile_features(uspmm_lib PUBLIC cxx_std_20)

# 6. Testing Module
enable_testing()
add_subdirectory(tests)

# add_executable(example_uspmm examples/example_uspmm.cpp)
# target_link_libraries(example_uspmm PRIVATE uspmm_lib)

add_subdirectory(examples)