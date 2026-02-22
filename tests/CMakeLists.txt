include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

# 1. Clustering Test (CPU only)
add_executable(test_clustering test_clustering.cpp)
# Link against GTest and our main project's include directories
target_link_libraries(test_clustering PRIVATE gtest_main uspmm_lib)
add_test(NAME ClusteringTest COMMAND test_clustering)

# 2. Conversion to BCSR (CPU only)
add_executable(test_conversion test_conversion.cpp)
target_link_libraries(test_conversion PRIVATE gtest_main uspmm_lib)
add_test(NAME FormatConversionTest COMMAND test_conversion)

# 2. Future Hardware Test (HIP)
# add_executable(test_spmm_rocwmma test_spmm_rocwmma.hip)
# target_link_libraries(test_spmm_rocwmma PRIVATE gtest_main uspmm_lib)
# add_test(NAME SpmmHardwareTest COMMAND test_spmm_rocwmma)