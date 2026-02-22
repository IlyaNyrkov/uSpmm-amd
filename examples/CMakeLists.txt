# MTX file reading example
add_executable(mtx_file_reading mtx_file_reading.cpp)
target_link_libraries(mtx_file_reading PRIVATE uspmm_lib)

# Checkers clustering example
add_executable(checkers_reordering checkers_reordering.cpp)
target_link_libraries(checkers_reordering PRIVATE uspmm_lib)

# USPMM hip kernel example
add_executable(integration_test uspmm_checkers.hip)
target_link_libraries(integration_test PRIVATE uspmm_lib)