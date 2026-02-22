build_all:
	echo;

run_tests:
	echo;

build_rwma:
	hipcc -std=c++17 -O3 wmma_example.hip.cpp -o wmma_example --offload-arch=gfx90a