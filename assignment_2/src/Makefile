CC=gcc
CXX=g++
NVCC=nvcc
CFLAGS=-O2 -std=c17 -D_POSIX_C_SOURCE=199309L
CXXFLAGS=-O2 -std=c++17
CPPFLAGS=
CUDAFLAGS=-O2 -std=c++17 -fmad=false

# First target is the default target
all: problem1 problem2-v0 problem2-v1 problem3

# $@ is a variable that contains the target name.
problem1: problem1.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

problem2-v0: problem2-v0.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $(patsubst %.c,%.out,$<)

# -fmad=false: no FMA contraction, so rounding matches the C version
problem2-v1: problem2-v1.cu
	$(NVCC) $(CUDAFLAGS) $< -o $(patsubst %.cu,%.out,$<)

# The target is not a filename
clean:
	rm -f *.out compile_commands.json
