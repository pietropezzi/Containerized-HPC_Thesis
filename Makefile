CFLAGS+=-std=c99 -Wall -Wpedantic -O2
LDLIBS=-lm

.PHONY: clean

run: 
	gcc $(CFLAGS) -o bilateral-filter bilateral-filter.c -lm

mpi:
	mpicc $(CFLAGS) bilateral-filter-mpi.c -o mpi-bilateral $(LDLIBS)

cuda:
	nvcc bilateral-filter-cuda.cu -o cuda-bilateral

noise:
	gcc $(CFLAGS) -o noise noise.c -lm

clean:
	rm -f noise bilateral-filter mpi-bilateral cuda-bilateral *.o
