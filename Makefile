CC = cc
MPICC = mpicc
CFLAGS = -O2 -std=c11 -Wall -Wextra -Wpedantic

all: sequential openmp_for openmp_pipeline

sequential: src/sequential.c
	$(CC) $(CFLAGS) $< -lm -o $@

openmp_for: src/openmp_for.c
	$(CC) $(CFLAGS) -fopenmp $< -lm -o $@

openmp_pipeline: src/openmp_pipeline.c
	$(CC) $(CFLAGS) -fopenmp $< -lm -o $@

mpi: src/mpi.c
	$(MPICC) $(CFLAGS) $< -lm -o $@
