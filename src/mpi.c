#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

double maxeps = 1e-8;
int itmax = 10;
int rank, num_procs;
int i_start, local_rows;

double *allocate_matrix(int N)
{
    double *base = calloc((size_t)(local_rows + 2) * N, sizeof(double));
    if (base == NULL) {
        fprintf(stderr, "Process %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return base + N;
}

void free_matrix(double *A, int N)
{
    free(A - N);
}

void init_mpi(double *A, int N)
{
    for (int i = 0; i < local_rows; i++)
        for (int j = 1; j < N - 1; j++)
            A[i * N + j] = 2.0 + i_start + i + j;
}

void exchange_boundaries(double *A, int N)
{
    int up = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    int down = rank + 1 < num_procs ? rank + 1 : MPI_PROC_NULL;
    MPI_Sendrecv(&A[(local_rows - 1) * N], N, MPI_DOUBLE, down, 10,
                 &A[-N], N, MPI_DOUBLE, up, 10,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(A, N, MPI_DOUBLE, up, 20,
                 &A[local_rows * N], N, MPI_DOUBLE, down, 20,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void relax_mpi(double *A, int N, double *eps)
{
    int up = rank > 0 ? rank - 1 : MPI_PROC_NULL;
    int down = rank + 1 < num_procs ? rank + 1 : MPI_PROC_NULL;
    double local_eps = 0.0;
    exchange_boundaries(A, N);
    for (int j = 1; j < N - 1; j++) {
        MPI_Recv(&A[-N + j], 1, MPI_DOUBLE, up, 30,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < local_rows; i++) {
            int idx = i * N + j;
            double old = A[idx];
            A[idx] = (2.0 * A[idx - N] + A[idx + N]
                      + 2.0 * A[idx - 1] + A[idx + 1]) / 6.0;
            double difference = fabs(old - A[idx]);
            if (difference > local_eps)
                local_eps = difference;
        }
        MPI_Send(&A[(local_rows - 1) * N + j], 1, MPI_DOUBLE, down, 30,
                 MPI_COMM_WORLD);
    }
    MPI_Allreduce(&local_eps, eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

double verify_mpi(double *A, int N)
{
    double sum = 0.0;
    for (int i = 0; i < local_rows; i++)
        for (int j = 0; j < N; j++)
            sum += A[i * N + j] * (i_start + i + 1) * (j + 1) / ((double)N * N);
    return sum;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    int N = argc > 1 ? atoi(argv[1]) : 256;
    if (N < 3 || N > 8192 || num_procs > N - 2) {
        if (rank == 0)
            fprintf(stderr, "Expected 3 <= N <= 8192 and processes <= N - 2\n");
        MPI_Finalize();
        return 2;
    }
    int rows = N - 2;
    local_rows = rows / num_procs + (rank < rows % num_procs);
    i_start = 1 + rank * (rows / num_procs)
              + (rank < rows % num_procs ? rank : rows % num_procs);
    double *A = allocate_matrix(N);
    MPI_Barrier(MPI_COMM_WORLD);
    double started = MPI_Wtime();
    init_mpi(A, N);
    for (int iteration = 0; iteration < itmax; iteration++) {
        double eps;
        relax_mpi(A, N, &eps);
        if (eps < maxeps)
            break;
    }
    double elapsed = MPI_Wtime() - started;
    double maximum_elapsed, local_sum = verify_mpi(A, N), sum;
    MPI_Reduce(&elapsed, &maximum_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("%d,%.6f,%d,%.6f,mpi\n", num_procs, maximum_elapsed, N, sum);
    free_matrix(A, N);
    MPI_Finalize();
    return 0;
}
