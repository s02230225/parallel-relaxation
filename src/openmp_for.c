#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define Max(a,b) ((a)>(b)?(a):(b))

double maxeps = 0.1e-7;
int itmax = 10;

double* allocate_matrix(int N) {
    return (double*)malloc((size_t)N * N * sizeof(double));
}

void free_matrix(double *A) { free(A); }

void init(double *A, int N) {
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            if(i == 0 || i == N-1 || j == 0 || j == N-1)
                A[i*N + j] = 0.0;
            else
                A[i*N + j] = (2.0 + i + j);
        }
    }
}

void relax(double *A, int N, double *eps, int num_threads) {
    double eps_local = 0.0;
    #pragma omp parallel num_threads(num_threads) reduction(max:eps_local)
    {
        for (int diagonal = 2; diagonal <= 2 * (N - 2); diagonal++) {
            #pragma omp for schedule(static)
            for (int i = 1; i <= N - 2; i++) {
                int j = diagonal - i;
                if (j < 1 || j > N - 2) continue;
                int idx = i * N + j;
                double old = A[idx];
                A[idx] = (2.0 * A[idx-N] + A[idx+N]
                          + 2.0 * A[idx-1] + A[idx+1]) / 6.0;
                double diff = fabs(old - A[idx]);
                if (diff > eps_local) eps_local = diff;
            }
        }
    }
    *eps = eps_local;
}

void verify(double *A, int N, double *s) {
    const double inv_N_sq = 1.0 / (N * N);
    double local_s = 0.0;

    #pragma omp parallel for collapse(2) reduction(+:local_s)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            local_s += A[i*N + j] * (i+1) * (j+1) * inv_N_sq;
        }
    }
    *s = local_s;
}

int main(int argc, char **argv) {
    int num_threads = (argc > 1) ? atoi(argv[1]) : 1;
    int N = (argc > 2) ? atoi(argv[2]) : 256;
    int runs = (argc > 3) ? atoi(argv[3]) : 1;

    if (num_threads < 1 || num_threads > 256 || N < 3 || N > 8192 || runs < 1) {
        fprintf(stderr, "Invalid threads, matrix size or run count\n");
        return 2;
    }
    omp_set_num_threads(num_threads);

    double total_time = 0.0;
    double final_sum = 0.0;

    for(int run = 0; run < runs; run++) {
        double *A = allocate_matrix(N);
        if (A == NULL) return 1;
        double start = omp_get_wtime();

        init(A, N);

        for(int it = 1; it <= itmax; it++) {
            double eps = 0.0;
            relax(A, N, &eps, num_threads);
            if (eps < maxeps) break;
        }

        double s;
        verify(A, N, &s);
        double elapsed = omp_get_wtime() - start;

        total_time += elapsed;
        final_sum = s;
        free_matrix(A);
    }

    double avg_time = total_time / runs;
    printf("%d,%6f,%d,%6f,omp_for\n", num_threads, avg_time, N, final_sum);
    return 0;
}