#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv)
{
    int N = argc > 1 ? atoi(argv[1]) : 256;
    if (N < 3 || N > 8192) {
        fprintf(stderr, "Expected 3 <= N <= 8192\n");
        return 2;
    }
    double *A = calloc((size_t)N * N, sizeof(double));
    if (A == NULL)
        return 1;
    clock_t started = clock();
    for (int i = 1; i < N - 1; i++)
        for (int j = 1; j < N - 1; j++)
            A[i * N + j] = 2.0 + i + j;
    for (int iteration = 0; iteration < 10; iteration++) {
        double eps = 0.0;
        for (int i = 1; i < N - 1; i++)
            for (int j = 1; j < N - 1; j++) {
                int idx = i * N + j;
                double old = A[idx];
                A[idx] = (2.0 * A[idx - N] + A[idx + N]
                          + 2.0 * A[idx - 1] + A[idx + 1]) / 6.0;
                double difference = fabs(old - A[idx]);
                if (difference > eps)
                    eps = difference;
            }
        if (eps < 1e-8)
            break;
    }
    double sum = 0.0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            sum += A[i * N + j] * (i + 1) * (j + 1) / ((double)N * N);
    double elapsed = (double)(clock() - started) / CLOCKS_PER_SEC;
    printf("1,%.6f,%d,%.6f,sequential\n", elapsed, N, sum);
    free(A);
    return 0;
}
