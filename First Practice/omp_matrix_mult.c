#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define MAX_DOUBLE 1.7976931348623158E+3

double RandomReal(double low, double high)
{
  double d;
  d = (double) rand() / ((double) RAND_MAX + 1);
  return (low + d * (high - low));
}

void fill_matrix(double *matrix, int n){
    for (int i = 0; i < n * n; i++) {
        *(matrix + i) = RandomReal(-MAX_DOUBLE, MAX_DOUBLE) ;
    }
}

void print_matrix(double *matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.3f ", *(matrix + i * n + j));
        }
        printf("\n");
    }
}

void multiply_matrices(double *matrix1, double *matrix2, double *result, int n, int num_threads) {
    omp_set_num_threads(num_threads);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += *(matrix1 + i * n + k) * *(matrix2 + k * n + j);
            }
            *(result + i * n + j) = sum;
        }
    }
}

int main(int argc, char *argv[]) {   
    struct timespec start, end;
    double elapsed_time;
    clock_gettime(CLOCK_MONOTONIC, &start);
    int num_threads = atoi(argv[1]);
    int n = atoi(argv[2]);

    double *matrix1 = (double *) malloc(n * n * sizeof(double));
    double *matrix2 = (double *) malloc(n * n * sizeof(double));
    double *result = (double *) malloc(n * n * sizeof(double));

    srand(time(NULL));
    fill_matrix(matrix1, n);
    fill_matrix(matrix2, n);

    // printf("Matrix 1:\n");
    // print_matrix(matrix1, n);

    // printf("Matrix 2:\n");
    // print_matrix(matrix2, n);

    
    multiply_matrices(matrix1, matrix2, result, n, num_threads);    

    // printf("Result:\n");
    // print_matrix(result, n);    

    free(matrix1);
    free(matrix2);
    free(result);

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_time =  (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("%d %d %.5f \n", num_threads ,  n , elapsed_time);

    return 0;
}
