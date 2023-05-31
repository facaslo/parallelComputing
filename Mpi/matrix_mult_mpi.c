#include <string.h>
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>

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

void multiply_matrices(double *matrix1, double *matrix2, double *result, int n) {
    for (int i = 0; i < n; i++) {
        int row = i * n;
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += *(matrix1 + row + k) * *(matrix2 + k * n + j);
            }
            *(result + row + j) = sum;
        }
    }
}

bool compare_matrices(double *matrix1, double *matrix2, int n){
  bool isTheSame = true;
  for (int i=0; i< n; i++){
    int row = i*n;
    for(int j=0; j<n; j++){
      double difference = *(matrix1 + row + j) - *(matrix2 + row + j);
      if(abs(difference) > 1E-9){
        isTheSame = false;
        break;
      }
    }
  }
  return isTheSame;
}

void multiply_matrices_mpi(double *matrix1, double *matrix2, double *result, int n){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_process = n / size;
    int remaining_rows = n % size;

    int start_row = rank * rows_per_process;
    int end_row = start_row + rows_per_process;
    if (rank == size - 1) {
        end_row += remaining_rows;
    }

    for (int i = start_row; i < end_row; i++) {
        int row = i * n;
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += *(matrix1 + row + k) * *(matrix2 + k * n + j);
            }
            *(result + row + j) = sum;
        }
    }

    if (rank != 0) {
        MPI_Send(result + start_row * n, (end_row - start_row) * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        for (int i = 1; i < size; i++) {
            start_row = i * rows_per_process;
            end_row = start_row + rows_per_process;
            if (i == size - 1) {
                end_row += remaining_rows;
            }
            MPI_Recv(result + start_row * n, (end_row - start_row) * n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: %s <matrix_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);

    double *matrix1 = (double *) malloc(n * n * sizeof(double));
    double *matrix2 = (double *) malloc(n * n * sizeof(double));
    double *resultSequential = (double *) malloc(n * n * sizeof(double));
    double *resultMPI = (double *) malloc(n * n * sizeof(double));

    if (rank == 0) {
        srand(time(NULL));
        fill_matrix(matrix1, n);
        fill_matrix(matrix2, n);

        printf("Matrix 1:\n");
        print_matrix(matrix1, n);

        printf("Matrix 2:\n");
        print_matrix(matrix2, n);
    }

    // Broadcast the matrices to all processes
    MPI_Bcast(matrix1, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrix2, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Call the mpi matrix multiplication method here
    multiply_matrices_mpi(matrix1, matrix2, resultMPI, n);

    if (rank == 0) {
        multiply_matrices(matrix1, matrix2, resultSequential, n);
        bool comparisonResult = compare_matrices(resultMPI,resultSequential,n);
        printf("Matrix result Sequential:\n");
        print_matrix(resultSequential, n);
        printf("Matrix result MPI:\n");
        print_matrix(resultMPI, n);
        printf("Matrices iguales: %s \n", comparisonResult ? "true" : "false");
    }

    free(matrix1);
    free(matrix2);
    free(resultMPI);
    free(resultSequential);

    MPI_Finalize();
    return 0;
}