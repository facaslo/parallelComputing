#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <stdbool.h>
#include <math.h>

#define MATRIX_SIZE 1024
#define BLOCK_SIZE 8
#define THREADS 8
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

__global__ void matrixMul(double *a, double *b, double *c, int size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        double sum = 0.0;
        for (int k = 0; k < size; k++) {
            sum += *(a+ row*size + k) * *(b + size*k + col);            
        }
        *(c + size*row + col) = sum;
    }
    //printf("Block id x : %d , Block id y : %d",blockIdx.x, blockIdx.y);    
}

bool compare_matrices(double *matrix1, double *matrix2, int n){
  bool isTheSame = true;
  for (int i=0; i< n; i++){
    int row = i*n;
    for(int j=0; j<n; j++){
      double difference = *(matrix1 + row + j) - *(matrix2 + row + j);
      if(abs(difference) > 1E-6){
        isTheSame = false;
        break;
      }
    }
  }
  return isTheSame;
}


int main()
{
    double *a, *b, *c, *d;
    double *dev_a, *dev_b, *dev_c;
    int matrix_bytes = MATRIX_SIZE * MATRIX_SIZE * sizeof(double);

    // Allocate host memory
    a = (double*)malloc(matrix_bytes);
    b = (double*)malloc(matrix_bytes);
    c = (double*)malloc(matrix_bytes);
    d = (double*)malloc(matrix_bytes);
    // Initialize matrices with random doubles
    srand(time(NULL));
    fill_matrix(a,MATRIX_SIZE);
    fill_matrix(b,MATRIX_SIZE);

    // Allocate device memory
    cudaMalloc((void**)&dev_a, matrix_bytes);
    cudaMalloc((void**)&dev_b, matrix_bytes);
    cudaMalloc((void**)&dev_c, matrix_bytes);

    // Copy matrices to device
    cudaMemcpy(dev_a, a, matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, matrix_bytes, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 gridDim(ceil((float)MATRIX_SIZE / BLOCK_SIZE), ceil((float)MATRIX_SIZE / BLOCK_SIZE), 1);
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);

    // Launch kernel
    matrixMul<<<gridDim, blockDim>>>(dev_a, dev_b, dev_c, MATRIX_SIZE);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
      printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    // Copy result back to host
    cudaMemcpy(c, dev_c, matrix_bytes, cudaMemcpyDeviceToHost);

    // Sequential result
    
    multiply_matrices(a,b,d,MATRIX_SIZE);
    // print_matrix(a,MATRIX_SIZE);
    // print_matrix(b,MATRIX_SIZE);
    // printf("-------------------------------------------------------------------------------\n");
    // print_matrix(c,MATRIX_SIZE);
    // printf("-------------------------------------------------------------------------------\n");
    // print_matrix(d,MATRIX_SIZE);

    bool comparison_result = compare_matrices(c,d,MATRIX_SIZE);
    
    // Free memory
    free(a);
    free(b);
    free(c);
    free(d);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    printf("Matrices iguales: %s \n", comparison_result ? "true" : "false");
    return 0;
}
