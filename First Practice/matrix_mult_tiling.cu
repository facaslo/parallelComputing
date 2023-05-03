#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
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

__global__ void MatrixMulKernel(double* d_M, double* d_N, double* d_P, int Width, int tileWidth) {
  __shared__ double Mds[tileWidth][tileWidth];
  __shared__ double Nds[tileWidth][tileWidth];
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;
  // Identify the row and column of the d_P element to work on
  int Row = by * tileWidth + ty;
  int Col = bx * tileWidth + tx;
  double Pvalue = 0;
  // Loop over the d_M and d_N tiles required to compute d_P element
  for (int m = 0; m < Width/tileWidth; ++m) {
    // Coolaborative loading of d_M and d_N tiles into shared memory
    Mds[ty][tx] = *(d_M + Row*Width + m*tileWidth + tx);
    Nds[ty][tx] = *(d_N + (m*tileWidth + ty)*Width + Col);
    // Mds[ty][tx] = d_M[Row*Width + m*TILE_WIDTH + tx];
    // Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty)*Width + Col];
    __syncthreads();
    for (int k = 0; k < tileWidth; ++k) {
      Pvalue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }
  //d_P[Row*Width + Col] = Pvalue;
  *(d_P + Row*Width + Col) = Pvalue;
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

int main(int argc, char *argv[])
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int matrix_size = atoi(argv[1]);
    int tile_width = atoi(argv[2]);        
    double *a, *b, *c, *d;
    double *dev_a, *dev_b, *dev_c;
    int matrix_bytes = matrix_size * matrix_size * sizeof(double);

    // Allocate host memory
    a = (double*)malloc(matrix_bytes);
    b = (double*)malloc(matrix_bytes);
    c = (double*)malloc(matrix_bytes);
    d = (double*)malloc(matrix_bytes);
    // Initialize matrices with random doubles
    srand(time(NULL));
    fill_matrix(a,matrix_size);
    fill_matrix(b,matrix_size);

    // Allocate device memory
    cudaMalloc((void**)&dev_a, matrix_bytes);
    cudaMalloc((void**)&dev_b, matrix_bytes);
    cudaMalloc((void**)&dev_c, matrix_bytes);

    // Copy matrices to device
    cudaMemcpy(dev_a, a, matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, matrix_bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 grimDim(ceil((float)matrix_size/tile_width), ceil((float)matrix_size/tile_width) ,1);
    dim3 blockDim(tile_width,tile_width,1);
    cudaEventRecord(start);
    MatrixMulKernel<<<gridDim, blockDim>>>(dev_a, dev_b, dev_c, matrix_size, tile_width);
    cudaEventRecord(stop);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
      printf("Error: %s\n", cudaGetErrorString(err));
    cudaDeviceSynchronize();
    // Copy result back to host
    cudaMemcpy(c, dev_c, matrix_bytes, cudaMemcpyDeviceToHost);

    // Sequential result
    
    // multiply_matrices(a,b,d,matrix_size);
    // print_matrix(a,matrix_size);
    // print_matrix(b,matrix_size);
    // printf("-------------------------------------------------------------------------------\n");
    // print_matrix(c,matrix_size);
    // printf("-------------------------------------------------------------------------------\n");
    // print_matrix(d,matrix_size);

    bool comparison_result = compare_matrices(c,d,matrix_size);
    printf("Are matrices the same: %s \n", comparison_result==true?"verdadero":"falso");
    
    // Free memory
    free(a);
    free(b);
    free(c);
    free(d);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    
    int number_of_blocks = grid_dimensions*grid_dimensions;
    
    cudaEventSynchronize(stop);
    double milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Matrix-size:%d - threads per block :%d - Number of blocks:%d - Time:%.20f mS", matrix_size , threads_per_block , number_of_blocks ,  milliseconds);
    return 0;
}