/*
  Este programa calcula la multiplicacion de dos matrices usando hilos posix
*/

#include "stdlib.h"
#include "pthread.h"
#include "stdio.h"
#include "stddef.h"
#include "math.h"
#include "time.h"

#define ROW_SIZE 1024
#define COL_SIZE 1024
#define MATRIX_SIZE ROW_SIZE*COL_SIZE

int main(int argc, char* argv[]){
  // Initialize a random seed for the rand() generator
  srand(time(NULL));
  int *A, *B, *C;
  A = (int*)calloc(MATRIX_SIZE , sizeof(int));
  B = (int*)calloc(MATRIX_SIZE , sizeof(int));
  C = (int*)calloc(MATRIX_SIZE , sizeof(int)); 

  for(int i=0; i<ROW_SIZE; i++){
    for(int j=0; i<COL_SIZE; j++){
      
    }

  }

  
  return 0;
}