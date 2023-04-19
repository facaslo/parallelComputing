#include <omp.h>
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
// #define THREADS_NUMBER 2

static int  threads_number;
static long iterations;

typedef struct{
  double sum;
  char padding[64];
} padded_double ;

void leibniz_series(long begin, long end, long max , double *sum){  
  
  for(long i=begin; i < end && i<=max; i++){ 
    (*sum) += 1.0/(2*(float)i+1);
    i++;    
    (*sum) += -1.0/(2*(float)i+1);      
  }    
}

void main (int argc, char **argv) 
{
  struct timespec start, end;
  double elapsed_time;
  long blockSize;
  threads_number = atoi(argv[1]);
  iterations = atol(argv[2]);

  clock_gettime(CLOCK_MONOTONIC, &start);
  int i ; double pi;
  padded_double sum[threads_number];
  blockSize = (long)ceil(iterations /(double) threads_number);
  omp_set_num_threads(threads_number);
  #pragma omp parallel
  {
    int id;    
    id = omp_get_thread_num();    
    sum[id].sum = 0.0;
    long begin = id * blockSize;    
    long end = (id+1) * blockSize;    
    leibniz_series(begin,end,iterations , &sum[id].sum);
    
  }  
  for(i=0, pi=0.0;i<threads_number;i++) {    
    pi += sum[i].sum ;
  }
    
  pi *= 4;
  
  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed_time =  (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("Iteraciones:%d\nHilos:%d\nPi:%.10f\nTiempo:%.10f segundos\n", (int)iterations, threads_number, pi, elapsed_time);
}