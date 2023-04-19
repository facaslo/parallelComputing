#include <omp.h>
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"

void main(int argc, char **argv) {
  struct timespec start, end;
  double elapsed_time;
  long iterations;
  int threads_number;
  double pi = 0.0;

  threads_number = atoi(argv[1]);
  iterations = atol(argv[2]);

  clock_gettime(CLOCK_MONOTONIC, &start);

  omp_set_num_threads(threads_number);

  #pragma omp parallel for reduction(+:pi)
  for (long i = 0; i < iterations; i++) {
    pi += (i%2==0? 1 : -1) / (2.0 * i + 1.0);
  }

  pi *= 4.0;

  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed_time = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("Iteraciones:%d\nHilos:%d\nPi:%.10f\nTiempo:%.10f segundos\n", (int)iterations, threads_number, pi, elapsed_time);
}
