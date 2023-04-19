/*
  Este programa calcula el valor de pi, usando la serie de Leibniz por medio de hilos posix 
*/

#include "stdlib.h"
#include "pthread.h"
#include "stdio.h"
#include "stddef.h"
#include "math.h"
#include "time.h"

// Defiinición de numero de hilos y el valor de N
// #define THREADS_NUMBER 4
// #define ITERATIONS 1e9
static int threads_number;
static long iterations;

// Se tiene que definir una estructura para que la funcion del hilo reciba un único argumento de este tipo
typedef struct {
  double* sum;
  int thread_number;
  long begin;
  long end;
  long iterations;
} thread_data;

// La función que va a calcular una parte de la serie de leibniz y que va a recibir cada hilo
void* leibniz_series(void* arg){
  thread_data* data = (thread_data*) arg;
  for(long i=data->begin; i< data->end && i<=data->iterations; i++){ 
    //printf("Iteración:%i - begin:%i - end:%i\n", i, data->begin, data->end);
    // *(data->sum) += ((i%2==0)?1.0/(2*(float)i+1):-1.0/(2*(float)i+1));    Más ineficiente
    *(data->sum) += 1.0/(2*(float)i+1);
    i++;    
    *(data->sum) += -1.0/(2*(float)i+1);    
  }
}

int main(int argc, char **argvc){
  // Inicializaciones para el reloj  
  struct timespec start, end;
  double elapsed_time;
  threads_number = atoi(argvc[1]);
  iterations = atol(argvc[2]);
  clock_gettime(CLOCK_MONOTONIC, &start);
  // Inicialización requerida para los hilos
  pthread_t threads[threads_number]; 
  thread_data thread_args[threads_number];
  // Para crear un arreglo inicializado con ceros con el tamaño dado por la directiva Thread number
  double* sum_arr = (double*)calloc(threads_number, sizeof(double));  
  int thread_size = (int)ceil((float)iterations/(float)threads_number);    

  // Creación de los hilos y asignación de los argumentos  
  for(int i = 0; i<threads_number; i++){
    thread_args[i].sum = &sum_arr[i];    
    thread_args[i].thread_number = i;
    thread_args[i].begin = i*thread_size;
    thread_args[i].end = (i+1)*thread_size;    
    thread_args[i].iterations = iterations; 
    int r = pthread_create(&threads[i], NULL, leibniz_series, &thread_args[i]);
    if (r != 0)
      perror("Error al crear el hilo");
  }

  // Espera que cada hilo se termine de ejecutar
  for(int i = 0; i<threads_number; i++){
    if (pthread_join(threads[i], NULL) != 0) {
        fprintf(stderr, "Error en el joining de los hilos %d\n", i);
        exit(1);
    }
  } 

  // Obtención de Pi y muestra de resultados
  double pi = 0;
  for(int i = 0; i<threads_number; i++){
    // printf("sum_arr[%i]: %.10f \n", i, sum_arr[i]);
    pi += sum_arr[i];
  }
  pi = 4*pi;

  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed_time =  (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("Iteraciones:%d\nHilos:%d\nPi:%.10f\nTiempo:%.10f segundos\n", (int)iterations, threads_number, pi, elapsed_time);

  free(sum_arr);
  return 0;
}
