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
#define THREADS_NUMBER 14
#define ITERATIONS 2e9

// Se tiene que definir una estructura para que la funcion del hilo reciba un único argumento de este tipo
typedef struct {
  double* sum;
  int thread_number;
  int begin;
  int end;
} thread_data;

// La función que va a calcular una parte de la serie de leibniz y que va a recibir cada hilo
void* leibniz_series(void* arg){
  thread_data* data = (thread_data*) arg;
  for(int i=data->begin; i< data->end && i<=ITERATIONS; i++){ 
    //printf("Iteración:%i - begin:%i - end:%i\n", i, data->begin, data->end);
    *(data->sum) += ((i%2==0)?1.0/(2*(float)i+1):-1.0/(2*(float)i+1));    
  }
}

int main(){
  // Inicializaciones para el reloj
  struct timespec start, end;
  double elapsed_time;
  // Inicialización requerida para los hilos
  pthread_t threads[THREADS_NUMBER]; 
  thread_data thread_args[THREADS_NUMBER];
  // Para crear un arreglo inicializado con ceros con el tamaño dado por la directiva Thread number
  double* sum_arr = (double*)calloc(THREADS_NUMBER, sizeof(double));  
  int thread_size = (int)ceil((float)ITERATIONS/(float)THREADS_NUMBER);    

  // Creación de los hilos y asignación de los argumentos
  clock_gettime(CLOCK_MONOTONIC, &start);
  for(int i = 0; i<THREADS_NUMBER; i++){
    thread_args[i].sum = &sum_arr[i];    
    thread_args[i].thread_number = i;
    thread_args[i].begin = i*thread_size;
    thread_args[i].end = (i+1)*thread_size;    
    int r = pthread_create(&threads[i], NULL, leibniz_series, &thread_args[i]);
    if (r != 0)
      perror("Error al crear el hilo");
  }

  // Espera que cada hilo se termine de ejecutar
  for(int i = 0; i<THREADS_NUMBER; i++){
    if (pthread_join(threads[i], NULL) != 0) {
        fprintf(stderr, "Error en el joining de los hilos %d\n", i);
        exit(1);
    }
  }

  clock_gettime(CLOCK_MONOTONIC, &end);
  elapsed_time =  (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  // Obtención de Pi y muestra de resultados
  double pi = 0;
  for(int i = 0; i<THREADS_NUMBER; i++){
    // printf("sum_arr[%i]: %.10f \n", i, sum_arr[i]);
    pi += sum_arr[i];
  }
  pi = 4*pi;

  printf("El valor de pi, con %d iteraciones y %d hilos, es: %.10f \nEl tiempo que se tardó en ejecutar el programa fue de %.10f segundos\n" , (int)ITERATIONS, THREADS_NUMBER,  pi , elapsed_time);

  free(sum_arr);
  return 0;
}
