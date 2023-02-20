/*
  Este programa calcula la suma de los primeros N numeros por medio de hilos posix. 
*/

#include "stdlib.h"
#include "pthread.h"
#include "stdio.h"
#include "stddef.h"
#include "math.h"

// Defiinición de numero de hilos y el valor de N
#define THREADS_NUMBER 5
#define N 99

// Se tiene que definir una estructura para que la funcion del hilo reciba un único argumento de este tipo
typedef struct {
  int* sum;
  int thread_number;
  int begin;
  int end;
} thread_data;

// La función de suma que va a recibir cada hilo, vamos a partir el problema en sumas particionadas para cada hilo
void* sum(void* arg){
  thread_data* data = (thread_data*) arg;
  for(int i=data->begin+1; i<= data->end && i<=N; i++){ 
    //printf("Iteración:%i - begin:%i - end:%i\n", i, data->begin, data->end);
    *(data->sum) += i;    
  }
}

int main(){
  // Inicialización de los elementos para cada hilo
  pthread_t threads[THREADS_NUMBER]; 
  thread_data thread_args[THREADS_NUMBER];

  // Para crear un arreglo inicializado con ceros con el tamaño dado por la directiva
  int* sum_arr = (int*)calloc(THREADS_NUMBER, sizeof(int));  
  int thread_size = (int)ceil((float)N/(float)THREADS_NUMBER);    

  // Creación de los hilos e inicialización de los argumentos
  for(int i = 0; i<THREADS_NUMBER; i++){
    thread_args[i].sum = &sum_arr[i];    
    thread_args[i].thread_number = i;
    thread_args[i].begin = i*thread_size;
    thread_args[i].end = (i+1)*thread_size;    
    int r = pthread_create(&threads[i], NULL, sum, &thread_args[i]);
    if (r != 0)
      perror("Error al crear el hilo");
  }

  // Espera de que acaben los hilos
  for(int i = 0; i<THREADS_NUMBER; i++){
    if (pthread_join(threads[i], NULL) != 0) {
        fprintf(stderr, "Error joining thread %d\n", i);
        exit(1);
    }
  }

  // Presentación de resultados
  int total_sum = 0;
  for(int i = 0; i<THREADS_NUMBER; i++){
    printf("sum_arr[%i]: %i \n", i, sum_arr[i]);
    total_sum += sum_arr[i];
  }

  printf("total_sum: %i \n" , total_sum);

  free(sum_arr);
  return 0;
}
