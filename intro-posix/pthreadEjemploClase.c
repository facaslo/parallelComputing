#include "stdlib.h"
#include "pthread.h"
#include "stddef.h"
#include "stdio.h"

#define num 10

// This is the function that each thread will run
void* func(void* ap) {
    printf("%i ", *(int*)ap); // Print the value passed as argument to the thread
}

int main() {
    pthread_t threads[num]; // Declare an array of thread identifiers
    int arg[num]; // Declare an array of arguments to pass to the threads

    // Create each thread and pass an integer value as an argument
    for (int i = 0; i < num; i++) {
        arg[i] = i; // Set the argument for the thread
        int r = pthread_create(&threads[i], NULL, func, (void*)&arg[i]); // Create the thread
        if (r != 0)
            perror("Error al crear el hilo"); // Print an error message if the thread creation fails
    }

    // Wait for each thread to finish
    for (int i = 0; i < num; i++) {
        pthread_join(threads[i], NULL);
    }

    return 0; // Exit the program
}