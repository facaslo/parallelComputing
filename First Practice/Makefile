CC = gcc
CFLAGS = -fopenmp
EXE = ./Out
SRCS = sequential_matrix_mult.c

all: sequential_matrix_mult omp_matrix_mult

sequential_matrix_mult: sequential_matrix_mult.c omp_matrix_mult.c
	gcc sequential_matrix_mult.c -o sequential_matrix_mult.out

omp_matrix_mult:
	gcc omp_matrix_mult.c -o omp_matrix_mult.out -fopenmp