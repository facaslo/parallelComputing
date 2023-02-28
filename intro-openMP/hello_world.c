#include "omp.h"
#include "stdlib.h"
#include "stdio.h"

void main()
{  
  #pragma omp parallel
  {
    int ID = omp_get_thread_num();
    printf("Hello (%d) \n", ID);
    printf("World (%d) \n", ID);
  }  
}
