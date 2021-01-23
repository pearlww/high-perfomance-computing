#include <stdio.h>

void main(){
    int i;
    int N = 1000000000;
    double pi = 0;
    double h=1.0/(double) N;

    #pragma omp parallel for default(none) shared(N,h) private(i) reduction(+:pi)
    for (i=1;i<=N;i++){
        pi += 4.0/(1.0+ ((i-0.5)*h)*((i-0.5)*h) );
    }//end of omp directive

    pi=pi*h;
    printf("pi= %f \n", pi);
}