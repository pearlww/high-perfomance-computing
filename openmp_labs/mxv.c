#include <stdio.h>
#include <stdlib.h>


void mxv(int m, int n, double* restrict a, double* restrict b, double* restrict c){
    int i,j;
    double sum;

    #pragma omp parallel if(m>10|n>10) shared(m,n,a,b,c) private(i,j,sum)
    #pragma omp for
    for(i=0;i<m;i++){
        sum=0.0;
        for(j=0;j<n;j++){
            sum += b[i*n+j]*c[j];
        }
        a[i]=sum;
    }
}

// main
void main(){
    int m = 500;
    int n = 1000; 

    /* Allocate memory */
    double* a = malloc(m*sizeof(double));
    double* b = malloc(m*n*sizeof(double));
    double* c = malloc(n*sizeof(double));

    for( int i = 0; i < m; i++)
        a[i] = 0;	    	    
    for( int i = 0; i < m*n; i++)
        b[i] = (double) rand();	    	    
    for( int i = 0; i < n; i++)
        c[i] = (double) rand();	    	    

    int max_iter = 10000;

    for(int i = 0; i < max_iter; i++)
        mxv(m, n, a, b, c);

    free(a);
    free(b);
    free(c);
}