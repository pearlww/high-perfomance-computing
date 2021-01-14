/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include "print.h"

#ifdef _JACOBI
#include "jacobi.h"
#endif

#ifdef _GAUSS_SEIDEL
#include "gauss_seidel.h"
#endif

#define N_DEFAULT 100


int main(int argc, char *argv[]) {

    int 	N = N_DEFAULT;
    int 	iter_max = 1000;
    double	tolerance;
    double	start_T;
    int		output_type = 0;
    char	*output_prefix = "poisson_res";
    char        *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    
    /* get the paramters from the command line */
    N         = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[4]);  // start T for all inner grid points
    if (argc == 6) {
	output_type = atoi(argv[5]);  // ouput type
    }

    // allocate memory
    double 	***u = d_malloc_3d(N+2, N+2, N+2);
    double 	***u_old = d_malloc_3d(N+2, N+2, N+2);
    double 	***f = d_malloc_3d(N+2, N+2, N+2);
    if (u == NULL || u_old == NULL || f==NULL ) {
        perror("array u: allocation failed");
        exit(-1);
    }

    //initialization
    for(int i=0;i<N+2;i++){
        for(int j=0;j<N+2;j++){
            for(int k=0;k<N+2;k++){
                if(i==0||i==N+1||k==0||k==N+1||j==N+1){
                    u[i][j][k] = 20;
                    u_old[i][j][k] = 20;
                    //printf("wall20 ");
                    }
                else if(j==0)
                    {u[i][j][k] = 0;
                    u_old[i][j][k] = 0;
                    //printf("wall0 ");
                    }
                else{
                    u[i][j][k] = start_T;
                    u_old[i][j][k] = start_T;}
            }
        }
    }

    double lx=0, ux=5.0/16.0*(N+2);
    double ly=0, uy=0.25*(N+2);
    double lz=1.0/6.0, uz=0.5*(N+2);

    for(int i=0; i<N+2; i++){
        for(int j=0; j<N+2; j++){
            for(int k=0; k<N+2; k++){
                if(k>=lx && k<=ux && j>=ly && j<=uy && i>=lz && i<=uz)
                    f[i][j][k] = 200;
                else
                    f[i][j][k] = 0;
            }
        }
    }

    //update
    int num_iter;
    #ifdef _JACOBI
    num_iter = jacobi(u, u_old, f, N, iter_max, tolerance);
    #endif

    #ifdef _GAUSS_SEIDEL
    num_iter = gauss_seidel(u, f, N, iter_max, tolerance);
    #endif

    printf("number of iterations = %d \n", num_iter);

    // dump  results if wanted 
    switch(output_type) {
	case 0:
	    // no output at all
	    break;
	case 3:
	    output_ext = ".bin";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N+2, output_ext);
	    fprintf(stderr, "Write binary dump to %s: ", output_filename);
	    print_binary(output_filename, N+2, u);
	    break;
	case 4:
	    output_ext = ".vtk";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N+2, output_ext);
	    fprintf(stderr, "Write VTK file to %s: ", output_filename);
	    print_vtk(output_filename, N+2, u);
	    break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // de-allocate memory
    free(u);

    return(0);
}
