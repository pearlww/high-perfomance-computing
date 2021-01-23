#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "datatools.h"		/* helper functions	        */
#include "matadd.h"		/* my matrix add fucntion	*/
#include "matmul.h"
#include "cblas.h"

#define NREPEAT 100		/* repeat count for the experiment loop */

#define mytimer clock
#define delta_t(a,b) (1e3 * (b - a) / CLOCKS_PER_SEC)

int
main(int argc, char *argv[]) {

    int    i, m, n, k, N = NREPEAT;
    double **A, **B, **C;
    double tcpu1; 

    clock_t t1, t2;

    //for (m = 200; m <= 3500; m += 300) {
		m = 3;
		n = 2; //m + 25;
		k = 5; //m + 50;

		/* Allocate memory */
		A = malloc_2d(m, k);
		B = malloc_2d(k, n);
		C = malloc_2d(m, n);
		if (A == NULL || B == NULL | C == NULL) {
			fprintf(stderr, "Memory allocation error...\n");
			exit(EXIT_FAILURE);
		}

		/* initialize with useful data - last argument is reference */
		init_data(m,n,k,A,B);

		/* timings for matadd */
		t1 = mytimer();
		for (i = 0; i < N; i++)
			matmul(m, n, k, A, B, C);
		t2 = mytimer();
		tcpu1 = delta_t(t1, t2) / N;

		//check_results("main", m, n, k, C);

		/* Print n and results  */
		printf("%4d %4d %4d %8.3f\n", m, n, k, tcpu1);
		print_matrix(m,n,C);
		
		/* Free memory */
		free_2d(A);
		free_2d(B);
		free_2d(C);
    //}

    return EXIT_SUCCESS;
}
