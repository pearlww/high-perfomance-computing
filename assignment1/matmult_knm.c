void
matmult_knm(int m, int n, int k, double **A, double **B, double **C) {

    for(int s = 0; s < k; s++){
	    for(int j = 0; j < n; j++){
<<<<<<< HEAD
            for(int i = 0; i < m; i++){
=======
	      for(int i = 0; i < m; i++){
>>>>>>> 0832cef2a2229833bf22d94d551e9022dc535785
                C[i][j] += A[i][s] * B[s][j];
            }
        }    
    }

<<<<<<< HEAD
}
=======
}
>>>>>>> 0832cef2a2229833bf22d94d551e9022dc535785
