void matmult_mnk(int m, int n, int k, double **A, double **B, double **C) {

    for(int i = 0; i < m; i++){
	    for(int j = 0; j < n; j++){
            double cij =0;
            for(int s =0; s<k; s++){
                cij += A[i][s] * B[s][j];
            }
            C[i][j] =cij;
        }    
    }

}
