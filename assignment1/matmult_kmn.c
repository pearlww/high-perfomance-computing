void
matmult_kmn(int m, int n, int k, double **A, double **B, double **C) {

    for(int s = 0; s < k; s++){
	    for(int i = 0; i < m; i++){
            double cij =0;
            for(int j = 0; j < n; j++){
                cij += A[i][s] * B[s][j];
            }
            C[i][j] =cij;
        }    
    }

}
