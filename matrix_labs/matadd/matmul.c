void
matmul(int m, int n, int k, double **A, double **B, double **C) {
    
    int i, j, s;

    for(i = 0; i < m; i++){
	    for(j = 0; j < n; j++){
            double cij =0;
            for(s=0; s<k; s++){
                cij += A[i][s] * B[s][j];
            }
            C[i][j] =cij;
        }    
    }

}