void
matmult_mkn(int m, int n, int k, double **A, double **B, double **C) {

    for(int i = 0; i < m; i++){
	    for(int s = 0; s < k; s++){
            	for(int j = 0; j < n; j++){
		  C[i][j] += A[i][s] * B[s][j];
            }
        }    
    }
}