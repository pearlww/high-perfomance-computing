void
matmult_knm(int m, int n, int k, double **A, double **B, double **C) {
    for(int i = 0; i < m; i++)
        for(int j =0; j<n; j++)
            C[i][j] = 0;

    for(int s = 0; s < k; s++){
	    for(int j = 0; j < n; j++){
	      for(int i = 0; i < m; i++){
                C[i][j] += A[i][s] * B[s][j];
            }
        }    
    }

}
