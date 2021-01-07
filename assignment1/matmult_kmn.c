<<<<<<< HEAD

=======
>>>>>>> b4b6c96db473fa790d2a7fb20304c89e1ab077e7
void
matmult_kmn(int m, int n, int k, double **A, double **B, double **C) {

    for(int s = 0; s < k; s++){
	    for(int i = 0; i < m; i++){
<<<<<<< HEAD
              for(int j=0; j<n; j++){
                C[i][j] += A[i][s] * B[s][j];
            }

=======
            for(int j = 0; j < n; j++){
                C[i][j] += A[i][s] * B[s][j];
            }
>>>>>>> b4b6c96db473fa790d2a7fb20304c89e1ab077e7
        }    
    }

}
