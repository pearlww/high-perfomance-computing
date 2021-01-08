void matmult_blk(int m, int n, int k, double **A, double **B, double **C, int bs){

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

for(int i = 0; i < m; i++){
    for(int j = 0; j < n; j++){
        C[i][j] = 0;
    }
}

for(int s1 =0; s1<k; s1+=bs){
    for(int i = 0; i < m; i++){
	    for(int j = 0; j < n; j++){
            for(int s2=0; s2<MIN(k-s1,bs); s2++){
                C[i][j] += A[i][s1+s2] * B[s1+s2][j];
            }
        }
    }    
}
}