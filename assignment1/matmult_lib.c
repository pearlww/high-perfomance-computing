# include "cblas.h"
void matmult_lib(int m, int n, int k, double **A, double **B, double **C) {
    double alpha = 1, beta = 0;
    int lda=k, ldb=n, ldc=n;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, &A[0][0], lda, &B[0][0], ldb, beta, &C[0][0], ldc);
}