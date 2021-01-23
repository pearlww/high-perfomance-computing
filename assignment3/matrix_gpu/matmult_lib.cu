extern "C" {
	#include <cblas.h>

    void matmult_lib(int m, int n, int k, double *A, double *B, double *C) {
    double alpha = 1.0, beta = 0.0;
    int lda=k, ldb=n, ldc=n;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
}

