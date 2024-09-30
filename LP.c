#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

extern void dgesv_(int* N, int* NRHS, double* A, int* LDA, int* IPIV, 
                   double* B, int* LDB, int* INFO);

extern void dgemv_(char *TRANS, int *M, int *N, double *ALPHA,
                   double *A, int *LDA, double *X, int *INCX,
                   double *BETA, double *Y, int *INCY);

int main() {
    int n = 3;
    int m = 2;

    double *A = malloc(m * n * sizeof(double));
    double *b = malloc(m * sizeof(double));
    double *c = malloc(n * sizeof(double));

    double *x = malloc(n * sizeof(double));
    double *lmbda = malloc(m * sizeof(double));
    double *s = malloc(n * sizeof(double));

    for (int i = 0; i < m*n; i++) {
    //    A[i] = drand48();
        A[i] = i;
    }

    for (int i = 0; i < n; i++) {
         c[i] = drand48();
         x[i] = drand48();
//         s[i] = drand48();
         s[i] = -i;
    }

    for (int i = 0; i < m; i++) {
        b[i] = drand48();
//        lmbda[i] = drand48();
        lmbda[i] = i;
    }

    int LDA = n;
    double alpha = 1.0;
    double beta = 0.0;
    int incX = 1;
    int incY = 1;
    
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, A, LDA, x, incX, beta, b, incY);


    cblas_dcopy(n, s, 1, c, 1); // We copy s into c.
    beta = 1.0;
    
    for (int i = 0; i < m * n; i++) {
        printf("A[%d] = %f\n", i, A[i]);
    }
    for (int i = 0; i < n; i++) {
        printf("s[%d] = %f\n", i, s[i]);
    }
    for (int i = 0; i < m; i++) {
        printf("lmbda[%d] = %f\n", i, lmbda[i]);
    }
    cblas_dgemv(CblasRowMajor, CblasTrans, m, n, alpha, A, LDA, lmbda, incX, beta, c, incY); 

    for (int i = 0; i < n; i++) {
        printf("c[%d] = %f\n", i, c[i]);
    }

    
//    int N = 3;
//    int NRHS = 1;
//    int LDA = N;
//    int LDB = N;
//    int INFO;
//    int IPIV[N];
//
//    double A[9] = {2.0, 1.0, 1.0, 4.0, -6.0, 0.0, -2.0, 7.0, 2.0};
//    double B[3] = {5.0, -2.0, 9.0};
//
//    dgesv_(&N, &NRHS, A, &LDA, IPIV, B, &LDB, &INFO);
//
//    if (INFO == 0) {
//        // Success: the solution is stored in B.
//        printf("Solution:\n");
//        for (int i = 0; i < N; i++) {
//            printf("x[%d] = %f\n", i, B[i]);
//        }
//    } else {
//        printf("An error occurred: INFO = %d\n", INFO);
//    }


    return 0;
}
