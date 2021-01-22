/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_H
#define _JACOBI_H

__global__ void jacobi_seq(double ***, double ***, double ***, int);
__global__ void jacobi_nat(double ***, double ***, double ***, int);
__global__ void jacobi_nat_with_norm(double ***, double ***, double ***, int, double*);

__global__ void jacobi_gpu0(double ***, double ***, double ***, double ***, int);
__global__ void jacobi_gpu1(double ***, double ***, double ***, double ***, int);
#endif
