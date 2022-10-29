#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "def.h"


//==========================================================
//
//  JACOBI METHOD
//
//

__global__ void
jacobi_1(int *d_ptr,int *d_index,double *d_value,double *d_i,double *d_o,int n)
{
    int       tid = blockIdx.x*blockDim.x + threadIdx.x;
    int       i, jj, max = d_ptr[tid];
    double    sum = 0.0;

    for(i = 0; i < max; i++){
        jj  = i*n + tid;
        if(d_index[jj] == tid) continue;
        sum += d_value[jj]*d_i[d_index[jj]];
    }
    d_o[tid] = sum;
}

__global__ void
jacobi_2(double *d_i1, double *d_i2, double *d_o)
{
    int       tid = blockIdx.x*blockDim.x + threadIdx.x;

    d_o[tid] = d_i1[tid] - d_i2[tid];
}

__global__ void
jacobi_3(int *d_ptr,int *d_index,double *d_value,double *d_i1,double *d_i2,double *d_o,int n)
{
    int       tid = blockIdx.x*blockDim.x + threadIdx.x;
    int       i, jj, max = d_ptr[tid];
    double    sum = 0.0;

    for(i = 0; i < max; i++){
        jj  = i*n + tid;
        sum += d_value[jj]*d_i1[d_index[jj]];
    }
    d_o[tid] = d_i2[tid] - sum;
}

int  jacobi
//==========================================================
//
//  JACOBI METHOD ON GPU
//
//
(
    int       *d_ptr,
    int       *d_index,
    double    *d_value,
    double    *d_x,
    double    *d_b,
    double    *d_r,
    double    *d_m,
    double    *d_o,
    int       n,
    int       nnz
)
//----------------------------------------------------------
{
    double    err, b_dot_b;
    int       iter, m_size = sizeof(double)*n;
    int       thread = 256, block = n/thread; 
    dim3      dimB(thread,1,1), dimG(block,1,1);


    if(block > 65535){ printf("Matrix Error: Block Overflow\n"); exit(1); }

// Allocate local variable ---
    checkCudaErrors(cudaMemset(d_r, 0,m_size));
    checkCudaErrors(cudaMemset(d_m, 0,m_size));

// JACOBI method ---
    b_dot_b = dot_product(d_b,d_b,d_o,n);

    for(iter = 0; iter < ITRMAX; iter++){
        jacobi_1<<<dimG,dimB>>>(d_ptr,d_index,d_value,d_x,d_r,n);
        jacobi_2<<<dimG,dimB>>>(d_b,d_r,d_x);
        jacobi_3<<<dimG,dimB>>>(d_ptr,d_index,d_value,d_x,d_b,d_m,n);

        err = sqrt(dot_product(d_m,d_m,d_o,n)/b_dot_b);
//        printf("LOOP : %d\t Error: %e\n",iter,err);
        if(err <  RESIDUAL) break;
    }

    if(iter >= ITRMAX){ printf("Matrix Error: Not Converge\n"); return 2; }

    return 0;
}


//==========================================================
//
//  RED-BLACK SOR METHOD ON GPU
//
//

__global__ void
redblacksor_1(int *d_ptr,int *d_index,double *d_value,double *d_x,double *d_b,double *d_r,
              double alpha,int n,int nx,int ny,int nz,int nn)
{
    int       tid = blockIdx.x*blockDim.x + threadIdx.x;
    int       i, ix, iy, iz, ii, jj, max = d_ptr[tid];
    double    sum = 0.0;

    if(tid < nn){
        iz = (int)(tid/(nx*ny));
        iy = (int)(tid/nx) - ny*iz;
        ix = tid - nx*iy - nx*ny*iz;
        ii = ix + iy + iz;
    }
    else{
        ii = tid;
    }

    if(ii%2 == 1) return;
    for(i = 0; i < max; i++){
        jj  = i*n + tid;
        sum += d_value[jj]*d_x[d_index[jj]];
    }
    d_r[tid] = d_b[tid] - sum;
    d_x[tid] += alpha*d_r[tid];
}

__global__ void
redblacksor_2(int *d_ptr,int *d_index,double *d_value,double *d_x,double *d_b,double *d_r,
              double alpha,int n,int nx,int ny,int nz,int nn)
{
    int       tid = blockIdx.x*blockDim.x + threadIdx.x;
    int       i, ix, iy, iz, ii, jj, max = d_ptr[tid];
    double    sum = 0.0;

    if(tid < nn){
        iz = (int)(tid/(nx*ny));
        iy = (int)(tid/nx) - ny*iz;
        ix = tid - nx*iy - nx*ny*iz;
        ii = ix + iy + iz;
    }
    else{
        ii = tid;
    }

    if(ii%2 == 0) return;
    for(i = 0; i < max; i++){
        jj  = i*n + tid;
        sum += d_value[jj]*d_x[d_index[jj]];
    }
    d_r[tid] = d_b[tid] - sum;
    d_x[tid] += alpha*d_r[tid];
}

int  redblacksor
//==========================================================
//
//  RED-BLACK SOR METHOD ON GPU
//
//
(
    int       *d_ptr,
    int       *d_index,
    double    *d_value,
    double    *d_x,
    double    *d_b,
    double    *d_r,
    double    *d_o,
    int       n,
    int       nnz,
    int       nx,
    int       ny,
    int       nz,
    int       nn
)
//----------------------------------------------------------
{
    double    err, b_dot_b, alpha = 1.3;
    int       iter, m_size = sizeof(double)*n;
    int       thread = 512, block = n/thread; 
//    int       thread = 1024, block = n/thread; //+1; 
    dim3      dimB(thread,1,1), dimG(block,1,1);


    if(block > 65535){ printf("Matrix Error: Block Overflow\n"); exit(1); }

// Allocate local variable ---
    checkCudaErrors(cudaMemset(d_r, 0, m_size));

// Red-Black SOR method ---
    b_dot_b = dot_product(d_b,d_b,d_o,n);

    for(iter = 0; iter < ITRMAX; iter++){
        redblacksor_1<<<dimG,dimB>>>(d_ptr,d_index,d_value,d_x,d_b,d_r,alpha,n,nx,ny,nz,nn);
        redblacksor_2<<<dimG,dimB>>>(d_ptr,d_index,d_value,d_x,d_b,d_r,alpha,n,nx,ny,nz,nn);

        err = sqrt(dot_product(d_r,d_r,d_o,n)/b_dot_b);
//        printf("LOOP : %d\t Error: %e\n",iter,err);
        if(err < RESIDUAL) break;
    }

    if(iter >= ITRMAX){ printf("Matrix Error: Not Converge\n"); return 2; }

    return 0;
}


//==========================================================
//
//  BiCGSTAB METHOD
//
//

__global__ void 
vec_init(double *d_b, double *d_r, double *d_ra, double *d_p)
{
    int       tid = blockIdx.x*blockDim.x + threadIdx.x;
    double    t   = d_b[tid] - d_r[tid];

    d_r[tid]  = t;
    d_ra[tid] = t;
    d_p[tid]  = t;
}

__global__ void 
r_update(double alfa, double *d_v, double *d_r)
{
    int       tid = blockIdx.x*blockDim.x + threadIdx.x;
    double    r   = d_r[tid];

    d_r[tid] = r - alfa*d_v[tid];
}

__global__ void
x_r_update(double alfa,double *d_p,double omeg,double *d_r,double *d_x,double *d_t)
{
    int       tid = blockIdx.x*blockDim.x + threadIdx.x;
    double    r   = d_r[tid];
    double    x   = d_x[tid];

    d_x[tid] = x + alfa*d_p[tid] + omeg*r;
    d_r[tid] = r - omeg*d_t[tid];
}

__global__ void
p_update(double *d_r, double beta, double omeg, double *d_v, double *d_p)
{
    int       tid = blockIdx.x*blockDim.x + threadIdx.x;
    double    p   = d_p[tid];

    d_p[tid] = d_r[tid] + beta*(p - omeg*d_v[tid]);
}

__global__ void
spmv_ell(int *d_ptr,int *d_index,double *d_value,double *d_i,double *d_o,int n)
{
    int       tid = blockIdx.x*blockDim.x + threadIdx.x;
    int       i, jj, max = d_ptr[tid];
    double    sum = 0.0;

    for(i = 0; i < max; i++){
        jj  = i*n + tid;
        sum += d_value[jj]*d_i[d_index[jj]];
    }
    d_o[tid] = sum;
}

int  bicgstab
//==========================================================
//
//  BiCGSTAB METHOD ON GPU
//
//
(
    int       *d_ptr,
    int       *d_index,
    double    *d_value,
    double    *d_x,
    double    *d_b,
    double    *d_ra,
    double    *d_r,
    double    *d_p,
    double    *d_m,
    double    *d_n,
    double    *d_o,
    int       n,
    int       nnz
)
//----------------------------------------------------------
{
    double         err, alfa = 1.0, beta = 0.0, omeg = 1.0;
    double         b_dot_b, r_dot_ra, r_dot_ra_old;
    int            iter, m_size = sizeof(double)*n;
    int            thread = 256, block = n/thread; 
//    int            thread = 512, block = n/thread; 
    dim3           dimB(thread,1,1), dimG(block,1,1);


    if(block > 65535){ printf("Matrix Error: Block Overflow\n"); exit(1); }

// Allocate local variable ---
    checkCudaErrors(cudaMemset(d_ra,0,m_size));
    checkCudaErrors(cudaMemset(d_r, 0,m_size));
    checkCudaErrors(cudaMemset(d_p, 0,m_size));
    checkCudaErrors(cudaMemset(d_m, 0,m_size));
    checkCudaErrors(cudaMemset(d_n, 0,m_size));

// BiCGSTAB method ---
    //
    // r  = A*x
    // r  = b - r
    // ra = r
    // p  = r
    //
    spmv_ell<<<dimG,dimB>>>(d_ptr,d_index,d_value,d_x,d_r,n);
    vec_init<<<dimG,dimB>>>(d_b, d_r, d_ra, d_p); 
    b_dot_b  = dot_product(d_b, d_b, d_o, n);

    for(iter = 0; iter < ITRMAX; iter++){
        r_dot_ra_old = r_dot_ra;
        r_dot_ra = dot_product(d_r, d_ra, d_o, n);
//        if(r_dot_ra == 0){ printf("Matrix Error: Iteration %d\n",iter); exit(1); }
        if(r_dot_ra == 0){ printf("Matrix Error: Iteration %d\n",iter); return 1; }

        if(iter > 0){
        //
        // beta = (r_new, ra) / (r_old, ra) * (alfa / omeg)
        // p = r + beta*(p - omeg*m)
        //
            beta = (r_dot_ra / r_dot_ra_old) * (alfa / omeg);
            p_update<<<dimG,dimB>>>(d_r, beta, omeg, d_m, d_p);
        }

        //
        // m = A*p
        // alfa = (r, ra) / (m, ra)
        // r = r - alfa*m
        //
        spmv_ell<<<dimG,dimB>>>(d_ptr,d_index,d_value,d_p,d_m,n);
        alfa = r_dot_ra/dot_product(d_m, d_ra, d_o, n);
        r_update<<<dimG,dimB>>>(alfa, d_m, d_r);

        //
        // n = A*r
        // omeg = (r, n) / (n, n)
        // x = x + alfa*p + omeg*r
        // r = r - omeg*n
        //
        spmv_ell<<<dimG,dimB>>>(d_ptr,d_index,d_value,d_r,d_n,n);
        omeg = dot_product(d_r, d_n, d_o, n)/dot_product(d_n, d_n, d_o, n);

        x_r_update<<<dimG,dimB>>>(alfa, d_p, omeg, d_r, d_x, d_n);

        //
        // Residual
        //
        err = sqrt(dot_product(d_r, d_r, d_o, n)/b_dot_b);
//        printf("LOOP : %d\t Error: %e\n",iter,err);
        if(err <  RESIDUAL) break;
    }

//    if(iter >= ITRMAX){ printf("Matrix Error: Not Converge\n"); exit(1); }
    if(iter >= ITRMAX){ printf("Matrix Error: Not Converge\n"); return 2; }

    return 0;
}

