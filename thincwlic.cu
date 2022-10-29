#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "def.h"


__global__ void  normal3d_st1
//==========================================================
//
//  THINC/WLIC METHOD ON GPU
//
//
(
    double    dx,
    double    *vf,
    double    *nx2,
    double    *ny2,
    double    *nz2,
    int       nx,
    int       ny,
    int       nz,
    int       n
)
//----------------------------------------------------------
{
    int       i, ix, iy, iz;


    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= n) return;

    iz = (int)(i/(nx*ny));
    iy = (int)(i/nx) - ny*iz;
    ix = i - nx*iy - nx*ny*iz;
    if(ix == 0 || iy == 0 || iz == 0) return;

    nx2[i] = 0.25/dx*(vf[i]          - vf[i-1]
                    + vf[i-nx]       - vf[i-1-nx]
                    + vf[i-nx*ny]    - vf[i-1-nx*ny]
                    + vf[i-nx-nx*ny] - vf[i-1-nx-nx*ny]);
    ny2[i] = 0.25/dx*(vf[i]          - vf[i-nx]
                    + vf[i-1]        - vf[i-1-nx]
                    + vf[i-nx*ny]    - vf[i-nx-nx*ny]
                    + vf[i-1-nx*ny]  - vf[i-1-nx-nx*ny]);
    nz2[i] = 0.25/dx*(vf[i]          - vf[i-nx*ny]
                    + vf[i-1]        - vf[i-1-nx*ny]
                    + vf[i-nx]       - vf[i-nx-nx*ny]
                    + vf[i-1-nx]     - vf[i-1-nx-nx*ny]);

    return;
}


__global__ void  normal3d_st2
//==========================================================
//
//  THINC/WLIC METHOD ON GPU
//
//
(
    double    *vf,
    double    *nx1,
    double    *ny1,
    double    *nz1,
    double    *nx2,
    double    *ny2,
    double    *nz2,
    int       nx,
    int       ny,
    int       nz,
    int       n
)
//----------------------------------------------------------
{
    int       i, ix, iy, iz;


    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= n) return;

    iz = (int)(i/(nx*ny));
    iy = (int)(i/nx) - ny*iz;
    ix = i - nx*iy - nx*ny*iz;
    if(ix == 0    || iy == 0    || iz == 0   ) return;
    if(ix == nx-1 || iy == ny-1 || iz == nz-1) return;

    nx1[i] = 0.125*(nx2[i]      + nx2[i+nx*ny]
                  + nx2[i+1]    + nx2[i+1+nx*ny]
                  + nx2[i+nx]   + nx2[i+nx+nx*ny]
                  + nx2[i+1+nx] + nx2[i+1+nx+nx*ny]);
    ny1[i] = 0.125*(ny2[i]      + ny2[i+nx*ny]
                  + ny2[i+1]    + ny2[i+1+nx*ny]
                  + ny2[i+nx]   + ny2[i+nx+nx*ny]
                  + ny2[i+1+nx] + ny2[i+1+nx+nx*ny]);
    nz1[i] = 0.125*(nz2[i]      + nz2[i+nx*ny]
                  + nz2[i+1]    + nz2[i+1+nx*ny]
                  + nz2[i+nx]   + nz2[i+nx+nx*ny]
                  + nz2[i+1+nx] + nz2[i+1+nx+nx*ny]);

    return;
}

__global__ void  wlic1d4md_st1
//==========================================================
//
//  THINC/WLIC METHOD ON GPU
//
//
(
    double    *flux,
    double    *nx1,
    double    *ny1,
    double    *nz1,
    double    dt,
    double    dx,
    int       nx,
    int       ny,
    int       nz,
    int       n,
    double    *u,
    double    *v,
    double    *w,
    double    *vf,
    int       *bc,
    int       idr
)
//----------------------------------------------------------
{
    double    nn, alpha, beta = 2.0,
              xc, a1, a3, a4, a5, weight,
              um, up, uc, n1;
    int       i, ix, iy, iz, j, isgn, ib,
              ii, iix, iiy, iiz, acs;


    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= n) return;

    flux[i] = 0.0;

    iz = (int)(i/(nx*ny));
    iy = (int)(i/nx) - ny*iz;
    ix = i - nx*iy - nx*ny*iz;
    if(ix == 0    || iy == 0    || iz == 0   ) return;
    if(ix == 1    || iy == 1    || iz == 1   ) return;
    if(ix == nx-1 || iy == ny-1 || iz == nz-1) return;

// Velocity at cell surface ---
    if(idr == 0){ acs = 1;     up = u[i]; um = u[i-acs]; }
    if(idr == 1){ acs = nx;    up = v[i]; um = v[i-acs]; }
    if(idr == 2){ acs = nx*ny; up = w[i]; um = w[i-acs]; }
    uc = (up + um)/2.0;

// Flux ---
    if(uc <= 0.0){
        isgn = 0;
        ii = i;
    }
    else{
        isgn = 1;
        ii = i-acs;
    }
    iiz = (int)(ii/(nx*ny));
    iiy = (int)(ii/nx) - ny*iiz;
    iix = ii - nx*iiy - nx*ny*iiz;

    nn = 0.0;
    for(j = 0;j < 3;j++){
        nn = fabs(nx1[ii]) + fabs(ny1[ii]) + fabs(nz1[ii]);
    }

    if(vf[ii] >= 1.0-1.0e-6 || vf[ii] <= 1.0e-6){
//      flux[i] = vf[ii]*uc*(VTYPE)dt;
        flux[i] = vf[ii]*uc*(double)M*dt;
    }
    else if(nn <= 1.0e-6){
//      flux[i] = vf[ii]*uc*(VTYPE)dt;
        flux[i] = vf[ii]*uc*(double)M*dt;
    }
    else{
        if(iix-1 < 0) ib = ii; else ib = ii-acs;
        if(iiy-1 < 0) ib = ii; else ib = ii-acs;
        if(iiz-1 < 0) ib = ii; else ib = ii-acs;

        if(vf[ib] <= vf[ii+acs]) alpha =  1.0;
        else                     alpha = -1.0;

        a1 = expf(beta*(2.0*vf[ii]-1.0)/alpha);
        a3 = expf(beta);
        xc = 0.5/beta*log((a3*a3 - a1*a3)/(a1*a3 - 1.0));
//      a4 = coshf(beta*((double)isgn - uc*(VTYPE)dt - xc));
        a4 = coshf(beta*((double)isgn - uc*(double)M*dt/dx - xc));
        a5 = coshf(beta*((double)isgn - xc));
//      flux[i] = 0.5*(uc*(VTYPE)dt - alpha/beta*log(a4/a5));
        flux[i] = 0.5*(uc*(double)M*dt - alpha*dx/beta*log(a4/a5));

        if(idr == 0) n1 = nx1[ii];
        if(idr == 1) n1 = ny1[ii];
        if(idr == 2) n1 = nz1[ii];
        weight  = fabs(n1)/nn;
//      flux[i] = flux[i]*weight + vf[ii]*uc*(VTYPE)dt*(1.0-weight);
        flux[i] = flux[i]*weight + vf[ii]*uc*(double)M*dt*(1.0-weight);
    }

// Boundary condition ---
    if(bc[i] == WALL || bc[i-acs] == WALL) flux[i] = 0.0;

    return;
}


__global__ void  wlic1d4md_st2
//==========================================================
//
//  THINC/WLIC METHOD ON GPU
//
//
(
    double    *flux,
    double    dx,
    int       nx,
    int       ny,
    int       nz,
    int       n,
    double    *vf,
    int       *bc,
    int       idr
)
//----------------------------------------------------------
{
    int       i, acs;


    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= n)        return;
    if(bc[i] != COMP) return;

    if(idr == 0) acs = 1;
    if(idr == 1) acs = nx;
    if(idr == 2) acs = nx*ny;

//  vf[i] = vf[i] - (flux[i+acs] - flux[i]);
    vf[i] = vf[i] - (flux[i+acs] - flux[i])/dx;

    return;
}


__global__ void  wlic1d4md_st3
//==========================================================
//
//  THINC/WLIC METHOD ON GPU
//
//
(
    double    *flux,
    int       nx,
    int       ny,
    int       nz,
    int       n,
    double    *vf,
    int       *bc,
    int       idr
)
//----------------------------------------------------------
{
    int       i, i1, i2;


    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= n)        return;
    if(bc[i] == COMP) return;

    i1 = i;
    i2 = i1;

#if defined(CapsuleSHEARFLOW)
    int  ix, iy, iz; 
    iz = (int)(i/(nx*ny));
    iy = (int)(i/nx) - ny*iz;
    ix = i - nx*iy - nx*ny*iz;

    if(bc[i] == BUFF){
        if(ix <= 1)    i2 += (nx-4);
        if(ix >= nx-2) i2 -= (nx-4);
        if(iz <= 1)    i2 += nx*ny*(nz-4);
        if(iz >= nz-2) i2 -= nx*ny*(nz-4);
        vf[i1] = vf[i2];
    }
#elif defined(CapsuleCHANNELFLOW)
//    else if(bc[i] == WALL){
//        vf[i] = 0.0;
//    }
    int  iz;
    iz = (int)(i/(nx*ny));

    if(bc[i] == BUFF){
        if(iz <= 1)    i2 += nx*ny*(nz-4);
        if(iz >= nz-2) i2 -= nx*ny*(nz-4);
        vf[i1] = vf[i2];
    }
#endif
    else if(bc[i] == WALL){
        vf[i] = 0.0;
    }

    return;
}


void  thinc_wlic
//==========================================================
//
//  THINC/WLIC METHOD
//
//
(
    domain    *cdo,
    lattice   *ltc
)
//----------------------------------------------------------
{
#if defined(KARMAN) || defined(SHEARFLOW) || defined(CHANNELFLOW)
  return;
#elif defined(CapsuleSHEARFLOW) || defined(CapsuleCHANNELFLOW)
  int            idr;
  int            thread = 128, block = cdo->n/thread+1;
  dim3           dimB(thread,1,1), dimG(block,1,1);

  #if(N_R > 0)
  static double  *nx1D, *ny1D, *nz1D,
                 *nx2D, *ny2D, *nz2D,
                 *fluxD;
  static int     n1st = 0;

//llocate ---
  if(n1st == 0){
      checkCudaErrors(cudaMalloc((void **)&fluxD,sizeof(double)*cdo->n));
      checkCudaErrors(cudaMalloc((void **)&nx1D, sizeof(double)*cdo->n));
      checkCudaErrors(cudaMalloc((void **)&ny1D, sizeof(double)*cdo->n));
      checkCudaErrors(cudaMalloc((void **)&nz1D, sizeof(double)*cdo->n));
      checkCudaErrors(cudaMalloc((void **)&nx2D, sizeof(double)*cdo->n));
      checkCudaErrors(cudaMalloc((void **)&ny2D, sizeof(double)*cdo->n));
      checkCudaErrors(cudaMalloc((void **)&nz2D, sizeof(double)*cdo->n));

      n1st++;
  }

//HINC/WLIC method ---
  for(idr = 0;idr < 3;idr++){
      normal3d_st1 <<< dimG,dimB >>>
      (cdo->dx,ltc->vfD,nx2D,ny2D,nz2D,cdo->nx,cdo->ny,cdo->nz,cdo->n);

      normal3d_st2 <<< dimG,dimB >>>
      (ltc->vfD,nx1D,ny1D,nz1D,nx2D,ny2D,nz2D,cdo->nx,cdo->ny,cdo->nz,cdo->n);

      wlic1d4md_st1 <<< dimG,dimB >>>
      (fluxD,nx1D,ny1D,nz1D,
       cdo->dt,cdo->dx,cdo->nx,cdo->ny,cdo->nz,cdo->n,
       ltc->umD,ltc->vmD,ltc->wmD,ltc->vfD,ltc->bcD,idr);

      wlic1d4md_st2 <<< dimG,dimB >>>
      (fluxD,cdo->dx,cdo->nx,cdo->ny,cdo->nz,cdo->n,ltc->vfD,ltc->bcD,idr);

      wlic1d4md_st3 <<< dimG,dimB >>>
      (fluxD,cdo->nx,cdo->ny,cdo->nz,cdo->n,ltc->vfD,ltc->bcD,idr);
  }

  cudaThreadSynchronize();
  #endif // N_R
  #if(N_W > 0)
  static double  *nx1D_w, *ny1D_w, *nz1D_w,
                 *nx2D_w, *ny2D_w, *nz2D_w,
                 *fluxD_w;
  static int     n1st_w = 0;

//llocate ---
  if(n1st_w == 0){
      checkCudaErrors(cudaMalloc((void **)&fluxD_w,sizeof(double)*cdo->n));
      checkCudaErrors(cudaMalloc((void **)&nx1D_w, sizeof(double)*cdo->n));
      checkCudaErrors(cudaMalloc((void **)&ny1D_w, sizeof(double)*cdo->n));
      checkCudaErrors(cudaMalloc((void **)&nz1D_w, sizeof(double)*cdo->n));
      checkCudaErrors(cudaMalloc((void **)&nx2D_w, sizeof(double)*cdo->n));
      checkCudaErrors(cudaMalloc((void **)&ny2D_w, sizeof(double)*cdo->n));
      checkCudaErrors(cudaMalloc((void **)&nz2D_w, sizeof(double)*cdo->n));

      n1st_w++;
  }

//HINC/WLIC method for WBC ---
  for(idr = 0;idr < 3;idr++){
      normal3d_st1 <<< dimG,dimB >>>
      (cdo->dx,ltc->vfD2,nx2D_w,ny2D_w,nz2D_w,cdo->nx,cdo->ny,cdo->nz,cdo->n);

      normal3d_st2 <<< dimG,dimB >>>
      (ltc->vfD2,nx1D_w,ny1D_w,nz1D_w,nx2D_w,ny2D_w,nz2D_w,cdo->nx,cdo->ny,cdo->nz,cdo->n);

      wlic1d4md_st1 <<< dimG,dimB >>>
      (fluxD_w,nx1D_w,ny1D_w,nz1D_w,
       cdo->dt,cdo->dx,cdo->nx,cdo->ny,cdo->nz,cdo->n,
       ltc->umD,ltc->vmD,ltc->wmD,ltc->vfD2,ltc->bcD,idr);

      wlic1d4md_st2 <<< dimG,dimB >>>
      (fluxD_w,cdo->dx,cdo->nx,cdo->ny,cdo->nz,cdo->n,ltc->vfD2,ltc->bcD,idr);

      wlic1d4md_st3 <<< dimG,dimB >>>
      (fluxD_w,cdo->nx,cdo->ny,cdo->nz,cdo->n,ltc->vfD2,ltc->bcD,idr);
  }
  cudaThreadSynchronize();
  #endif // N_W

  return;
#endif // CapsuleSHEARFLOW || CapsuleCHANNELFLOW
}

