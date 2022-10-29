#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "def.h"


__device__ double  atomicDoubleAdd1(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int  old = *address_as_ull, assumed;

    do{
        assumed = old;
        old = atomicCAS(address_as_ull,assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    }while(assumed != old);

    return __longlong_as_double(old);
}


__global__ void  ibm_st1a
//==========================================================
//
//  IMMERSED BOUNDARY METHOD ON GPU
//
//
(
    int       n,
    double    *ltfx,
    double    *ltfy,
    double    *ltfz
)
//----------------------------------------------------------
{
    int       i;


    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= n) return;

    ltfx[i] = 0.0;
    ltfy[i] = 0.0;
    ltfz[i] = 0.0;

    return;
}


__global__ void  ibm_st1b
//==========================================================
//
//  IMMERSED BOUNDARY METHOD ON GPU
//
//
(
    double    dt,
    double    dx,
    int       nx,
    int       ny,
    int       nz,
    double    *ltfx,
    double    *ltfy,
    double    *ltfz,
    double    *x,
    double    *y,
    double    *z,
    double    *fx,
    double    *fy,
    double    *fz,
    int       n,
    int       vertex,
    int       element,
    int       *ele
)
//----------------------------------------------------------
{
    int       i,  jid,
              ix, iy, iz,
              x1, y1, z1,
              x2, y2, z2;
    double    xx,   yy,   zz,
              tx,   ty,   tz,
              xtr,  ytr,  ztr,
              xcos, ycos, zcos, dlt;


    jid = blockDim.x*blockIdx.x + threadIdx.x;
    if(jid >= n*vertex) return;

// Distribute membrane force (delta function) ---
    tx  = x[jid];
    ty  = y[jid];
    tz  = z[jid];
    xtr = tx + (double)(nx-1)*dx/2.0;
    ytr = ty + (double)(ny-1)*dx/2.0;
    ztr = tz + (double)(nz-1)*dx/2.0;
    x1 = (int)floor(xtr/dx) - 1; x2 = (int)ceil(xtr/dx) + 1;
    y1 = (int)floor(ytr/dx) - 1; y2 = (int)ceil(ytr/dx) + 1;
    z1 = (int)floor(ztr/dx) - 1; z2 = (int)ceil(ztr/dx) + 1;

    for(iz = z1;iz <= z2;iz++){
        for(iy = y1;iy <= y2;iy++){
            for(ix = x1;ix <= x2;ix++){
                if(ix < 0 || ix >= nx
                || iy < 0 || iy >= ny
                || iz < 0 || iz >= nz) continue;

                i  = ix + nx*iy + nx*ny*iz;
                xx = fabs((double)ix*dx - xtr);
                yy = fabs((double)iy*dx - ytr);
                zz = fabs((double)iz*dx - ztr);
                xcos = 1.0 + cos(M_PI*xx/(2.0*dx));
                ycos = 1.0 + cos(M_PI*yy/(2.0*dx));
                zcos = 1.0 + cos(M_PI*zz/(2.0*dx));
                dlt  = xcos*ycos*zcos/(64.0*dx*dx*dx);

                atomicDoubleAdd1(&ltfx[i],dlt*fx[jid]);
                atomicDoubleAdd1(&ltfy[i],dlt*fy[jid]);
                atomicDoubleAdd1(&ltfz[i],dlt*fz[jid]);
            }
        }
    }

    return;
}

__global__ void  ibm_st1bb
//==========================================================
//
//  IMMERSED BOUNDARY METHOD ON GPU
//
//
(
    double    dt,
    double    dx,
    int       nx,
    int       ny,
    int       nz,
    double    *ltfx,
    double    *ltfy,
    double    *ltfz,
    double    *x,
    double    *y,
    double    *z,
    double    *fx,
    double    *fy,
    double    *fz,
    double    *f_adh,
    int       n,
    int       vertex,
    int       element,
    int       *ele
)
//----------------------------------------------------------
{
    int       i,  jid,
              ix, iy, iz,
              x1, y1, z1,
              x2, y2, z2;
    double    xx,   yy,   zz,
              tx,   ty,   tz,
              xtr,  ytr,  ztr,
              xcos, ycos, zcos, dlt;

    jid = blockDim.x*blockIdx.x + threadIdx.x;
    if(jid >= n*vertex) return;

// Distribute membrane force (delta function) ---
    tx  = x[jid];
    ty  = y[jid];
    tz  = z[jid];
    xtr = tx + (double)(nx-1)*dx/2.0;
    ytr = ty + (double)(ny-1)*dx/2.0;
    ztr = tz + (double)(nz-1)*dx/2.0;
    x1 = (int)floor(xtr/dx) - 1; x2 = (int)ceil(xtr/dx) + 1;
    y1 = (int)floor(ytr/dx) - 1; y2 = (int)ceil(ytr/dx) + 1;
    z1 = (int)floor(ztr/dx) - 1; z2 = (int)ceil(ztr/dx) + 1;

    for(iz = z1;iz <= z2;iz++){
        for(iy = y1;iy <= y2;iy++){
            for(ix = x1;ix <= x2;ix++){
                if(ix < 0 || ix >= nx
                || iy < 0 || iy >= ny
                || iz < 0 || iz >= nz) continue;

                i  = ix + nx*iy + nx*ny*iz;
                xx = fabs((double)ix*dx - xtr);
                yy = fabs((double)iy*dx - ytr);
                zz = fabs((double)iz*dx - ztr);
                xcos = 1.0 + cos(M_PI*xx/(2.0*dx));
                ycos = 1.0 + cos(M_PI*yy/(2.0*dx));
                zcos = 1.0 + cos(M_PI*zz/(2.0*dx));
                dlt  = xcos*ycos*zcos/(64.0*dx*dx*dx);

//                atomicDoubleAdd1(&ltfx[i],dlt*GSratio*(fx[jid] + f_adh[jid*3 + 0]));
//                atomicDoubleAdd1(&ltfy[i],dlt*GSratio*(fy[jid] + f_adh[jid*3 + 1]));
//                atomicDoubleAdd1(&ltfz[i],dlt*GSratio*(fz[jid] + f_adh[jid*3 + 2]));
                atomicDoubleAdd1(&ltfx[i],dlt*(fx[jid] + f_adh[jid*3 + 0]));
                atomicDoubleAdd1(&ltfy[i],dlt*(fy[jid] + f_adh[jid*3 + 1]));
                atomicDoubleAdd1(&ltfz[i],dlt*(fz[jid] + f_adh[jid*3 + 2]));
            }
        }
    }

    return;
}

__global__ void  ibm_st1c
//==========================================================
//
//  IMMERSED BOUNDARY METHOD ON GPU
//
//
(
    int       cdnx,
    int       cdny,
    int       cdnz,
    int       cdn,
    double    *ltfx,
    double    *ltfy,
    double    *ltfz,
    int       *ltbc
)
//----------------------------------------------------------
{
    int       i;


    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= cdn)        return;
    if(ltbc[i] != COMP) return;

#if defined(CapsuleSHEARFLOW)
    int       ix, iy, iz, i1, i2;
    i1 = i;
    iz = (int)(i/(cdnx*cdny));
    iy = (int)(i/cdnx) - cdny*iz;
    ix = i - cdnx*iy - cdnx*cdny*iz;

    if(ix <= 3){
        i2 = i1 + (cdnx-4);
        ltfx[i1] += ltfx[i2]; ltfy[i1] += ltfy[i2]; ltfz[i1] += ltfz[i2];
    }
    if(ix >= cdnx-4){
        i2 = i1 - (cdnx-4);
        ltfx[i1] += ltfx[i2]; ltfy[i1] += ltfy[i2]; ltfz[i1] += ltfz[i2];
    }

    if(iz <= 3){
        i2 = i1 + cdnx*cdny*(cdnz-4);
        ltfx[i1] += ltfx[i2]; ltfy[i1] += ltfy[i2]; ltfz[i1] += ltfz[i2];
    }
    if(iz >= cdnz-4){
        i2 = i1 - cdnx*cdny*(cdnz-4);
        ltfx[i1] += ltfx[i2]; ltfy[i1] += ltfy[i2]; ltfz[i1] += ltfz[i2];
    }

    if(ix <= 3 && iz <= 3){
        i2 = i1 + (cdnx-4) + cdnx*cdny*(cdnz-4);
        ltfx[i1] += ltfx[i2]; ltfy[i1] += ltfy[i2]; ltfz[i1] += ltfz[i2];
    }
    if(ix >= cdnx-4 && iz <= 3){
        i2 = i1 - (cdnx-4) + cdnx*cdny*(cdnz-4);
        ltfx[i1] += ltfx[i2]; ltfy[i1] += ltfy[i2]; ltfz[i1] += ltfz[i2];
    }
    if(ix <= 3 && iz >= cdnz-4){
        i2 = i1 + (cdnx-4) - cdnx*cdny*(cdnz-4);
        ltfx[i1] += ltfx[i2]; ltfy[i1] += ltfy[i2]; ltfz[i1] += ltfz[i2];
    }
    if(ix >= cdnx-4 && iz >= cdnz-4){
        i2 = i1 - (cdnx-4) - cdnx*cdny*(cdnz-4);
        ltfx[i1] += ltfx[i2]; ltfy[i1] += ltfy[i2]; ltfz[i1] += ltfz[i2];
    }
#elif defined(CapsuleCHANNELFLOW)
    int  iz, i1, i2;
    i1 = i;
    iz = (int)(i/(cdnx*cdny));

    if(iz <= 3){
        i2 = i1 + cdnx*cdny*(cdnz-4);
        ltfx[i1] += ltfx[i2]; ltfy[i1] += ltfy[i2]; ltfz[i1] += ltfz[i2];
    }
    if(iz >= cdnz-4){
        i2 = i1 - cdnx*cdny*(cdnz-4);
        ltfx[i1] += ltfx[i2]; ltfy[i1] += ltfy[i2]; ltfz[i1] += ltfz[i2];
    }
#endif

    return;
}


void  ibm_st1
//==========================================================
//
//  IMMERSED BOUNDARY METHOD
//
//
(
    domain    *cdo,
    lattice   *ltc,
    cell      *cel
)
//----------------------------------------------------------
{
#if defined(KARMAN) || defined(SHEARFLOW) || defined(CHANNELFLOW)
  return;
#elif defined(CapsuleSHEARFLOW) || defined(CapsuleCHANNELFLOW)
  int       thread1 = 128, block1 = cdo->n/thread1+1;
  dim3      dimB1(thread1,1,1), dimG1(block1,1,1);

// Reset force ---
  ibm_st1a <<< dimG1,dimB1 >>>
  (cdo->n,ltc->fxD,ltc->fyD,ltc->fzD);

// Distribute membrane force (RBC) ---
  #if(N_R > 0)
  int       thread2 = 128, block2 = (cel->n*cel->vertex)/thread2+1;
  dim3      dimB2(thread2,1,1), dimG2(block2,1,1);

//  ibm_st1bb <<< dimG2,dimB2 >>>
//  (cdo->dt,cdo->dx,cdo->nx,cdo->ny,cdo->nz,
//   ltc->fxD,ltc->fyD,ltc->fzD,cel->xD,cel->yD,cel->zD,
//   cel->fxD,cel->fyD,cel->fzD,cel->f_adhD,
//   cel->n,cel->vertex,cel->element,cel->eleD);

  ibm_st1b <<< dimG2,dimB2 >>>
  (cdo->dt,cdo->dx,cdo->nx,cdo->ny,cdo->nz,
   ltc->fxD,ltc->fyD,ltc->fzD,cel->xD,cel->yD,cel->zD,
   cel->fxD,cel->fyD,cel->fzD,
   cel->n,cel->vertex,cel->element,cel->eleD);

  #endif //N_R

// Distribute membrane force (WBC) ---
  #if(N_W > 0)
  int       thread3 = 128, block3 = (cel->n_w*cel->vertex_w)/thread3+1;
  dim3      dimB3(thread3,1,1), dimG3(block3,1,1);

  ibm_st1bb <<< dimG3,dimB3 >>>
  (cdo->dt,cdo->dx,cdo->nx,cdo->ny,cdo->nz,
   ltc->fxD,ltc->fyD,ltc->fzD,cel->xD_w,cel->yD_w,cel->zD_w,
   cel->fxD_w,cel->fyD_w,cel->fzD_w,cel->f_adhD,
   cel->n_w,cel->vertex_w,cel->element_w,cel->eleD_w);

/*
  ibm_st1b <<< dimG3,dimB3 >>>
  (cdo->dt,cdo->dx,cdo->nx,cdo->ny,cdo->nz,
   ltc->fxD,ltc->fyD,ltc->fzD,cel->xD_w,cel->yD_w,cel->zD_w,
   cel->fxD_w,cel->fyD_w,cel->fzD_w,
   cel->n_w,cel->vertex_w,cel->element_w,cel->eleD_w);
*/
  #endif //N_W

// Periodic ---
  ibm_st1c <<< dimG1,dimB1 >>>
  (cdo->nx,cdo->ny,cdo->nz,cdo->n,ltc->fxD,ltc->fyD,ltc->fzD,ltc->bcD);

  cudaThreadSynchronize();

  return;
#endif // CapsuleSHEARFLOW || CapsuleCHANNELFLOW
}


__global__ void  ibm_st2a
//==========================================================
//
//  IMMERSED BOUNDARY METHOD ON GPU
//
//
(
    double    dt,
    double    dx,
    double    xmin,
    double    xmax,
    double    ymin,
    double    ymax,
    double    zmin,
    double    zmax,
    int       nx,
    int       ny,
    int       nz,
    double    *ltu,
    double    *ltv,
    double    *ltw,
    double    *x,
    double    *y,
    double    *z,
    double    *u,
    double    *v,
    double    *w,
    int       n,
    int       vertex,
    double    *xd
)
//----------------------------------------------------------
{
    int       i, jid, rk,
              ix, iy, iz,
              x1, y1, z1,
              x2, y2, z2;
    double    xx,   yy,   zz,
              xtr,  ytr,  ztr,
              xcos, ycos, zcos, dlt;
    double    xn, yn, zn,
              xr, yr, zr,
              ur, vr, wr,
              um, vm, wm,
              u1, v1, w1,
              u2, v2, w2,
              u3, v3, w3,
              u4, v4, w4;


    jid = blockDim.x*blockIdx.x + threadIdx.x;
    if(jid >= n*vertex) return;

// Interpolate membrane velocity (delta function) ---
    xn = x[jid]; xr = xn;
    yn = y[jid]; yr = yn;
    zn = z[jid]; zr = zn;

#if RUNGE_KUTTA == 4
    // 4th order Runge-Kutta
    for(rk = 0;rk < RUNGE_KUTTA;rk++){
        ur = 0.0;
        vr = 0.0;
        wr = 0.0;
//      xtr = xr + (double)(nx-1)/2.0;
//      ytr = yr + (double)(ny-1)/2.0;
//      ztr = zr + (double)(nz-1)/2.0;
//      x1 = (int)floor(xtr); x2 = (int)ceil(xtr);
//      y1 = (int)floor(ytr); y2 = (int)ceil(ytr);
//      z1 = (int)floor(ztr); z2 = (int)ceil(ztr);
//      if(x1 == x2){ x1 -= 1; x2 += 1; }
//      if(y1 == y2){ y1 -= 1; y2 += 1; }
//      if(z1 == z2){ z1 -= 1; z2 += 1; }
//      x1 -= 1; x2 += 1;
//      y1 -= 1; y2 += 1;
//      z1 -= 1; z2 += 1;
        xtr = xr + (double)(nx-1)*dx/2.0;
        ytr = yr + (double)(ny-1)*dx/2.0;
        ztr = zr + (double)(nz-1)*dx/2.0;
        x1 = (int)floor(xtr/dx) - 1; x2 = (int)ceil(xtr/dx) + 1;
        y1 = (int)floor(ytr/dx) - 1; y2 = (int)ceil(ytr/dx) + 1;
        z1 = (int)floor(ztr/dx) - 1; z2 = (int)ceil(ztr/dx) + 1;

        for(iz = z1;iz <= z2;iz++){
            for(iy = y1;iy <= y2;iy++){
                for(ix = x1;ix <= x2;ix++){
                    if(ix < 0 || ix >= nx
                    || iy < 0 || iy >= ny
                    || iz < 0 || iz >= nz) continue;

                    i  = ix + nx*iy + nx*ny*iz;
//                  xx = fabs((double)ix - xtr);
//                  yy = fabs((double)iy - ytr);
//                  zz = fabs((double)iz - ztr);
                    xx = fabs((double)ix*dx - xtr);
                    yy = fabs((double)iy*dx - ytr);
                    zz = fabs((double)iz*dx - ztr);
//                  xcos = 1.0 + cos(M_PI*xx/2.0);
//                  ycos = 1.0 + cos(M_PI*yy/2.0);
//                  zcos = 1.0 + cos(M_PI*zz/2.0);
                    xcos = 1.0 + cos(M_PI*xx/(2.0*dx));
                    ycos = 1.0 + cos(M_PI*yy/(2.0*dx));
                    zcos = 1.0 + cos(M_PI*zz/(2.0*dx));
                    dlt  = xcos*ycos*zcos/64.0;
                    ur += dlt*ltu[i];
                    vr += dlt*ltv[i];
                    wr += dlt*ltw[i];
                }
            }
        }

        if(rk == 0){
//          u1 = ur; xr = xn + u1*(double)M/2.0;
//          v1 = vr; yr = yn + v1*(double)M/2.0;
//          w1 = wr; zr = zn + w1*(double)M/2.0;
            u1 = ur; xr = xn + u1*(double)M*dt/2.0;
            v1 = vr; yr = yn + v1*(double)M*dt/2.0;
            w1 = wr; zr = zn + w1*(double)M*dt/2.0;
        }
        else if(rk == 1){
//          u2 = ur; xr = xn + u2*(double)M/2.0;
//          v2 = vr; yr = yn + v2*(double)M/2.0;
//          w2 = wr; zr = zn + w2*(double)M/2.0;
            u2 = ur; xr = xn + u2*(double)M*dt/2.0;
            v2 = vr; yr = yn + v2*(double)M*dt/2.0;
            w2 = wr; zr = zn + w2*(double)M*dt/2.0;
        }
        else if(rk == 2){
//          u3 = ur; xr = xn + u3*(double)M;
//          v3 = vr; yr = yn + v3*(double)M;
//          w3 = wr; zr = zn + w3*(double)M;
            u3 = ur; xr = xn + u3*(double)M*dt;
            v3 = vr; yr = yn + v3*(double)M*dt;
            w3 = wr; zr = zn + w3*(double)M*dt;
        }
        else if(rk == 3){
            u4 = ur;
            v4 = vr;
            w4 = wr;
        }
        else break;
    }

    um = (u1 + 2.0*u2 + 2.0*u3 + u4)/6.0;
    vm = (v1 + 2.0*v2 + 2.0*v3 + v4)/6.0;
    wm = (w1 + 2.0*w2 + 2.0*w3 + w4)/6.0;
#elif RUNGE_KUTTA == 2
    // 2nd order Runge-Kutta
    for(rk = 0;rk < RUNGE_KUTTA;rk++){
        ur = 0.0;
        vr = 0.0;
        wr = 0.0;
        xtr = xr + (double)(nx-1)/2.0;
        ytr = yr + (double)(ny-1)/2.0;
        ztr = zr + (double)(nz-1)/2.0;
        x1 = (int)floor(xtr); x2 = (int)ceil(xtr);
        y1 = (int)floor(ytr); y2 = (int)ceil(ytr);
        z1 = (int)floor(ztr); z2 = (int)ceil(ztr);
        if(x1 == x2){ x1 -= 1; x2 += 1; }
        if(y1 == y2){ y1 -= 1; y2 += 1; }
        if(z1 == z2){ z1 -= 1; z2 += 1; }
        x1 -= 1; x2 += 1;
        y1 -= 1; y2 += 1;
        z1 -= 1; z2 += 1;

        for(iz = z1;iz <= z2;iz++){
            for(iy = y1;iy <= y2;iy++){
                for(ix = x1;ix <= x2;ix++){
                    if(ix < 0 || ix >= nx
                    || iy < 0 || iy >= ny
                    || iz < 0 || iz >= nz) continue;

                    i  = ix + nx*iy + nx*ny*iz;
                    xx = fabs((double)ix - xtr);
                    yy = fabs((double)iy - ytr);
                    zz = fabs((double)iz - ztr);
                    xcos = 1.0 + cos(M_PI*xx/2.0);
                    ycos = 1.0 + cos(M_PI*yy/2.0);
                    zcos = 1.0 + cos(M_PI*zz/2.0);
                    dlt  = xcos*ycos*zcos/64.0;
                    ur += dlt*ltu[i];
                    vr += dlt*ltv[i];
                    wr += dlt*ltw[i];
                }
            }
        }

        if(rk == 0){
            u1 = ur; xr = xn + u1*(double)M;
            v1 = vr; yr = yn + v1*(double)M;
            w1 = wr; zr = zn + w1*(double)M;
        }
        else if(rk == 1){
            u2 = ur;
            v2 = vr;
            w2 = wr;
        }
        else break;
    }

    um = (u1 + u2)/2.0;
    vm = (v1 + v2)/2.0;
    wm = (w1 + w2)/2.0;
#elif RUNGE_KUTTA == 1
    // 1st order Euler
    for(rk = 0;rk < RUNGE_KUTTA;rk++){
        ur = 0.0;
        vr = 0.0;
        wr = 0.0;
        xtr = xr + (double)(nx-1)/2.0;
        ytr = yr + (double)(ny-1)/2.0;
        ztr = zr + (double)(nz-1)/2.0;
        x1 = (int)floor(xtr); x2 = (int)ceil(xtr);
        y1 = (int)floor(ytr); y2 = (int)ceil(ytr);
        z1 = (int)floor(ztr); z2 = (int)ceil(ztr);
        if(x1 == x2){ x1 -= 1; x2 += 1; }
        if(y1 == y2){ y1 -= 1; y2 += 1; }
        if(z1 == z2){ z1 -= 1; z2 += 1; }
        x1 -= 1; x2 += 1;
        y1 -= 1; y2 += 1;
        z1 -= 1; z2 += 1;

        for(iz = z1;iz <= z2;iz++){
            for(iy = y1;iy <= y2;iy++){
                for(ix = x1;ix <= x2;ix++){
                    if(ix < 0 || ix >= nx
                    || iy < 0 || iy >= ny
                    || iz < 0 || iz >= nz) continue;

                    i  = ix + nx*iy + nx*ny*iz;
                    xx = fabs((double)ix - xtr);
                    yy = fabs((double)iy - ytr);
                    zz = fabs((double)iz - ztr);
                    xcos = 1.0 + cos(M_PI*xx/2.0);
                    ycos = 1.0 + cos(M_PI*yy/2.0);
                    zcos = 1.0 + cos(M_PI*zz/2.0);
                    dlt  = xcos*ycos*zcos/64.0;
                    ur += dlt*ltu[i];
                    vr += dlt*ltv[i];
                    wr += dlt*ltw[i];
                }
            }
        }
    }
    um = ur;
    vm = vr;
    wm = wr;
#endif

// Update ---
//  xr = xn + um*(double)M;
//  yr = yn + vm*(double)M;
//  zr = zn + wm*(double)M;
    xr = xn + um*(double)M*dt;
    yr = yn + vm*(double)M*dt;
    zr = zn + wm*(double)M*dt;

    if(xr < xmin) xr += LX;
    if(xr > xmax) xr -= LX;
    if(zr < zmin) zr += LZ;
    if(zr > zmax) zr -= LZ;

    u[jid] = um; x[jid] = xr; xd[jid*3 + 0] = xr;
    v[jid] = vm; y[jid] = yr; xd[jid*3 + 1] = yr;
    w[jid] = wm; z[jid] = zr; xd[jid*3 + 2] = zr;

    return;
}


void  ibm_st2
//==========================================================
//
//  IMMERSED BOUNDARY METHOD
//
//
(
    domain    *cdo,
    lattice   *ltc,
    cell      *cel,
    fem       *fem
)
//----------------------------------------------------------
{
#if defined(KARMAN) || defined(SHEARFLOW) || defined(CHANNELFLOW)
  return;
#elif defined(CapsuleSHEARFLOW) || defined(CapsuleCHANNELFLOW)
  #if(CN == 0)
  return;
  #else

  #if(N_R > 0)
  int       thread = 128, block = (cel->n*cel->vertex)/thread+1;
  dim3      dimB(thread,1,1), dimG(block,1,1);

// Interpolate membrane velocity & Update ---
  ibm_st2a <<< dimG,dimB >>>
  (cdo->dt,cdo->dx,cdo->xmin,cdo->xmax,cdo->ymin,cdo->ymax,cdo->zmin,cdo->zmax,
   cdo->nx,cdo->ny,cdo->nz,ltc->umD,ltc->vmD,ltc->wmD,
   cel->xD,cel->yD,cel->zD,cel->uD,cel->vD,cel->wD,
   cel->n,cel->vertex,fem->xd);

  cudaThreadSynchronize();
  #endif // N_R

// Interpolate membrane velocity & Update for WBC ---
  #if(N_W > 0)
  int       thread2 = 128, block2 = (cel->n_w*cel->vertex_w)/thread2+1;
  dim3      dimB2(thread2,1,1), dimG2(block2,1,1);

  ibm_st2a <<< dimG2,dimB2 >>>
  (cdo->dt,cdo->dx,cdo->xmin,cdo->xmax,cdo->ymin,cdo->ymax,cdo->zmin,cdo->zmax,
   cdo->nx,cdo->ny,cdo->nz,ltc->umD,ltc->vmD,ltc->wmD,
   cel->xD_w,cel->yD_w,cel->zD_w,cel->uD_w,cel->vD_w,cel->wD_w,
   cel->n_w,cel->vertex_w,fem->xd_w);

  cudaThreadSynchronize();
  #endif // N_W

  return;
  #endif // CN
#endif // CapsuleSHEARFLOW || CapsuleCHANNELFLOW
}

