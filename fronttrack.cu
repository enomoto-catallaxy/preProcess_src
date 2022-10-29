#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "def.h"


__device__ double  atomicDoubleAdd3(double* address, double val)
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


void  front_tracking_matrix
//==========================================================
//
//  FRONT TRACKING METHOD ON GPU
//
//
(
    int       **CRS_ptrH,
    int       **CRS_indexH,
    double    **CRS_valueH,
    int       **ELL_ptrH,
    int       **ELL_indexH,
    double    **ELL_valueH,
    double    **diagonalH,
    int       **ELL_ptrD,
    int       **ELL_indexD,
    double    **ELL_valueD,
    double    **diagonalD,
    double    **xD,
    double    **dgD,
    double    **gxD,
    double    **gyD,
    double    **gzD,
    int       *mn,
    int       *nnz,
    double    dx,
    int       cdnx,
    int       cdny,
    int       cdnz,
    int       cdn,
    int       *ltbc
)
//----------------------------------------------------------
{
    int       tmn, tnnz = 0,
              i, j, tindex1, tindex2, cnt = 0;


// Matrix and Vector size ---
    tmn = cdn;
    for(i = 0;i < cdn;i++){
        if     (ltbc[i] == COMP) tnnz += 7;
        else if(ltbc[i] == BUFF) tnnz += 2;
        else                     tnnz += 1;
    }

    (*mn)  = expand2pow2(tmn);
    (*nnz) = tnnz + ((*mn)- tmn);

// Allocate with CRS ---
    (*CRS_ptrH)   = (int    *)malloc(sizeof(int   )*((*mn)+1)); if((*CRS_ptrH)   == NULL) error(0);
    (*CRS_indexH) = (int    *)malloc(sizeof(int   )*(*nnz)   ); if((*CRS_indexH) == NULL) error(0);
    (*CRS_valueH) = (double *)malloc(sizeof(double)*(*nnz)   ); if((*CRS_valueH) == NULL) error(0);

#if defined(CapsuleSHEARFLOW)
    int ix, iy, iz; 

// Matrix component with CRS ---
    (*CRS_ptrH)[0] = 0;
    for(i = 0;i < (*mn);i++){
        iz = (int)(i/(cdnx*cdny));
        iy = (int)(i/cdnx) - cdny*iz;
        ix = i - cdnx*iy - cdnx*cdny*iz;

        if(i >= cdn){
            (*CRS_ptrH)[i+1]  = (*CRS_ptrH)[i] + 1;
            (*CRS_indexH)[cnt] = i;
            (*CRS_valueH)[cnt] = 1.0;
            cnt += 1;
        }
        else if(ltbc[i] == COMP){
            (*CRS_ptrH)[i+1]     = (*CRS_ptrH)[i] + 7;
            (*CRS_indexH)[cnt]   = i - cdnx*cdny;
            (*CRS_indexH)[cnt+1] = i - cdnx;
            (*CRS_indexH)[cnt+2] = i - 1;
            (*CRS_indexH)[cnt+3] = i;
            (*CRS_indexH)[cnt+4] = i + 1;
            (*CRS_indexH)[cnt+5] = i + cdnx;
            (*CRS_indexH)[cnt+6] = i + cdnx*cdny;
            (*CRS_valueH)[cnt]   =  1.0/(dx*dx);
            (*CRS_valueH)[cnt+1] =  1.0/(dx*dx);
            (*CRS_valueH)[cnt+2] =  1.0/(dx*dx);
            (*CRS_valueH)[cnt+3] = -6.0/(dx*dx);
            (*CRS_valueH)[cnt+4] =  1.0/(dx*dx);
            (*CRS_valueH)[cnt+5] =  1.0/(dx*dx);
            (*CRS_valueH)[cnt+6] =  1.0/(dx*dx);
            cnt += 7;
        }
        else if(ltbc[i] == BUFF){
            (*CRS_ptrH)[i+1]  = (*CRS_ptrH)[i] + 2;
            tindex1 = i;
            if     (ix <= 1      && iz <= 1)      tindex2 = i + (cdnx-4) + cdnx*cdny*(cdnz-4);
            else if(ix <= 1      && iz >= cdnz-2) tindex2 = i + (cdnx-4) - cdnx*cdny*(cdnz-4);
            else if(ix >= cdnx-2 && iz <= 1)      tindex2 = i - (cdnx-4) + cdnx*cdny*(cdnz-4);
            else if(ix >= cdnx-2 && iz >= cdnz-2) tindex2 = i - (cdnx-4) - cdnx*cdny*(cdnz-4);
            else if(ix <= 1)                      tindex2 = i + (cdnx-4);
            else if(ix >= cdnx-2)                 tindex2 = i - (cdnx-4);
            else if(iz <= 1)                      tindex2 = i + cdnx*cdny*(cdnz-4);
            else if(iz >= cdnz-2)                 tindex2 = i - cdnx*cdny*(cdnz-4);

#elif defined(CapsuleCHANNELFLOW)
    int iz; 

// Matrix component with CRS ---
    (*CRS_ptrH)[0] = 0;
    for(i = 0;i < (*mn);i++){
        iz = (int)(i/(cdnx*cdny));

        if(i >= cdn){
            (*CRS_ptrH)[i+1]  = (*CRS_ptrH)[i] + 1;
            (*CRS_indexH)[cnt] = i;
            (*CRS_valueH)[cnt] = 1.0;
            cnt += 1;
        }
        else if(ltbc[i] == COMP){
            (*CRS_ptrH)[i+1]     = (*CRS_ptrH)[i] + 7;
            (*CRS_indexH)[cnt]   = i - cdnx*cdny;
            (*CRS_indexH)[cnt+1] = i - cdnx;
            (*CRS_indexH)[cnt+2] = i - 1;
            (*CRS_indexH)[cnt+3] = i;
            (*CRS_indexH)[cnt+4] = i + 1;
            (*CRS_indexH)[cnt+5] = i + cdnx;
            (*CRS_indexH)[cnt+6] = i + cdnx*cdny;
            (*CRS_valueH)[cnt]   =  1.0/(dx*dx);
            (*CRS_valueH)[cnt+1] =  1.0/(dx*dx);
            (*CRS_valueH)[cnt+2] =  1.0/(dx*dx);
            (*CRS_valueH)[cnt+3] = -6.0/(dx*dx);
            (*CRS_valueH)[cnt+4] =  1.0/(dx*dx);
            (*CRS_valueH)[cnt+5] =  1.0/(dx*dx);
            (*CRS_valueH)[cnt+6] =  1.0/(dx*dx);
            cnt += 7;
        }
        else if(ltbc[i] == BUFF){
            (*CRS_ptrH)[i+1]  = (*CRS_ptrH)[i] + 2;
            tindex1 = i;
            if     (iz <= 1)      tindex2 = i + cdnx*cdny*(cdnz-4);
            else if(iz >= cdnz-2) tindex2 = i - cdnx*cdny*(cdnz-4);
#endif
            if(tindex1 < tindex2){
                (*CRS_indexH)[cnt]   = tindex1;
                (*CRS_indexH)[cnt+1] = tindex2;
                (*CRS_valueH)[cnt]   =  1.0;
                (*CRS_valueH)[cnt+1] = -1.0;
            }
            else{
                (*CRS_indexH)[cnt]   = tindex2;
                (*CRS_indexH)[cnt+1] = tindex1;
                (*CRS_valueH)[cnt]   = -1.0;
                (*CRS_valueH)[cnt+1] =  1.0;
            }
            cnt += 2;
        }
        else if(ltbc[i] == WALL){
            (*CRS_ptrH)[i+1]   = (*CRS_ptrH)[i] + 1;
            (*CRS_indexH)[cnt] = i;
            (*CRS_valueH)[cnt] = 1.0;
            cnt += 1;
        }
    }

// CRS to ELL ---
    (*nnz) = 0;
    (*ELL_ptrH) = (int *)malloc(sizeof(int)*(*mn));
    for(j = 0;j < (*mn);j++){
        (*ELL_ptrH)[j] = (*CRS_ptrH)[j+1] - (*CRS_ptrH)[j];
        if((*nnz) < (*ELL_ptrH)[j]) (*nnz) = (*ELL_ptrH)[j];
    }

    (*ELL_indexH) = (int    *)malloc(sizeof(int   )*(*mn)*(*nnz));
    (*ELL_valueH) = (double *)malloc(sizeof(double)*(*mn)*(*nnz));
    for(j = 0;j < (*mn)*(*nnz);j++){
        (*ELL_indexH)[j] = 0;
        (*ELL_valueH)[j] = 0.0;
    }

    for(j = 0;j < (*mn);j++){
        for(i = 0;i < (*ELL_ptrH)[j];i++){
            (*ELL_indexH)[i*(*mn) + j] = (*CRS_indexH)[(*CRS_ptrH)[j] + i]; 
            (*ELL_valueH)[i*(*mn) + j] = (*CRS_valueH)[(*CRS_ptrH)[j] + i];
        }
    }

// Diagonal component ---
    (*diagonalH) = (double *)malloc(sizeof(double)*(*mn));

    for(i = 0;i < (*mn);i++){
        for(j = 0;j < (*ELL_ptrH)[i];j++){
            if((*ELL_indexH)[j*(*mn) + i] == i)
            (*diagonalH)[i] = (*ELL_valueH)[j*(*mn) + i]; 
        }
    }

// Allocate Matrix value for GPU ---
    checkCudaErrors(cudaMalloc((void**)&(*ELL_ptrD  ),sizeof(int   )*(*mn)       ));
    checkCudaErrors(cudaMalloc((void**)&(*ELL_indexD),sizeof(int   )*(*mn)*(*nnz)));
    checkCudaErrors(cudaMalloc((void**)&(*ELL_valueD),sizeof(double)*(*mn)*(*nnz))); 
    checkCudaErrors(cudaMalloc((void**)&(*diagonalD ),sizeof(double)*(*mn)       )); 

    checkCudaErrors(cudaMemcpy((*ELL_ptrD  ),(*ELL_ptrH  ),sizeof(int   )*(*mn)       ,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((*ELL_indexD),(*ELL_indexH),sizeof(int   )*(*mn)*(*nnz),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((*ELL_valueD),(*ELL_valueH),sizeof(double)*(*mn)*(*nnz),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy((*diagonalD ),(*diagonalH ),sizeof(double)*(*mn)       ,cudaMemcpyHostToDevice));

// Allocate Vector value for GPU ---
    (cudaMalloc((void**)&(*xD ),sizeof(double)*(*mn)));
    (cudaMalloc((void**)&(*dgD),sizeof(double)*(*mn)));
    (cudaMalloc((void**)&(*gxD),sizeof(double)*(*mn)));
    (cudaMalloc((void**)&(*gyD),sizeof(double)*(*mn)));
    (cudaMalloc((void**)&(*gzD),sizeof(double)*(*mn)));

    return;
}


void  allocateMatrixVariables
//==========================================================
//
//  FRONT TRACKING METHOD ON GPU
//
//
(
    double    **d_raD,
    double    **d_rD,
    double    **d_pD,
    double    **d_mD,
    double    **d_nD,
    double    **d_oD,
    int       mn,
    int       blocks
)
//----------------------------------------------------------
{
    int       m_size1 = sizeof(double)*mn,
              m_size2 = sizeof(double)*blocks;


//  (cudaMalloc((void**)&(*d_raD),m_size1));
    (cudaMalloc((void**)&(*d_rD), m_size1));
//  (cudaMalloc((void**)&(*d_pD), m_size1));
//  (cudaMalloc((void**)&(*d_mD), m_size1));
//  (cudaMalloc((void**)&(*d_nD), m_size1));
    (cudaMalloc((void**)&(*d_oD), m_size2));

    return;
}


__global__ void  jacobi_matrix
//==========================================================
//
//  JACOBI PRE-CONDITIONING
//
//
(
    int       *ptr,
    double    *value,
    double    *diagonal,
    int       mn
)
//----------------------------------------------------------
{
    int       i, j;


    i = blockDim.x*blockIdx.x + threadIdx.x;

    for(j = 0;j < ptr[i];j++) value[j*mn + i] /= diagonal[i];

    return;
}


__global__ void  jacobi_source
//==========================================================
//
//  JACOBI PRE-CONDITIONING
//
//
(
    double    *b,
    double    *diagonal
)
//----------------------------------------------------------
{
    int       i;


    i = blockDim.x*blockIdx.x + threadIdx.x;

    b[i] /= diagonal[i];

    return;
}


__global__ void  front_tracking_source1
//==========================================================
//
//  FRONT TRACKING METHOD ON GPU
//
//
(
    double    *dg,
    double    *gx,
    double    *gy,
    double    *gz
)
//----------------------------------------------------------
{
    int       i;


    i = blockDim.x*blockIdx.x + threadIdx.x;

    dg[i] = 0.0;
    gx[i] = 0.0;
    gy[i] = 0.0;
    gz[i] = 0.0;

    return;
}


__global__ void  front_tracking_source2
//==========================================================
//
//  FRONT TRACKING METHOD ON GPU
//
//
(
    double    *gx,
    double    *gy,
    double    *gz,
    double    dx,
    int       cdnx,
    int       cdny,
    int       cdnz,
    double    *clx,
    double    *cly,
    double    *clz,
    int       cln,
    int       vertex,
    int       element,
    int       *ele
)
//----------------------------------------------------------
{
    int       i,  eid,
              ix, iy, iz,
              j1, j2, j3,
              x1, y1, z1,
              x2, y2, z2;
    double    xg,   yg,   zg,
              xx,   yy,   zz,
              tx1,  ty1,  tz1,
              tx2,  ty2,  tz2,
              tx3,  ty3,  tz3,
              xtr,  ytr,  ztr,
              xx1,  yy1,  zz1,
              xx2,  yy2,  zz2,
              nvx,  nvy,  nvz,
              xcos, ycos, zcos,
              nn,   ds,   dlt;


    eid = blockDim.x*blockIdx.x + threadIdx.x;
    if(eid >= cln*element) return;

    j1 = ele[eid*TRI  ];
    j2 = ele[eid*TRI+1];
    j3 = ele[eid*TRI+2];

    tx1 = clx[j1]; tx2 = clx[j2]; tx3 = clx[j3];
    ty1 = cly[j1]; ty2 = cly[j2]; ty3 = cly[j3];
    tz1 = clz[j1]; tz2 = clz[j2]; tz3 = clz[j3];

    xx1 = tx2 - tx1;
    yy1 = ty2 - ty1;
    zz1 = tz2 - tz1;
    if     (xx1 >  LX*0.5){ xx1 -= LX; tx1 += LX; }
    else if(xx1 < -LX*0.5){ xx1 += LX; tx2 += LX; }
    if     (zz1 >  LZ*0.5){ zz1 -= LZ; tz1 += LZ; }
    else if(zz1 < -LZ*0.5){ zz1 += LZ; tz2 += LZ; }
    xx2 = tx3 - tx1;
    yy2 = ty3 - ty1;
    zz2 = tz3 - tz1;
    if     (xx2 >  LX*0.5){ xx2 -= LX; tx1 += LX; }
    else if(xx2 < -LX*0.5){ xx2 += LX; tx3 += LX; }
    if     (zz2 >  LZ*0.5){ zz2 -= LZ; tz1 += LZ; }
    else if(zz2 < -LZ*0.5){ zz2 += LZ; tz3 += LZ; }

    xg = (tx1 + tx2 + tx3)/3.0;
    yg = (ty1 + ty2 + ty3)/3.0;
    zg = (tz1 + tz2 + tz3)/3.0;

    xtr = xg + (double)(cdnx-1)*dx/2.0;
    ytr = yg + (double)(cdny-1)*dx/2.0;
    ztr = zg + (double)(cdnz-1)*dx/2.0;
    x1 = (int)floor(xtr/dx) - 1; x2 = (int)ceil(xtr/dx) + 1;
    y1 = (int)floor(ytr/dx) - 1; y2 = (int)ceil(ytr/dx) + 1;
    z1 = (int)floor(ztr/dx) - 1; z2 = (int)ceil(ztr/dx) + 1;

    nvx = -(yy1*zz2 - zz1*yy2);
    nvy = -(zz1*xx2 - xx1*zz2);
    nvz = -(xx1*yy2 - yy1*xx2);
    nn = sqrt(nvx*nvx + nvy*nvy + nvz*nvz);
    ds = nn/2.0;
    nvx /= nn;
    nvy /= nn;
    nvz /= nn;

    for(iz = z1;iz <= z2;iz++){
        for(iy = y1;iy <= y2;iy++){
            for(ix = x1;ix <= x2;ix++){
                if(ix < 0 || ix >= cdnx
                || iy < 0 || iy >= cdny
                || iz < 0 || iz >= cdnz) continue;

                i = ix + cdnx*iy + cdnx*cdny*iz;
                xx = (double)ix*dx - xtr;
                yy = (double)iy*dx - ytr;
                zz = (double)iz*dx - ztr;
                xcos = 1.0 + cos(M_PI*xx/(2.0*dx));
                ycos = 1.0 + cos(M_PI*yy/(2.0*dx));
                zcos = 1.0 + cos(M_PI*zz/(2.0*dx));
                dlt  = xcos*ycos*zcos/(64.0*dx*dx*dx);

                atomicDoubleAdd3(&gx[i],dlt*nvx*ds);
                atomicDoubleAdd3(&gy[i],dlt*nvy*ds);
                atomicDoubleAdd3(&gz[i],dlt*nvz*ds);
            }
        }
    }

    return;
}


__global__ void  front_tracking_source3
//==========================================================
//
//  FRONT TRACKING METHOD ON GPU
//
//
(
    double    *gx,
    double    *gy,
    double    *gz,
    int       cdnx,
    int       cdny,
    int       cdnz,
    int       cdn,
    int       *ltbc
)
//----------------------------------------------------------
{
    int       i;


    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= cdn)        return;
    if(ltbc[i] != COMP) return;

#if defined(CapsuleSHEARFLOW)
    int ix, iy, iz, i1, i2, i3; 

    iz = (int)(i/(cdnx*cdny));
    iy = (int)(i/cdnx) - cdny*iz;
    ix = i - cdnx*iy - cdnx*cdny*iz;

    if(ix <= 3 && iz <= 3){
        i1 = i + (cdnx-4);
        i2 = i + cdnx*cdny*(cdnz-4);
        i3 = i + (cdnx-4) + cdnx*cdny*(cdnz-4);
    }
    else if(ix >= cdnx-4 && iz <= 3){
        i1 = i - (cdnx-4);
        i2 = i + cdnx*cdny*(cdnz-4);
        i3 = i - (cdnx-4) + cdnx*cdny*(cdnz-4);
    }
    else if(ix <= 3 && iz >= cdnz-4){
        i1 = i + (cdnx-4);
        i2 = i - cdnx*cdny*(cdnz-4);
        i3 = i + (cdnx-4) - cdnx*cdny*(cdnz-4);
    }
    else if(ix >= cdnx-4 && iz >= cdnz-4){
        i1 = i - (cdnx-4);
        i2 = i - cdnx*cdny*(cdnz-4);
        i3 = i - (cdnx-4) - cdnx*cdny*(cdnz-4);
    }
    else if(ix <= 3){
        i1 = i + (cdnx-4); i2 = -1; i3 = -1;
    }
    else if(ix >= cdnx-4){
        i1 = i - (cdnx-4); i2 = -1; i3 = -1;
    }
    else if(iz <= 3){
        i1 = i + cdnx*cdny*(cdnz-4); i2 = -1; i3 = -1;
    }
    else if(iz >= cdnz-4){
        i1 = i - cdnx*cdny*(cdnz-4); i2 = -1; i3 = -1;
    }
    else{
        i1 = -1; i2 = -1; i3 = -1;
    }

    if(i1 >= 0){ gx[i] += gx[i1]; gy[i] += gy[i1]; gz[i] += gz[i1]; }
    if(i2 >= 0){ gx[i] += gx[i2]; gy[i] += gy[i2]; gz[i] += gz[i2]; }
    if(i3 >= 0){ gx[i] += gx[i3]; gy[i] += gy[i3]; gz[i] += gz[i3]; }
    if(i1 >= 0){ gx[i1] = gx[i];  gy[i1] = gy[i];  gz[i1] = gz[i];  }
    if(i2 >= 0){ gx[i2] = gx[i];  gy[i2] = gy[i];  gz[i2] = gz[i];  }
    if(i3 >= 0){ gx[i3] = gx[i];  gy[i3] = gy[i];  gz[i3] = gz[i];  }
#elif defined(CapsuleCHANNELFLOW)
    int iz, i1; 

    iz = (int)(i/(cdnx*cdny));

    if(iz <= 3)           i1 = i + cdnx*cdny*(cdnz-4);
    else if(iz >= cdnz-4) i1 = i - cdnx*cdny*(cdnz-4);
    else                  i1 = -1;

    if(i1 >= 0){ gx[i] += gx[i1]; gy[i] += gy[i1]; gz[i] += gz[i1]; }
    if(i1 >= 0){ gx[i1] = gx[i];  gy[i1] = gy[i];  gz[i1] = gz[i];  }
#endif

    return;
}


__global__ void  front_tracking_source4
//==========================================================
//
//  FRONT TRACKING METHOD ON GPU
//
//
(
    double    *dg,
    double    *gx,
    double    *gy,
    double    *gz,
    double    dx,
    int       cdnx,
    int       cdny,
    int       cdnz,
    int       cdn,
    int       *ltbc
)
//----------------------------------------------------------
{
    int       i,
              ixm, ixp,
              iym, iyp,
              izm, izp;


    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= cdn) return;

    ixm = i - 1;
    ixp = i + 1;
    iym = i - cdnx;
    iyp = i + cdnx;
    izm = i - cdnx*cdny;
    izp = i + cdnx*cdny;

    if(ltbc[i] == COMP){
        dg[i] = (gx[ixp] - gx[ixm]
               + gy[iyp] - gy[iym]
               + gz[izp] - gz[izm])/(2.0*dx);
    }
    else{
        dg[i] = 0.0;
    }

    return;
}


__global__ void  front_tracking_solution
//==========================================================
//
//  FRONT TRACKING METHOD ON GPU
//
//
(
    double    *x,
    int       cnt
)
//----------------------------------------------------------
{
    int       i;


    i = blockDim.x*blockIdx.x + threadIdx.x;

    x[i]  = (double)cnt;

    return;
}


void  front_tracking
//==========================================================
//
//  FRONT-TRACKING METHOD
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
  #if(CN == 0)
  return;
  #else

  int            exp_n = expand2pow2(cdo->n);
  int            thread1 = 256, block1 = exp_n/thread1;
  int            maxThreads = 256, threads, blocks;
  dim3           dimB1(thread1,1,1), dimG1(block1,1,1);

//Gradient of indicator function for RBC ---
  #if(N_R > 0)
  int            exp_element = expand2pow2(cel->n*cel->element);
  int            thread2 = 256, block2 = exp_element/thread2;
  dim3           dimB2(thread2,1,1), dimG2(block2,1,1);

  static double  *CRS_valueH, *ELL_valueH, *diagonalH, *ELL_valueD, *diagonalD;
  static double  *xD, *dgD, *gxD, *gyD, *gzD;
  static double  *d_raD, *d_rD, *d_pD, *d_mD, *d_nD, *d_oD;
  static int     *CRS_ptrH, *ELL_ptrH, *CRS_indexH, *ELL_indexH, *ELL_ptrD, *ELL_indexD;
  static int     mn, nnz;
  static int     n1st = 0;
  int            flag, cnt = 0;

// Matrix generation ---
  if(n1st == 0){
      front_tracking_matrix
      (&CRS_ptrH,&CRS_indexH,&CRS_valueH,&ELL_ptrH,&ELL_indexH,&ELL_valueH,&diagonalH,
       &ELL_ptrD,&ELL_indexD,&ELL_valueD,&diagonalD,&xD,&dgD,&gxD,&gyD,&gzD,
       &mn,&nnz,cdo->dx,cdo->nx,cdo->ny,cdo->nz,cdo->n,ltc->bcH);

      getNumBlocksAndThreads
      (mn,maxThreads,blocks,threads);

      allocateMatrixVariables
      (&d_raD,&d_rD,&d_pD,&d_mD,&d_nD,&d_oD,mn,blocks);

      jacobi_matrix <<< dimG1,dimB1 >>>
      (ELL_ptrD,ELL_valueD,diagonalD,mn);

      cudaThreadSynchronize();

      n1st++;
  }

// Reset gradient of indicator function ---
  front_tracking_source1 <<< dimG1,dimB1 >>>
  (dgD,gxD,gyD,gzD);

// Gradient of indicator function for RBC ---
  front_tracking_source2 <<< dimG2,dimB2 >>>
  (gxD,gyD,gzD,cdo->dx,cdo->nx,cdo->ny,cdo->nz,
   cel->xD,cel->yD,cel->zD,cel->n,cel->vertex,cel->element,cel->eleD);

// Periodic & Copy ---
  front_tracking_source3 <<< dimG1,dimB1 >>>
  (gxD,gyD,gzD,cdo->nx,cdo->ny,cdo->nz,cdo->n,ltc->bcD);

// Divergence of gradient field ---
  front_tracking_source4 <<< dimG1,dimB1 >>>
  (dgD,gxD,gyD,gzD,cdo->dx,cdo->nx,cdo->ny,cdo->nz,cdo->n,ltc->bcD);

// Solve matrix of indicator function ---
  jacobi_source <<< dimG1,dimB1 >>>
  (dgD,diagonalD);

  do{
     front_tracking_solution <<< dimG1,dimB1 >>> (xD,cnt);
//    flag = jacobi
//    (ELL_ptrD,ELL_indexD,ELL_valueD,xD,dgD,d_rD,d_mD,d_oD,mn,nnz);
      flag = redblacksor
      (ELL_ptrD,ELL_indexD,ELL_valueD,xD,dgD,d_rD,d_oD,mn,nnz,cdo->nx,cdo->ny,cdo->nz,cdo->n);
//    flag = bicgstab
//    (ELL_ptrD,ELL_indexD,ELL_valueD,xD,dgD,d_raD,d_rD,d_pD,d_mD,d_nD,d_oD,mn,nnz);
      cudaThreadSynchronize();
      cnt++; if(cnt > 10) error(5);
  }while(flag != 0);
  cudaMemcpy(ltc->vfD ,xD ,sizeof(double)*cdo->n,cudaMemcpyDeviceToDevice);

  cudaThreadSynchronize();
  #endif // N_R > 0

// Gradient of indicator function for WBC ---
  #if(N_W > 0)
  int            exp_element_w = expand2pow2(cel->n_w*cel->element_w);
  int            thread3 = 256, block3 = exp_element_w/thread3;
  dim3           dimB3(thread3,1,1), dimG3(block3,1,1);

  static double  *CRS_valueH2, *ELL_valueH2, *diagonalH2, *ELL_valueD2, *diagonalD2;
  static double  *xD2, *dgD2, *gxD2, *gyD2, *gzD2;
  static double  *d_raD2, *d_rD2, *d_pD2, *d_mD2, *d_nD2, *d_oD2;
  static int     *CRS_ptrH2, *ELL_ptrH2, *CRS_indexH2, *ELL_indexH2, *ELL_ptrD2, *ELL_indexD2;
  static int     mn2, nnz2;
  static int     n2st = 0;
  int            flag2, cnt2 = 0;

// Matrix generation ---
  if(n2st == 0){
      front_tracking_matrix
      (&CRS_ptrH2,&CRS_indexH2,&CRS_valueH2,&ELL_ptrH2,&ELL_indexH2,&ELL_valueH2,&diagonalH2,
       &ELL_ptrD2,&ELL_indexD2,&ELL_valueD2,&diagonalD2,&xD2,&dgD2,&gxD2,&gyD2,&gzD2,
       &mn2,&nnz2,cdo->dx,cdo->nx,cdo->ny,cdo->nz,cdo->n,ltc->bcH);

      getNumBlocksAndThreads
      (mn2,maxThreads,blocks,threads);
      allocateMatrixVariables
      (&d_raD2,&d_rD2,&d_pD2,&d_mD2,&d_nD2,&d_oD2,mn2,blocks);

      jacobi_matrix <<< dimG1,dimB1 >>>
      (ELL_ptrD2,ELL_valueD2,diagonalD2,mn2);
      cudaThreadSynchronize();

      n2st++;
  }

// Reset gradient of indicator function ---
  front_tracking_source1 <<< dimG1,dimB1 >>>
  (dgD2,gxD2,gyD2,gzD2);

// Gradient of indicator function for WBC ---
  front_tracking_source2 <<< dimG3,dimB3 >>>
  (gxD2,gyD2,gzD2,cdo->dx,cdo->nx,cdo->ny,cdo->nz,
   cel->xD_w,cel->yD_w,cel->zD_w,cel->n_w,cel->vertex_w,cel->element_w,cel->eleD_w);

// Periodic & Copy ---
  front_tracking_source3 <<< dimG1,dimB1 >>>
  (gxD2,gyD2,gzD2,cdo->nx,cdo->ny,cdo->nz,cdo->n,ltc->bcD);

// Divergence of gradient field ---
  front_tracking_source4 <<< dimG1,dimB1 >>>
  (dgD2,gxD2,gyD2,gzD2,cdo->dx,cdo->nx,cdo->ny,cdo->nz,cdo->n,ltc->bcD);

// Solve matrix of indicator function ---
  jacobi_source <<< dimG1,dimB1 >>>
  (dgD2,diagonalD2);

  do{
      front_tracking_solution <<< dimG1,dimB1 >>> (xD2,cnt2);
//    static double  *d_raD2, *d_mD2;
//    flag2 = jacobi
//    (ELL_ptrD2,ELL_indexD2,ELL_valueD2,xD2,dgD2,d_rD2,d_mD2,d_oD2,mn2,nnz2);
      flag2 = redblacksor
      (ELL_ptrD2,ELL_indexD2,ELL_valueD2,xD2,dgD2,d_rD2,d_oD2,mn2,nnz2,cdo->nx,cdo->ny,cdo->nz,cdo->n);
//    static double  *d_raD2, *d_pD2, *d_mD2, *d_nD2;
//    flag2 = bicgstab
//    (ELL_ptrD2,ELL_indexD2,ELL_valueD2,xD2,dgD2,d_raD2,d_rD2,d_pD2,d_mD2,d_nD2,d_oD2,mn2,nnz2);
      cudaThreadSynchronize();
      cnt2++; if(cnt2 > 10) error(5);
  }while(flag2 != 0);
  cudaMemcpy(ltc->vfD2,xD2,sizeof(double)*cdo->n,cudaMemcpyDeviceToDevice);

  cudaThreadSynchronize();
  #endif // N_W

// Gradient of indicator function for Platelet ---
  #if(N_P > 0)
  int            exp_element_p = expand2pow2(cel->n_p*cel->element_p);
  int            thread4 = 256, block4 = exp_element_p/thread4;
  dim3           dimB4(thread4,1,1), dimG4(block4,1,1);

  static double  *CRS_valueH2, *ELL_valueH2, *diagonalH2, *ELL_valueD2, *diagonalD2;
  static double  *xD2, *dgD2, *gxD2, *gyD2, *gzD2;
  static double  *d_raD2, *d_rD2, *d_pD2, *d_mD2, *d_nD2, *d_oD2;
  static int     *CRS_ptrH2, *ELL_ptrH2, *CRS_indexH2, *ELL_indexH2, *ELL_ptrD2, *ELL_indexD2;
  static int     mn2, nnz2;
  static int     n2st = 0;
  int            flag2, cnt2 = 0;

// Matrix generation ---
  if(n2st == 0){
      front_tracking_matrix
      (&CRS_ptrH2,&CRS_indexH2,&CRS_valueH2,&ELL_ptrH2,&ELL_indexH2,&ELL_valueH2,&diagonalH2,
       &ELL_ptrD2,&ELL_indexD2,&ELL_valueD2,&diagonalD2,&xD2,&dgD2,&gxD2,&gyD2,&gzD2,
       &mn2,&nnz2,cdo->dx,cdo->nx,cdo->ny,cdo->nz,cdo->n,ltc->bcH);

      getNumBlocksAndThreads
      (mn2,maxThreads,blocks,threads);
      allocateMatrixVariables
      (&d_raD2,&d_rD2,&d_pD2,&d_mD2,&d_nD2,&d_oD2,mn2,blocks);

      jacobi_matrix <<< dimG1,dimB1 >>>
      (ELL_ptrD2,ELL_valueD2,diagonalD2,mn2);
      cudaThreadSynchronize();

      n2st++;
  }

// Reset gradient of indicator function ---
  front_tracking_source1 <<< dimG1,dimB1 >>>
  (dgD2,gxD2,gyD2,gzD2);

// Gradient of indicator function ---
  front_tracking_source2 <<< dimG4,dimB4 >>>
  (gxD2,gyD2,gzD2,cdo->dx,cdo->nx,cdo->ny,cdo->nz,
   cel->xD_p,cel->yD_p,cel->zD_p,cel->n_p,cel->vertex_p,cel->element_p,cel->eleD_p);

// Periodic & Copy ---
  front_tracking_source3 <<< dimG1,dimB1 >>>
  (gxD2,gyD2,gzD2,cdo->nx,cdo->ny,cdo->nz,cdo->n,ltc->bcD);

// Divergence of gradient field ---
  front_tracking_source4 <<< dimG1,dimB1 >>>
  (dgD2,gxD2,gyD2,gzD2,cdo->dx,cdo->nx,cdo->ny,cdo->nz,cdo->n,ltc->bcD);

// Solve matrix of indicator function ---
  jacobi_source <<< dimG1,dimB1 >>>
  (dgD2,diagonalD2);

  do{
      front_tracking_solution <<< dimG1,dimB1 >>> (xD2,cnt2);
//    static double  *d_raD2, *d_mD2;
//    flag2 = jacobi
//    (ELL_ptrD2,ELL_indexD2,ELL_valueD2,xD2,dgD2,d_rD2,d_mD2,d_oD2,mn2,nnz2);
      flag2 = redblacksor
      (ELL_ptrD2,ELL_indexD2,ELL_valueD2,xD2,dgD2,d_rD2,d_oD2,mn2,nnz2,cdo->nx,cdo->ny,cdo->nz,cdo->n);
//    static double  *d_raD2, *d_pD2, *d_mD2, *d_nD2;
//    flag2 = bicgstab
//    (ELL_ptrD2,ELL_indexD2,ELL_valueD2,xD2,dgD2,d_raD2,d_rD2,d_pD2,d_mD2,d_nD2,d_oD2,mn2,nnz2);
      cudaThreadSynchronize();
      cnt2++; if(cnt2 > 10) error(5);
  }while(flag2 != 0);
  cudaMemcpy(ltc->vfD2,xD2,sizeof(double)*cdo->n,cudaMemcpyDeviceToDevice);

  cudaThreadSynchronize();
  #endif // N_P

// Gradient of indicator function for Cancer ---
  #if(N_C > 0)
  int            exp_element_c = expand2pow2(cel->n_c*cel->element_c);
  int            thread5 = 256, block5 = exp_element_c/thread5;
  dim3           dimB5(thread5,1,1), dimG5(block5,1,1);

  static double  *CRS_valueH2, *ELL_valueH2, *diagonalH2, *ELL_valueD2, *diagonalD2;
  static double  *xD2, *dgD2, *gxD2, *gyD2, *gzD2;
  static double  *d_raD2, *d_rD2, *d_pD2, *d_mD2, *d_nD2, *d_oD2;
  static int     *CRS_ptrH2, *ELL_ptrH2, *CRS_indexH2, *ELL_indexH2, *ELL_ptrD2, *ELL_indexD2;
  static int     mn2, nnz2;
  static int     n2st = 0;
  int            flag2, cnt2 = 0;

// Matrix generation ---
  if(n2st == 0){
      front_tracking_matrix
      (&CRS_ptrH2,&CRS_indexH2,&CRS_valueH2,&ELL_ptrH2,&ELL_indexH2,&ELL_valueH2,&diagonalH2,
       &ELL_ptrD2,&ELL_indexD2,&ELL_valueD2,&diagonalD2,&xD2,&dgD2,&gxD2,&gyD2,&gzD2,
       &mn2,&nnz2,cdo->dx,cdo->nx,cdo->ny,cdo->nz,cdo->n,ltc->bcH);

      getNumBlocksAndThreads
      (mn2,maxThreads,blocks,threads);
      allocateMatrixVariables
      (&d_raD2,&d_rD2,&d_pD2,&d_mD2,&d_nD2,&d_oD2,mn2,blocks);

      jacobi_matrix <<< dimG1,dimB1 >>>
      (ELL_ptrD2,ELL_valueD2,diagonalD2,mn2);
      cudaThreadSynchronize();

      n2st++;
  }

// Reset gradient of indicator function ---
  front_tracking_source1 <<< dimG1,dimB1 >>>
  (dgD2,gxD2,gyD2,gzD2);

// Gradient of indicator function ---
  front_tracking_source2 <<< dimG5,dimB5 >>>
  (gxD2,gyD2,gzD2,cdo->dx,cdo->nx,cdo->ny,cdo->nz,
   cel->xD_c,cel->yD_c,cel->zD_c,cel->n_c,cel->vertex_c,cel->element_c,cel->eleD_c);

// Periodic & Copy ---
  front_tracking_source3 <<< dimG1,dimB1 >>>
  (gxD2,gyD2,gzD2,cdo->nx,cdo->ny,cdo->nz,cdo->n,ltc->bcD);

// Divergence of gradient field ---
  front_tracking_source4 <<< dimG1,dimB1 >>>
  (dgD2,gxD2,gyD2,gzD2,cdo->dx,cdo->nx,cdo->ny,cdo->nz,cdo->n,ltc->bcD);

// Solve matrix of indicator function ---
  jacobi_source <<< dimG1,dimB1 >>>
  (dgD2,diagonalD2);

  do{
      front_tracking_solution <<< dimG1,dimB1 >>> (xD2,cnt2);
//    static double  *d_raD2, *d_mD2;
//    flag2 = jacobi
//    (ELL_ptrD2,ELL_indexD2,ELL_valueD2,xD2,dgD2,d_rD2,d_mD2,d_oD2,mn2,nnz2);
      flag2 = redblacksor
      (ELL_ptrD2,ELL_indexD2,ELL_valueD2,xD2,dgD2,d_rD2,d_oD2,mn2,nnz2,cdo->nx,cdo->ny,cdo->nz,cdo->n);
//    static double  *d_raD2, *d_pD2, *d_mD2, *d_nD2;
//    flag2 = bicgstab
//    (ELL_ptrD2,ELL_indexD2,ELL_valueD2,xD2,dgD2,d_raD2,d_rD2,d_pD2,d_mD2,d_nD2,d_oD2,mn2,nnz2);
      cudaThreadSynchronize();
      cnt2++; if(cnt2 > 10) error(5);
  }while(flag2 != 0);
  cudaMemcpy(ltc->vfD2,xD2,sizeof(double)*cdo->n,cudaMemcpyDeviceToDevice);

  cudaThreadSynchronize();
  #endif // N_C

  return;
  #endif // CN
#endif // CapsuleSHEARFLOW || CapsuleCHANNELFLOW
}

