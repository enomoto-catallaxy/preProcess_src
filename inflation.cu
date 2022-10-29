#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "thrust/device_ptr.h"
#include "thrust/reduce.h"
#include "def.h"

#define INFLATION  (1.0e+00+1.0e-06)
#define THREAD     64
#define DEF        1.0e-03

template <typename T>void swap(T **f1, T **f2)
{
  T  (* tmp);

  tmp = *f1; 
  *f1 = *f2; 
  *f2 = tmp;
}

__global__ void capsulePeriodicBefore_GPU
// ====================================================================
//
(
    double   *xyz,
    double   *x,
    double   *y,
    double   *z,
    int      idCaps,
    int      numNode,
    double   gx,
    double   gy,
    double   gz
)
// ====================================================================
{
    double dz;
    int idNodeLocal = blockDim.x*blockIdx.x + threadIdx.x,
        idNode;
    if(idNodeLocal >= numNode) return;
    idNode = idCaps*numNode + idNodeLocal;

/*
    double dx;

// x-periodic
    dx = x[idNode] - gx; 
    if (dx > LX*0.5) {
      x[idNode] -= LX;
    } else if(dx < -LX*0.5) {
      x[idNode] += LX;
    }
*/
// z-periodic
    dz = z[idNode] - gz; 
    if (dz > LZ*0.5) {
      z[idNode] -= LZ;
      xyz[idNode*3 + 2] -= LZ;
    } else if(dz < -LZ*0.5) {
      z[idNode] += LZ;
      xyz[idNode*3 + 2] += LZ;
    }
}

__global__ void capsulePeriodicAfter_GPU
// ====================================================================
//
(
    double   *xyz,
    double   *x,
    double   *y,
    double   *z,
    double   xmax,
    double   xmin,
    double   zmax,
    double   zmin,
    int      numNode
)
// ====================================================================
{
    int idNode = blockDim.x*blockIdx.x + threadIdx.x;
    if(idNode >= numNode) return;

/*
    if(x[idNode] < xmin) x[idNode] += LX; 
    else if(x[idNode] > xmax) x[idNode] -= LX; 
*/
    if(z[idNode] < zmin){
       z[idNode] += LZ; 
       xyz[idNode*3 + 2] += LZ; 
    }
    else if(z[idNode] > zmax){
       z[idNode] -= LZ; 
       xyz[idNode*3 + 2] -= LZ; 
    }
}

__global__ void calcEleNormal_GPU
// ====================================================================
//
// purpose    :  offset capsule volulme (GPU KERNEL)
//
// date       :  May 8, 2014
// programmer :  Hiroki Ito 
// place      :  Ishikawa Lab, Tohoku University
//
(
    double *xr,
    double *x,
    double *y,
    double *z,
    double *normal,
    double *ds,
    double *dsRef,
    double *tempV,
    double *tempVRef,
    int    *tri,
    int     numElementWorld,
    int     flag
)
// --------------------------------------------------------------------
{
    int idElement = blockDim.x*blockIdx.x + threadIdx.x;
    if(idElement >= numElementWorld) return;

    int    i, na, nb, nc;
    double vec1[3], vec2[3], v1d[3], v2d[3], crossVec12[3], normCrossVec12;
    double r[3];

//----- Inflating capsule volume -----
    na = tri[3*idElement + 0]; 
    nb = tri[3*idElement + 1]; 
    nc = tri[3*idElement + 2]; 

    if(flag == 0){
       vec1[0] = (x[nb] - x[na])*RATIO;
       vec1[1] = (y[nb] - y[na])*RATIO;
       vec1[2] = (z[nb] - z[na])*RATIO;

       vec2[0] = (x[nc] - x[na])*RATIO;
       vec2[1] = (y[nc] - y[na])*RATIO;
       vec2[2] = (z[nc] - z[na])*RATIO;
    }
    else{
       vec1[0] = x[nb] - x[na];
       vec1[1] = y[nb] - y[na];
       vec1[2] = z[nb] - z[na];

       vec2[0] = x[nc] - x[na];
       vec2[1] = y[nc] - y[na];
       vec2[2] = z[nc] - z[na];
    }

    crossVec12[0] = vec1[1]*vec2[2] - vec1[2]*vec2[1];
    crossVec12[1] = vec1[2]*vec2[0] - vec1[0]*vec2[2];
    crossVec12[2] = vec1[0]*vec2[1] - vec1[1]*vec2[0];

    normCrossVec12 = sqrt(crossVec12[0]*crossVec12[0]
                        + crossVec12[1]*crossVec12[1]
                        + crossVec12[2]*crossVec12[2]);

    normal[3*idElement + 0] = crossVec12[0] / normCrossVec12;
    normal[3*idElement + 1] = crossVec12[1] / normCrossVec12;
    normal[3*idElement + 2] = crossVec12[2] / normCrossVec12;

    ds[idElement] = normCrossVec12/2.0;

    if(flag == 0){
       r[0] = (x[na] + x[nb] + x[nc])*RATIO/3.0;
       r[1] = (y[na] + y[nb] + y[nc])*RATIO/3.0;
       r[2] = (z[na] + z[nb] + z[nc])*RATIO/3.0;
    }
    else{
       r[0] = (x[na] + x[nb] + x[nc])/3.0;
       r[1] = (y[na] + y[nb] + y[nc])/3.0;
       r[2] = (z[na] + z[nb] + z[nc])/3.0;
    }

    tempV[idElement] = 1.0/3.0*(r[0]*normal[3*idElement + 0]
                      + r[1]*normal[3*idElement + 1]
                      + r[2]*normal[3*idElement + 2]) * ds[idElement];

//----- Reference capsule volume -----
    for(i = 0; i < 3; i++){
        v1d[i] = xr[3*nb + i] - xr[3*na + i];
        v2d[i] = xr[3*nc + i] - xr[3*na + i];
    }

    crossVec12[0] = v1d[1]*v2d[2] - v1d[2]*v2d[1];
    crossVec12[1] = v1d[2]*v2d[0] - v1d[0]*v2d[2];
    crossVec12[2] = v1d[0]*v2d[1] - v1d[1]*v2d[0];

    normCrossVec12 = sqrt(crossVec12[0]*crossVec12[0]
                        + crossVec12[1]*crossVec12[1]
                        + crossVec12[2]*crossVec12[2]);

    normal[3*idElement + 0] = crossVec12[0] / normCrossVec12;
    normal[3*idElement + 1] = crossVec12[1] / normCrossVec12;
    normal[3*idElement + 2] = crossVec12[2] / normCrossVec12;

    dsRef[idElement] = normCrossVec12/2.0;

    r[0] = (xr[3*na+0] + xr[3*nb+0] + xr[3*nc+0])/3.0;
    r[1] = (xr[3*na+1] + xr[3*nb+1] + xr[3*nc+1])/3.0;
    r[2] = (xr[3*na+2] + xr[3*nb+2] + xr[3*nc+2])/3.0;

    tempVRef[idElement] = 1.0/3.0*(r[0]*normal[3*idElement + 0]
                        + r[1]*normal[3*idElement + 1]
                        + r[2]*normal[3*idElement + 2]) * dsRef[idElement];
}

__global__ void calcInflation_GPU
// ====================================================================
//
// purpose    :  Inflation of capsule (GPU KERNEL)
//
// date       :  Jun 27, 2014
// programmer :  Naoki Takeishi 
// place      :  Ishikawa Lab, Tohoku University
//
(
    double    *xyz,
    double    *xr,
    double    *x,
    double    *y,
    double    *z,
    int       numVertexWorld
)
// --------------------------------------------------------------------
{
    int    idVertex = blockDim.x*blockIdx.x + threadIdx.x;
    if(idVertex >= numVertexWorld) return;

    x[idVertex] *= INFLATION;
    y[idVertex] *= INFLATION;
    z[idVertex] *= INFLATION;

    xr[3*idVertex + 0] *= INFLATION;
    xr[3*idVertex + 1] *= INFLATION;
    xr[3*idVertex + 2] *= INFLATION;

    xyz[3*idVertex + 0] = x[idVertex];
    xyz[3*idVertex + 1] = y[idVertex];
    xyz[3*idVertex + 2] = z[idVertex];

}


__global__ void update_contravariant_GPU
//==========================================================
//
//  RESET REFERENCE CONTRAVARIANT METRIC TENSOR
//
//
(
    double    *xr,
    double    *n,
    int       *ele,
    int       numElementWorld
)
//----------------------------------------------------------
{
    int idElement = blockDim.x*blockIdx.x + threadIdx.x;
    if(idElement >= numElementWorld) return;

    int       j, ea, eb, ec;
    double    v1[3], v2[3], v[4], lv1, lv2, inv12, dv12;

    ea = ele[6*idElement + 0];
    eb = ele[6*idElement + 1];
    ec = ele[6*idElement + 2];

    for(j = 0; j < 3; j++){
        v1[j] = xr[3*eb + j] - xr[3*ea + j];
        v2[j] = xr[3*ec + j] - xr[3*ea + j];
    }

    lv1   = v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2];
    lv2   = v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2];
    inv12 = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
    dv12  = lv1*lv2 - inv12*inv12;

    // covariant metric tensor
    v[0] = lv1;   v[1] = inv12;
    v[2] = inv12; v[3] = lv2;

    // contravariant metric tensor
    n[4*idElement + 0] = v[3]/dv12;  n[4*idElement + 1] = -v[1]/dv12;
    n[4*idElement + 2] = -v[2]/dv12; n[4*idElement + 3] = v[0]/dv12;
}


void mallocVolume (cell *cel)
//==========================================================
{
  #if (N_R > 0)
//    cel->volume0   = (double *)malloc(sizeof(double)*cel->n  ); if(cel->volume0   == NULL) error(0); 
//    cel->volume    = (double *)malloc(sizeof(double)*cel->n  ); if(cel->volume    == NULL) error(0); 
//    cel->area      = (double *)malloc(sizeof(double)*cel->n  ); if(cel->area      == NULL) error(0); 
//    cel->gx        = (double *)malloc(sizeof(double)*cel->n*3); if(cel->gx        == NULL) error(0);
    cel->volumeRef = (double *)malloc(sizeof(double)*cel->n); if(cel->volumeRef == NULL) error(0); 
    cel->areaRef   = (double *)malloc(sizeof(double)*cel->n); if(cel->areaRef   == NULL) error(0); 

//    checkCudaErrors(cudaMalloc((void **)&(cel->normal ),sizeof(double)*cel->n*cel->element*3));
//    checkCudaErrors(cudaMalloc((void **)&(cel->dS     ),sizeof(double)*cel->n*cel->element  ));
//    checkCudaErrors(cudaMalloc((void **)&(cel->vtmp   ),sizeof(double)*cel->n*cel->element  ));
    checkCudaErrors(cudaMalloc((void **)&(cel->vtmpRef),sizeof(double)*cel->n*cel->element));
    checkCudaErrors(cudaMalloc((void **)&(cel->dSRef  ),sizeof(double)*cel->n*cel->element));

  #endif
  #if (N_W > 0)
//    cel->volume0_w   = (double *)malloc(sizeof(double)*cel->n_w  ); if(cel->volume0_w   == NULL) error(0);
//    cel->volume_w    = (double *)malloc(sizeof(double)*cel->n_w  ); if(cel->volume_w    == NULL) error(0);
//    cel->area_w      = (double *)malloc(sizeof(double)*cel->n_w  ); if(cel->area_w      == NULL) error(0);
//    cel->gx_w        = (double *)malloc(sizeof(double)*cel->n_w*3); if(cel->gx_w        == NULL) error(0);
    cel->volumeRef_w = (double *)malloc(sizeof(double)*cel->n_w); if(cel->volumeRef_w == NULL) error(0);
    cel->areaRef_w   = (double *)malloc(sizeof(double)*cel->n_w); if(cel->areaRef_w   == NULL) error(0);

//    checkCudaErrors(cudaMalloc((void **)&(cel->normal_w ),sizeof(double)*cel->n_w*cel->element_w*3));
//    checkCudaErrors(cudaMalloc((void **)&(cel->dS_w     ),sizeof(double)*cel->n_w*cel->element_w  ));
//    checkCudaErrors(cudaMalloc((void **)&(cel->vtmp_w   ),sizeof(double)*cel->n_w*cel->element_w  ));
    checkCudaErrors(cudaMalloc((void **)&(cel->vtmpRef_w),sizeof(double)*cel->n_w*cel->element_w));
    checkCudaErrors(cudaMalloc((void **)&(cel->dSRef_w  ),sizeof(double)*cel->n_w*cel->element_w));

  #endif
}


double CalcSum_Thrust_Inflation(double *value, int n)
//==========================================================
{
  double sum;
  thrust::device_ptr<double>start(value);
  thrust::device_ptr<double>end(value + n);
  sum = thrust::reduce(start, end); 
  return sum;
}

void capsulePeriodicBefore (domain *cdo, cell *cel, fem *fem)
//==========================================================
{
  int    thread_all;
  dim3   dim_block, dim_grid;

  #if (N_R > 0)
    thread_all  = cel->vertex;
    dim_block.x = THREAD;
    dim_grid.x  = thread_all/THREAD + MIN(thread_all%THREAD, 1);
    for (int idCaps = 0; idCaps < cel->n; idCaps ++) {
      capsulePeriodicBefore_GPU<<< dim_grid, dim_block >>>
        (fem->xd,cel->xD,cel->yD,cel->zD,idCaps,cel->vertex,cel->gx[idCaps*3 + 0],cel->gx[idCaps*3 + 1],cel->gx[idCaps*3 + 2]);
    }
  #endif
  #if (N_W > 0)
    thread_all  = cel->vertex_w;
    dim_block.x = THREAD;
    dim_grid.x  = thread_all/THREAD + MIN(thread_all%THREAD, 1);
    for (int idCaps = 0; idCaps < cel->n_w; idCaps ++) {
      capsulePeriodicBefore_GPU<<< dim_grid, dim_block >>>
        (fem->xd_w,cel->xD_w,cel->yD_w,cel->zD_w,idCaps,cel->vertex_w,cel->gx_w[idCaps*3 + 0],cel->gx_w[idCaps*3 + 1],cel->gx_w[idCaps*3 + 2]);
    }
  #endif
}

void capsulePeriodicAfter (domain *cdo, cell *cel, fem *fem)
//==========================================================
{
  int    thread_all;
  dim3   dim_block, dim_grid;

  #if (N_R > 0)
    thread_all  = cel->vertex*cel->n;
    dim_block.x = THREAD;
    dim_grid.x  = thread_all/THREAD + MIN(thread_all%THREAD, 1);
    capsulePeriodicAfter_GPU<<< dim_grid, dim_block >>>
      (fem->xd,cel->xD,cel->yD,cel->zD,cdo->xmax,cdo->xmin,cdo->zmax,cdo->zmin,thread_all);
  #endif
  #if (N_W > 0)
    thread_all  = cel->vertex_w*cel->n_w;
    dim_block.x = THREAD;
    dim_grid.x  = thread_all/THREAD + MIN(thread_all%THREAD, 1);
    capsulePeriodicAfter_GPU<<< dim_grid, dim_block >>>
      (fem->xd_w,cel->xD_w,cel->yD_w,cel->zD_w,cdo->xmax,cdo->xmin,cdo->zmax,cdo->zmin,thread_all);
  #endif
}

void calcCenter (domain *cdo, cell *cel)
//==========================================================
{
  #if (N_R > 0)
    for (int idCaps = 0; idCaps < cel->n; idCaps++) {
      cel->gx[idCaps*3 + 0] = CalcSum_Thrust_Inflation(cel->xD + idCaps*cel->vertex, cel->vertex)/(double)cel->vertex;
      cel->gx[idCaps*3 + 1] = CalcSum_Thrust_Inflation(cel->yD + idCaps*cel->vertex, cel->vertex)/(double)cel->vertex;
      cel->gx[idCaps*3 + 2] = CalcSum_Thrust_Inflation(cel->zD + idCaps*cel->vertex, cel->vertex)/(double)cel->vertex;
/*  
      if (cel->gx[idCaps*3 + 0] < cdo->xmin) {
        cel->gx[idCaps*3 + 0] += LX;
      } else if(cel->gx[idCaps*3 + 0] > cdo->xmax) {
        cel->gx[idCaps*3 + 0] -= LX; 
      }
*/  
      if(cel->gx[idCaps*3 + 2] < cdo->zmin) {
        cel->gx[idCaps*3 + 2] += LZ; 
      } else if(cel->gx[idCaps*3 + 2] > cdo->zmax) {
        cel->gx[idCaps*3 + 2] -= LZ; 
      }
    }
  #endif
  #if (N_W > 0)
    for (int idCaps = 0; idCaps < cel->n_w; idCaps++) {
      cel->gx_w[idCaps*3 + 0] = CalcSum_Thrust_Inflation(cel->xD_w + idCaps*cel->vertex_w, cel->vertex_w)/(double)cel->vertex_w;
      cel->gx_w[idCaps*3 + 1] = CalcSum_Thrust_Inflation(cel->yD_w + idCaps*cel->vertex_w, cel->vertex_w)/(double)cel->vertex_w;
      cel->gx_w[idCaps*3 + 2] = CalcSum_Thrust_Inflation(cel->zD_w + idCaps*cel->vertex_w, cel->vertex_w)/(double)cel->vertex_w;
/*  
      if (cel->gx_w[idCaps*3 + 0] < cdo->xmin) {
        cel->gx_w[idCaps*3 + 0] += LX;
      } else if(cel->gx_w[idCaps*3 + 0] > cdo->xmax) {
        cel->gx_w[idCaps*3 + 0] -= LX; 
      }
*/  
      if(cel->gx_w[idCaps*3 + 2] < cdo->zmin) {
        cel->gx_w[idCaps*3 + 2] += LZ; 
      } else if(cel->gx_w[idCaps*3 + 2] > cdo->zmax) {
        cel->gx_w[idCaps*3 + 2] -= LZ; 
      }
    }
  #endif
}

void calcVolume(domain *cdo, cell *cel, fem *fem)
//==========================================================
//
//   Caluculate volume and surface area of capsule
//
{
  static int flag = 0;
  int        thread_all; 
  dim3       dim_block, dim_grid;

  #if (N_R > 0)
    thread_all  = cel->element*cel->n;
    dim_block.x = THREAD;
    dim_grid.x  = thread_all/THREAD + MIN(thread_all%THREAD, 1);

    calcEleNormal_GPU<<< dim_grid, dim_block >>>
      (fem->xrd, cel->xD, cel->yD, cel->zD, cel->normal, cel->dS, cel->dSRef, cel->vtmp, cel->vtmpRef, cel->eleD, thread_all, flag);

    cudaThreadSynchronize();

    for(int idCaps = 0; idCaps < cel->n; idCaps++){
      cel->volume[idCaps] = CalcSum_Thrust_Inflation(cel->vtmp + idCaps*cel->element, cel->element);
      cel->area[idCaps]   = CalcSum_Thrust_Inflation(cel->dS + idCaps*cel->element, cel->element);

      cel->volumeRef[idCaps] = CalcSum_Thrust_Inflation(cel->vtmpRef + idCaps*cel->element, cel->element);
      cel->areaRef[idCaps]   = CalcSum_Thrust_Inflation(cel->dSRef + idCaps*cel->element, cel->element);
    }
  #endif
  #if (N_W > 0)
    thread_all  = cel->element_w*cel->n_w;
    dim_block.x = THREAD;
    dim_grid.x  = thread_all/THREAD + MIN(thread_all%THREAD, 1);

    calcEleNormal_GPU<<< dim_grid, dim_block >>>
      (fem->xrd_w, cel->xD_w, cel->yD_w, cel->zD_w, cel->normal_w, cel->dS_w, cel->dSRef_w, cel->vtmp_w, cel->vtmpRef_w, cel->eleD_w, thread_all, flag);

    cudaThreadSynchronize();

    for(int idCaps = 0; idCaps < cel->n_w; idCaps++){
      cel->volume_w[idCaps] = CalcSum_Thrust_Inflation(cel->vtmp_w + idCaps*cel->element_w, cel->element_w);
      cel->area_w[idCaps]   = CalcSum_Thrust_Inflation(cel->dS_w + idCaps*cel->element_w, cel->element_w);

      cel->volumeRef_w[idCaps] = CalcSum_Thrust_Inflation(cel->vtmpRef_w + idCaps*cel->element_w, cel->element_w);
      cel->areaRef_w[idCaps]   = CalcSum_Thrust_Inflation(cel->dSRef_w + idCaps*cel->element_w, cel->element_w);
    }
  #endif
    flag = 1;
}

void inflation_init
//==========================================================
//
//   INITIAL SETTING OF INFLATION PROCESS
//
//
(
    domain   *cdo,
    cell     *cel,
    fem      *fem
)  
//----------------------------------------------------------
{
    mallocVolume(cel);

    if(cdo->restart == 1){
       cel->inflation_flag = NUM_FLAG;
       return;
    }
    cel->inflation_flag = 0;
    cdo->lambda = 1.0;

    calcVolume(cdo,cel,fem);
    calcCenter(cdo,cel);

  #if (N_R > 0)
    swap<double>(&(cel->volume), &(cel->volume0));

    printf("====================================================================\n");
    for(int idCaps = 0; idCaps < cel->n; idCaps++){
        printf("  Volume of RBC[%d] :%10.3e\n",idCaps,cel->volume0[idCaps]);
    }
    printf("====================================================================\n");
  #endif
  #if (N_W > 0)
    swap<double>(&(cel->volume_w), &(cel->volume0_w));

    printf("====================================================================\n");
    for(int idCaps = 0; idCaps < cel->n_w; idCaps++){
        printf("  Volume of WBC[%d] :%10.3e\n",idCaps,cel->volume0_w[idCaps]);
    }
    printf("====================================================================\n");
  #endif

}

void  inflation
//==========================================================
//
//   INFLATION PROCESS IN CAPSULE
//
//
(
    process  *prc, 
    domain   *cdo,
    lattice  *ltc,
    cell     *cel,
    fem      *fem,
    Wall     *wall,
    Wall     *wall_host
)
//----------------------------------------------------------
{
    int        caps1 = 0, caps2 = 0, thread_all; 
    dim3       dim_block, dim_grid;
    double     rate, ave_rate;

    capsulePeriodicBefore(cdo, cel, fem);
    calcVolume(cdo, cel, fem);

  #if (N_R > 0)
    ave_rate = 0.0e+00;
    thread_all  = cel->n*cel->vertex;
    dim_block.x = THREAD;
    dim_grid.x  = thread_all/THREAD + MIN(thread_all%THREAD, 1);

    // Inflation process ---
    for(int idCaps = 0; idCaps < cel->n; idCaps++){
      rate = fabs((cel->volume[idCaps] - cel->volume0[idCaps])/cel->volume0[idCaps]);
      ave_rate += rate;

      if(rate <= DEF){
        caps1 ++;
      }else if(cel->volume0[idCaps] > cel->volume[idCaps]){
        calcInflation_GPU<<< dim_grid, dim_block >>>
        (fem->xd, fem->xrd, cel->xD, cel->yD, cel->zD, thread_all);
      }else if(cel->volume0[idCaps] < cel->volume[idCaps]){
        printf("CAPSULE : %d rate : %10.4e\n",idCaps+1,rate);
        error(9);
      }
    }
    if((prc->iter-1)%STANDARDOUTPUT == 0) printf("Volume change ratio ..... %10.4e\n",ave_rate/(double)cel->n);

    // Update of reference contravariant metric tensor ---
    thread_all = cel->element*cel->n;
    dim_grid.x = thread_all/THREAD + MIN(thread_all%THREAD, 1);

    update_contravariant_GPU<<< dim_grid, dim_block >>>
    (fem->xrd, fem->nd, fem->eled, thread_all);
  #endif
  #if (N_W > 0)
    ave_rate = 0.0e+00;
    thread_all  = cel->n_w*cel->vertex_w;
    dim_block.x = THREAD;
    dim_grid.x  = thread_all/THREAD + MIN(thread_all%THREAD, 1);

    // Inflation process ---
    for(int idCaps = 0; idCaps < cel->n_w; idCaps++){
      rate = fabs((cel->volume_w[idCaps] - cel->volume0_w[idCaps])/cel->volume0_w[idCaps]);
      ave_rate += rate;

      if(rate <= DEF){
        caps2 ++;
      }else if(cel->volume0_w[idCaps] > cel->volume_w[idCaps]){
        calcInflation_GPU<<< dim_grid, dim_block >>>
        (fem->xd_w, fem->xrd_w, cel->xD_w, cel->yD_w, cel->zD_w, thread_all);
      } else if(cel->volume0_w[idCaps] < cel->volume_w[idCaps]){
        printf("CAPSULE : %d rate : %10.4e\n",idCaps+1,rate);
        error(9);
      }
    }
    if((prc->iter-1)%STANDARDOUTPUT == 0) printf("Volume change ratio ..... %10.4e\n",ave_rate/(double)cel->n_w);

    // Update of reference contravariant metric tensor ---
    thread_all = cel->element_w*cel->n_w;
    dim_grid.x = thread_all/THREAD + MIN(thread_all%THREAD, 1);

    update_contravariant_GPU<<< dim_grid, dim_block >>>
    (fem->xrd_w, fem->nd_w, fem->eled_w, thread_all);
  #endif

    calcCenter(cdo, cel);
    capsulePeriodicAfter(cdo, cel, fem);

    if(caps1 == cel->n && caps2 == cel->n_w){
      printf("----- Inflation process has done.\n");
      vof_init(cdo,ltc,cel);
      cel->inflation_flag ++;
    }
  return;
}
