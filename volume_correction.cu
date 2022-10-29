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

#define COEFF_VOL_CHANGE 0.1
#define EPS              1.0e-03

#define THREAD 64

template <typename T>void swap(T **f1, T **f2)
{
  T  (* tmp);

  tmp = *f1; 
  *f1 = *f2; 
  *f2 = tmp;
}

__global__ void CapsulePeriodicBefore_GPU
// ====================================================================
//
(
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
    int idNodeLocal = blockDim.x*blockIdx.x + threadIdx.x,
        idNode;
    if(idNodeLocal >= numNode) return;
    idNode = idCaps*numNode + idNodeLocal;

    #if defined(CapsuleSHEARFLOW)
    // x-periodic
    double dx;
    dx = x[idNode] - gx; 
    if (dx > LX*0.5) {
      x[idNode] -= LX;
    } else if(dx < -LX*0.5) {
      x[idNode] += LX;
    }
    #endif

    // z-periodic
    double dz;
    dz = z[idNode] - gz; 
    if (dz > LZ*0.5) {
      z[idNode] -= LZ;
    } else if(dz < -LZ*0.5) {
      z[idNode] += LZ;
    }
}
// --------------------------------------------------------------------

__global__ void CapsulePeriodicAfter_GPU
// ====================================================================
//
(
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

    #if defined(CapsuleSHEARFLOW)
    if(x[idNode] < xmin) x[idNode] += LX; 
    else if(x[idNode] > xmax) x[idNode] -= LX; 
    #endif
    if(z[idNode] < zmin) z[idNode] += LZ; 
    else if(z[idNode] > zmax) z[idNode] -= LZ; 
}
// --------------------------------------------------------------------

__global__ void VolumeCorrection_GPU
// ====================================================================
//
// purpose    :  offset capsule volulme (GPU KERNEL)
//
// date       :  May 8, 2014
// programmer :  Hiroki Ito 
// place      :  Ishikawa Lab, Tohoku University
//
(
    double  *x,
    double  *y,
    double  *z,
    double  *normal,
    double  *lambda,
    int     *NeighElement,
    int     *numNeighElement,
    int     numNode,
    int     numNodeWorld
)
// --------------------------------------------------------------------
{
    int idNode = blockDim.x*blockIdx.x + threadIdx.x;
    if(idNode >= numNodeWorld) return;

    int idCaps = idNode/numNode;
    double normalNode[3] = {0.0, 0.0, 0.0};
    int    idElement, ne;

    for(idElement = 0; idElement < numNeighElement[idNode]; idElement++){
        ne = NeighElement[6*idNode + idElement];
        normalNode[0] += normal[3*ne + 0]/numNeighElement[idNode];
        normalNode[1] += normal[3*ne + 1]/numNeighElement[idNode];
        normalNode[2] += normal[3*ne + 2]/numNeighElement[idNode];
    }
    x[idNode] += -lambda[idCaps]/6.0*normalNode[0]*COEFF_VOL_CHANGE;
    y[idNode] += -lambda[idCaps]/6.0*normalNode[1]*COEFF_VOL_CHANGE;
    z[idNode] += -lambda[idCaps]/6.0*normalNode[2]*COEFF_VOL_CHANGE;
}    
// --------------------------------------------------------------------

__global__ void CalcEleNormal_GPU
// ====================================================================
//
// purpose    :  offset capsule volulme (GPU KERNEL)
//
// date       :  May 8, 2014
// programmer :  Hiroki Ito 
// place      :  Ishikawa Lab, Tohoku University
//
(
    double *x,
    double *y,
    double *z,
    double *normal,
    double *ds,
    double *tempV,
    int    *tri,
    int     numElementWorld
)
// --------------------------------------------------------------------
{
    int idElement = blockDim.x*blockIdx.x + threadIdx.x;
    if(idElement >= numElementWorld);

    int na, nb, nc;
    double vec1[3], vec2[3], crossVec12[3], normCrossVec12;
    double r[3];

    na = tri[3*idElement + 0]; 
    nb = tri[3*idElement + 1]; 
    nc = tri[3*idElement + 2]; 

    vec1[0] = x[nb] - x[na];
    vec1[1] = y[nb] - y[na];
    vec1[2] = z[nb] - z[na];

    vec2[0] = x[nc] - x[na];
    vec2[1] = y[nc] - y[na];
    vec2[2] = z[nc] - z[na];

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

    r[0] = (x[na] + x[nb] + x[nc])/3.0;
    r[1] = (y[na] + y[nb] + y[nc])/3.0;
    r[2] = (z[na] + z[nb] + z[nc])/3.0;

    tempV[idElement] = 1.0/3.0*(r[0]*normal[3*idElement + 0]
                      + r[1]*normal[3*idElement + 1]
                      + r[2]*normal[3*idElement + 2]) * ds[idElement];

}
//----------------------------------------------------------

void MallocVolume (cell *cel)
//==========================================================
{
  #if(N_R > 0)
  cel->volume0 = (double *)malloc(sizeof(double)*cel->n  ); 
  cel->volume  = (double *)malloc(sizeof(double)*cel->n  ); 
  cel->area    = (double *)malloc(sizeof(double)*cel->n  ); 
  cel->gx      = (double *)malloc(sizeof(double)*cel->n*3);
  cel->lambdaH = (double *)malloc(sizeof(double)*cel->n  ); 
  cel->is_converged = (bool *)malloc(sizeof(bool)*cel->n ); 

  checkCudaErrors(cudaMalloc((void **) &(cel->normal),  sizeof(double)*cel->n*cel->element*3));
  checkCudaErrors(cudaMalloc((void **) &(cel->dS),      sizeof(double)*cel->n*cel->element  ));
  checkCudaErrors(cudaMalloc((void **) &(cel->vtmp),    sizeof(double)*cel->n*cel->element  ));
  checkCudaErrors(cudaMalloc((void **) &(cel->lambdaD), sizeof(double)*cel->n               ));
  #endif // N_R
  #if(N_W > 0)
  cel->volume0_w = (double *)malloc(sizeof(double)*cel->n_w  ); 
  cel->volume_w  = (double *)malloc(sizeof(double)*cel->n_w  ); 
  cel->area_w    = (double *)malloc(sizeof(double)*cel->n_w  ); 
  cel->gx_w      = (double *)malloc(sizeof(double)*cel->n_w*3);
  cel->lambdaH_w = (double *)malloc(sizeof(double)*cel->n_w  ); 
  cel->is_converged_w = (bool *)malloc(sizeof(bool)*cel->n_w ); 

  checkCudaErrors(cudaMalloc((void **) &(cel->normal_w),  sizeof(double)*cel->n_w*cel->element_w*3));
  checkCudaErrors(cudaMalloc((void **) &(cel->dS_w),      sizeof(double)*cel->n_w*cel->element_w  ));
  checkCudaErrors(cudaMalloc((void **) &(cel->vtmp_w),    sizeof(double)*cel->n_w*cel->element_w  ));
  checkCudaErrors(cudaMalloc((void **) &(cel->lambdaD_w), sizeof(double)*cel->n_w                 ));
  #endif // N_W
}
//----------------------------------------------------------

double CalcSum_Thrust(double *value, int n)
//==========================================================
{
  double sum;
  thrust::device_ptr<double>start(value);
  thrust::device_ptr<double>end(value + n);
  sum = thrust::reduce(start, end); 
  return sum;
}
//----------------------------------------------------------

void CalcVolume(domain *cdo, cell *cel)  
//==========================================================
//
//   Caluculate volume and surface area of capsule
//
{
  int thread_all; 
  dim3 dim_block, dim_grid;

  #if(N_R > 0)
  thread_all  = cel->n*cel->element;
  dim_block.x = THREAD;
  dim_grid.x  = thread_all/THREAD + MIN(thread_all%THREAD, 1);

  CalcEleNormal_GPU<<< dim_grid, dim_block >>>
    (cel->xD, cel->yD, cel->zD, cel->normal, cel->dS, cel->vtmp, cel->eleD, thread_all);
  for (int idCaps = 0; idCaps < cel->n; idCaps++) {
    cel->volume[idCaps] = CalcSum_Thrust(cel->vtmp + idCaps*cel->element, cel->element);
    cel->area[idCaps] = CalcSum_Thrust(cel->dS + idCaps*cel->element, cel->element);
//    printf(" %d %e %e\n", idCaps, cel->volume[idCaps], cel->area[idCaps]);
  }
  #endif // N_R
  #if(N_W > 0)
  thread_all  = cel->n_w*cel->element_w;
  dim_block.x = THREAD;
  dim_grid.x  = thread_all/THREAD + MIN(thread_all%THREAD, 1);

  CalcEleNormal_GPU<<< dim_grid, dim_block >>>
    (cel->xD_w, cel->yD_w, cel->zD_w, cel->normal_w, cel->dS_w, cel->vtmp_w, cel->eleD_w, thread_all);
  for (int idCaps = 0; idCaps < cel->n_w; idCaps++) {
    cel->volume_w[idCaps] = CalcSum_Thrust(cel->vtmp_w + idCaps*cel->element_w, cel->element_w);
    cel->area_w[idCaps] = CalcSum_Thrust(cel->dS_w + idCaps*cel->element_w, cel->element_w);
//    printf(" %d %e %e\n", idCaps, cel->volume_w[idCaps], cel->area_w[idCaps]);
  }
  #endif // N_W
}
//----------------------------------------------------------

void CapsulePeriodicBefore (domain *cdo, cell *cel)
//==========================================================
{
  int    thread_all;
  dim3   dim_block, dim_grid;

  #if(N_R > 0)
  thread_all  = cel->vertex;
  dim_block.x = THREAD;
  dim_grid.x  = thread_all/THREAD + MIN(thread_all%THREAD, 1);
  for (int idCaps = 0; idCaps < cel->n; idCaps ++) {
    CapsulePeriodicBefore_GPU<<< dim_grid, dim_block >>>
      (cel->xD, cel->yD, cel->zD, idCaps, cel->vertex, cel->gx[idCaps*3 + 0], cel->gx[idCaps*3 + 1], cel->gx[idCaps*3 + 2]);
  }
  #endif // N_R
  #if(N_W > 0)
  thread_all  = cel->vertex_w;
  dim_block.x = THREAD;
  dim_grid.x  = thread_all/THREAD + MIN(thread_all%THREAD, 1);
  for (int idCaps = 0; idCaps < cel->n_w; idCaps ++) {
    CapsulePeriodicBefore_GPU<<< dim_grid, dim_block >>>
      (cel->xD_w, cel->yD_w, cel->zD_w, idCaps, cel->vertex_w, cel->gx_w[idCaps*3 + 0], cel->gx_w[idCaps*3 + 1], cel->gx_w[idCaps*3 + 2]);
  }
  #endif // N_W
}

void CalcCenter (domain *cdo, cell *cel)
//==========================================================
{
  #if(N_R > 0)
  for (int idCaps = 0; idCaps < cel->n; idCaps++) {
    cel->gx[idCaps*3 + 0] = CalcSum_Thrust(cel->xD + idCaps*cel->vertex, cel->vertex)/(double)cel->vertex;
    cel->gx[idCaps*3 + 1] = CalcSum_Thrust(cel->yD + idCaps*cel->vertex, cel->vertex)/(double)cel->vertex;
    cel->gx[idCaps*3 + 2] = CalcSum_Thrust(cel->zD + idCaps*cel->vertex, cel->vertex)/(double)cel->vertex;

    #if defined(CapsuleSHEARFLOW)
    if (cel->gx[idCaps*3 + 0] < cdo->xmin) {
      cel->gx[idCaps*3 + 0] += LX;
    } else if(cel->gx[idCaps*3 + 0] > cdo->xmax) {
      cel->gx[idCaps*3 + 0] -= LX; 
    }
    #endif
    if(cel->gx[idCaps*3 + 2] < cdo->zmin) {
      cel->gx[idCaps*3 + 2] += LZ; 
    } else if(cel->gx[idCaps*3 + 2] > cdo->zmax) {
      cel->gx[idCaps*3 + 2] -= LZ; 
    }
  }
  #endif // N_R
  #if(N_W > 0)
  for (int idCaps = 0; idCaps < cel->n_w; idCaps++) {
    cel->gx_w[idCaps*3 + 0] = CalcSum_Thrust(cel->xD_w + idCaps*cel->vertex_w, cel->vertex_w)/(double)cel->vertex_w;
    cel->gx_w[idCaps*3 + 1] = CalcSum_Thrust(cel->yD_w + idCaps*cel->vertex_w, cel->vertex_w)/(double)cel->vertex_w;
    cel->gx_w[idCaps*3 + 2] = CalcSum_Thrust(cel->zD_w + idCaps*cel->vertex_w, cel->vertex_w)/(double)cel->vertex_w;
    #if defined(CapsuleSHEARFLOW)
    if (cel->gx[idCaps*3 + 0] < cdo->xmin) {
      cel->gx_w[idCaps*3 + 0] += LX;
    } else if(cel->gx_w[idCaps*3 + 0] > cdo->xmax) {
      cel->gx_w[idCaps*3 + 0] -= LX; 
    }
    #endif
    if(cel->gx_w[idCaps*3 + 2] < cdo->zmin) {
      cel->gx_w[idCaps*3 + 2] += LZ; 
    } else if(cel->gx_w[idCaps*3 + 2] > cdo->zmax) {
      cel->gx_w[idCaps*3 + 2] -= LZ; 
    }
  }
  #endif // N_W
}
//----------------------------------------------------------

void CapsulePeriodicAfter (domain *cdo, cell *cel)
//==========================================================
{
  int    thread_all;
  dim3   dim_block, dim_grid;

  #if(N_R > 0)
  thread_all  = cel->vertex*cel->n;
  dim_block.x = THREAD;
  dim_grid.x  = thread_all/THREAD + MIN(thread_all%THREAD, 1);
  CapsulePeriodicAfter_GPU<<< dim_grid, dim_block >>>
    (cel->xD, cel->yD, cel->zD, cdo->xmax, cdo->xmin, cdo->zmax, cdo->zmin, thread_all);
  #endif // N_R
  #if(N_W > 0)
  thread_all  = cel->vertex_w*cel->n_w;
  dim_block.x = THREAD;
  dim_grid.x  = thread_all/THREAD + MIN(thread_all%THREAD, 1);
  CapsulePeriodicAfter_GPU<<< dim_grid, dim_block >>>
    (cel->xD_w, cel->yD_w, cel->zD_w, cdo->xmax, cdo->xmin, cdo->zmax, cdo->zmin, thread_all);
  #endif // N_W
}
//----------------------------------------------------------

bool CalcLambda1 (domain *cdo, cell *cel)
//==========================================================
//
//   Caluculate lambda for volume correction
//
{
  bool   all_converged = true;
  double dV;

  for (int idCaps = 0; idCaps < cel->n; idCaps++) {
    dV = fabs(cel->volume[idCaps] - cel->volume0[idCaps])/cel->volume0[idCaps]*100.0;
    if (dV < EPS) {
      cel->lambdaH[idCaps] = 0.0;
      cel->is_converged[idCaps] = true;
    } else { 
      cel->lambdaH[idCaps] = 18.0*(cel->volume[idCaps] - cel->volume0[idCaps])/cel->area[idCaps];
    }
    all_converged = (all_converged & cel->is_converged[idCaps]);
  }
  checkCudaErrors(cudaMemcpy(cel->lambdaD, cel->lambdaH, cel->n*sizeof(double), cudaMemcpyHostToDevice));
  return all_converged;
}
//----------------------------------------------------------

bool CalcLambda2 (domain *cdo, cell *cel)
//==========================================================
//
//   Caluculate lambda for volume correction
//
{
  bool   all_converged = true;
  double dV;

  for (int idCaps = 0; idCaps < cel->n_w; idCaps++) {
    dV = fabs(cel->volume_w[idCaps] - cel->volume0_w[idCaps])/cel->volume0_w[idCaps]*100.0;
    if (dV < EPS) {
      cel->lambdaH_w[idCaps] = 0.0;
      cel->is_converged_w[idCaps] = true;
    } else { 
      cel->lambdaH_w[idCaps] = 18.0*(cel->volume_w[idCaps] - cel->volume0_w[idCaps])/cel->area_w[idCaps];
    }
    all_converged = (all_converged & cel->is_converged_w[idCaps]);
  }
  checkCudaErrors(cudaMemcpy(cel->lambdaD_w, cel->lambdaH_w, cel->n_w*sizeof(double), cudaMemcpyHostToDevice));

  return all_converged;
}
//----------------------------------------------------------

void volume_init (domain *cdo, cell *cel)
//==========================================================
//
//   Caluculate initial volume of capsule
//
{
#if defined(KARMAN) || defined(SHEARFLOW) || defined(CHANNELFLOW)
  return;
#elif defined(CapsuleSHEARFLOW) || defined(CapsuleCHANNELFLOW)
  static int n1st = 0;
  if(n1st == 0){
     MallocVolume(cel);
     n1st++;
  }
  CalcVolume(cdo, cel);
  #if(N_R > 0)
  swap<double>(&(cel->volume), &(cel->volume0));
  #endif
  #if(N_W > 0)
  swap<double>(&(cel->volume_w), &(cel->volume0_w));
  #endif
  CalcCenter(cdo, cel);
#endif
}
//----------------------------------------------------------

int volume_correction(domain *cdo, cell *cel, fem *fem)
//==========================================================
//
//   Correct volume of capsule
//
{
#if defined(KARMAN) || defined(SHEARFLOW) || defined(CHANNELFLOW)
  return 0;
#elif defined(CapsuleSHEARFLOW) || defined(CapsuleCHANNELFLOW)
  #if(CN == 0)
  return 0;
  #else

  dim3  dim_grid, dim_block;
  int   thread_all, iter;
  bool  all_converged;

// Volume correction for RBC
  #if(N_R > 0)
  thread_all = cel->n*cel->vertex;
  iter = 0;
  dim_block.x = THREAD;
  dim_grid.x = thread_all/THREAD + MIN(thread_all%THREAD, 1);

  for (int idCaps = 0; idCaps < cel->n; idCaps++) {
    cel->is_converged[idCaps] = false;
  }
  CapsulePeriodicBefore(cdo, cel);
  do {
    CalcVolume(cdo, cel);
    all_converged = CalcLambda1(cdo, cel);
    VolumeCorrection_GPU<<< dim_grid, dim_block >>>
      (cel->xD, cel->yD, cel->zD, cel->normal, cel->lambdaD, fem->led, fem->nled, cel->vertex, thread_all);
    iter++;
  } while (all_converged == false && iter < 1000);
  CalcCenter(cdo, cel);
  CapsulePeriodicAfter(cdo, cel);
  if (iter >= 1000) {
    printf(" Volume Correction Failed for RBC!\n");
    exit(1);
  }
  #endif // N_R

// Volume correction for WBC
  #if(N_W > 0)
  thread_all = cel->n_w*cel->vertex_w;
  iter = 0;
  dim_block.x = THREAD;
  dim_grid.x = thread_all/THREAD + MIN(thread_all%THREAD, 1);
  for (int idCaps = 0; idCaps < cel->n_w; idCaps++) {
    cel->is_converged_w[idCaps] = false;
  }
  CapsulePeriodicBefore(cdo, cel);
  do {
    CalcVolume(cdo, cel);
    all_converged = CalcLambda2(cdo, cel);
    VolumeCorrection_GPU<<< dim_grid, dim_block >>>
      (cel->xD_w, cel->yD_w, cel->zD_w, cel->normal_w, cel->lambdaD_w, fem->led_w, fem->nled_w, cel->vertex_w, thread_all);
    iter++;
  } while (all_converged == false && iter < 1000);
  CalcCenter(cdo, cel);
  CapsulePeriodicAfter(cdo, cel);
  if (iter >= 1000) {
    printf(" Volume Correction Failed for WBC!\n");
    exit(1);
  }
  #endif // N_W

  return (iter - 1);
  #endif // CN
#endif  // CapsuleSHEARFLOW || CapsuleCHANNELFLOW
}
//----------------------------------------------------------

