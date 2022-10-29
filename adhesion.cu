#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand_kernel.h>

#include "def.h"
#include "malloc.cuh"
#define THREAD         128
#define MAX_NEIGHWALL  100

void MallocHostAdhesion(cell *cel)
{
  MallocHost<double>(&(cel->f_adhH),         cel->adnum*cel->advertex*3             );
  MallocHost<int>(&(cel->attachment_pointH), cel->adnum*cel->advertex*NUMSPRING_KNOB);
  MallocHost<bool>(&(cel->is_knobH),         cel->adnum*cel->advertex               );
//  MallocHost<double>(&(cel->konH),           cel->adnum*cel->advertex*NUMSPRING_KNOB);
//  MallocHost<double>(&(cel->koffH),          cel->adnum*cel->advertex*NUMSPRING_KNOB);
  MallocHost<double>(&(cel->r_ijH),          cel->adnum*cel->advertex*NUMSPRING_KNOB);
}

void MallocDeviceAdhesion(cell *cel)
{
  MallocDevice<double>(&(cel->r_wall),         cel->adnum*cel->advertex*MAX_NEIGHWALL );
  MallocDevice<double>(&(cel->f_adhD),         cel->adnum*cel->advertex*3             );
  MallocDevice<int>(&(cel->id_wall),           cel->adnum*cel->advertex*MAX_NEIGHWALL );
  MallocDevice<int>(&(cel->attachment_pointD), cel->adnum*cel->advertex*NUMSPRING_KNOB);
  MallocDevice<bool>(&(cel->is_knobD),         cel->adnum*cel->advertex               );
//  MallocDevice<double>(&(cel->konD),           cel->adnum*cel->advertex*NUMSPRING_KNOB);
//  MallocDevice<double>(&(cel->koffD),          cel->adnum*cel->advertex*NUMSPRING_KNOB);
  MallocDevice<double>(&(cel->r_ijD),          cel->adnum*cel->advertex*NUMSPRING_KNOB);
}

void MemcpyAdhesionHtoD(cell *cel)
{
  MemcpyHtoD<int>(cel->attachment_pointD, cel->attachment_pointH, cel->adnum*cel->advertex*NUMSPRING_KNOB);
  MemcpyHtoD<bool>(cel->is_knobD,         cel->is_knobH,          cel->adnum*cel->advertex               );
//  MemcpyHtoD<double>(cel->konD,           cel->konH,              cel->adnum*cel->advertex*NUMSPRING_KNOB);
//  MemcpyHtoD<double>(cel->koffD,          cel->koffH,             cel->adnum*cel->advertex*NUMSPRING_KNOB);
  MemcpyHtoD<double>(cel->r_ijD,          cel->r_ijH,             cel->adnum*cel->advertex*NUMSPRING_KNOB);
}

void MemcpyAdhesionDtoH(cell *cel)
{
  MemcpyDtoH<int>(cel->attachment_pointD, cel->attachment_pointH, cel->adnum*cel->advertex*NUMSPRING_KNOB);
  MemcpyDtoH<bool>(cel->is_knobD,         cel->is_knobH,          cel->adnum*cel->advertex               );
}

__global__ void Initializing_GPU
// ====================================================================
(
        double      * f_adh,
        int         num_nodeALL
)
// ====================================================================
{
  int idNode = blockDim.x*blockIdx.x + threadIdx.x;  // sequential nodeID
  if (idNode >= num_nodeALL) return;

  f_adh[idNode] = 0.0e+10;
}
// --------------------------------------------------------------------

__device__ void ShellSort_DEV (double *r, int *index, int size)
// ====================================================================
{
  int h = 1, tmp_id, j;
  double tmp_r;
  for (int h_tmp = 1; h_tmp < size/9; h_tmp = h_tmp*3 + 1) h = h_tmp;

  while (h > 0) {
    for (int i = h; i < size; ++i) {
      tmp_r = r[i];
      tmp_id = index[i];
      for (j = i; j >= h && r[j - h] > tmp_r; j -= h) {
        r[j] = r[j - h];
        index[j] = index[j - h];
      }
      r[j] = tmp_r;
      index[j] = tmp_id;
    }
    h /= 3;
  }
}
// --------------------------------------------------------------------

__global__ void InitNeighWall_GPU
// ====================================================================
(
        int         * id_wall,
        double      * r_wall,
        int         num_nodeALL
)
// ====================================================================
{
  int idNode = blockDim.x*blockIdx.x + threadIdx.x;  // sequential nodeID
  if (idNode >= num_nodeALL) return;

  for (int i = 0; i < MAX_NEIGHWALL; i++) {
    id_wall[idNode*MAX_NEIGHWALL + i] = -1;
    r_wall[idNode*MAX_NEIGHWALL + i] = 1.0e+15;
  }
}
// --------------------------------------------------------------------

#if defined(CapsuleSHEARFLOW)
__global__ void StoreNeighWall_CUBE_GPU
// ====================================================================
(
#if(__CUDA_ARCH__ >= 350)
#define RESTRICT __restrict__
#else 
#define RESTRICT 
#endif
        double      * RESTRICT x,
        double      * RESTRICT y,
        double      * RESTRICT z,
        double      * r_wall,
        int         * id_wall,
        bool        * is_knob,
        double      * RESTRICT x_wall,
        double      dz,
        double      dx_lbm,
        double      zmin,
        int         nz,
        int         nx,
        int         num_nodeALL
)
// ====================================================================
{
  int    idNode = blockDim.x*blockIdx.x + threadIdx.x,  // sequential nodeID
         num_neigh = 0, id_j, iz, jz;
  double x_i, x_j, y_i, y_j, z_i, z_j, r;
  #if defined(ANGLE)
  int    tx = 0, tz = 0, nlx, nlz; 
  double xspan, zspan, txj, tzj;
  bool   xflag = false, zflag = false;
  #endif

  if (idNode >= num_nodeALL) return;
  if (is_knob[idNode] == false) return;

  x_i = x[idNode];
  y_i = y[idNode];
  z_i = z[idNode];
  r   = sqrt((y_i - 0.0)*(y_i - 0.0));
  if (r < 0.5*LY - LIMIT*dx_lbm) return;

  iz = (int)((z_i - zmin)/dz); 
  for (int i = 0; i < 2*LIMIT*((int)(dx_lbm/dz)); i++) { // store neighboring wall id
    jz = iz + i - (LIMIT*((int)(dx_lbm/dz)) - 1);
    z_j = (jz - (int)(nz/2))*dz;

    if (jz < 0) {
      jz += nz;
      z_j += LZ;
    } else if (jz >= nz) {
      jz -= nz;
      z_j -= LZ;
    }

    for (int jx = 0; jx < nx; jx++) {
      id_j = jz*nx + jx; 
      x_j = x_wall[id_j*3 + 0];
      y_j = x_wall[id_j*3 + 1]; // -LY*0.5
//      z_j = x_wall[id_j*3 + 2];
      r = sqrt((x_j - x_i)*(x_j - x_i) + (y_j - y_i)*(y_j - y_i) + (z_j - z_i)*(z_j - z_i));
      if (r > LIMIT*dx_lbm) continue;

      #if defined(ANGLE)
      // Angle limitation
      xspan = WIDTH/sin(THETA);
      zspan = WIDTH*cos(THETA);
      nlx = (int)(LX/xspan);
      nlz = (int)(LZ/zspan);
      txj = x_j + LX*0.5;
      tzj = z_j + LZ*0.5;
      while (tx < nlx) {
        if (txj >= (double)(4*nlx-1)*(0.5*xspan) && txj <= (double)(4*nlx+1)*(0.5*xspan)) {
          xflag = true;
          break;
        }
      }
      while (tz < nlz) {
        if (tzj >= (double)(4*nlz-1)*(0.5*zspan) && tzj <= (double)(4*nlz+1)*(0.5*zspan)) {
          zflag = true;
          break;
        }
      }
      if (xflag == false || zflag == false) continue; 
      #endif

      id_wall[idNode*MAX_NEIGHWALL + num_neigh] = id_j;
      r_wall[idNode*MAX_NEIGHWALL + num_neigh] = r;
      num_neigh++;
    }
  } // store End

  if (num_neigh == 0) return;
  ShellSort_DEV(r_wall + idNode*MAX_NEIGHWALL, id_wall + idNode*MAX_NEIGHWALL, num_neigh);
}
// --------------------------------------------------------------------
#endif

#if defined(CapsuleCHANNELFLOW)
__global__ void StoreNeighWall_GPU
// ====================================================================
(
#if(__CUDA_ARCH__ >= 350)
#define RESTRICT __restrict__
#else 
#define RESTRICT 
#endif
        double      * RESTRICT x,
        double      * RESTRICT y,
        double      * RESTRICT z,
        double      * r_wall,
        int         * id_wall,
        bool        * is_knob,
        double      * RESTRICT x_wall,
        double      dz,
        double      dx_lbm,
        double      zmin,
        int         nz,
        int         ntheta,
        double      radius,
        int         num_nodeALL
)
// ====================================================================
{
  int    idNode = blockDim.x*blockIdx.x + threadIdx.x,  // sequential nodeID
         num_neigh = 0, id_j, iz, jz;
  double x_i, x_j, y_i, y_j, z_i, z_j, r;

  if (idNode >= num_nodeALL) return;
  if (is_knob[idNode] == false) return;

  x_i = x[idNode];
  y_i = y[idNode];
  z_i = z[idNode];
  r   = sqrt((x_i - 0.0)*(x_i - 0.0) + (y_i - 0.0)*(y_i - 0.0));
  if (r < radius - LIMIT*dx_lbm) return;

  iz = (int)((z_i - zmin)/dz); 
  for (int i = 0; i < 2*LIMIT*((int)(dx_lbm/dz)); i++) { // store neighboring wall id
    jz = iz + i - (LIMIT*((int)(dx_lbm/dz)) - 1);
    z_j = (jz - (int)(nz/2))*dz;

    if (jz < 0) {
      jz += nz;
      z_j += LZ;
    } else if (jz >= nz) {
      jz -= nz;
      z_j -= LZ;
    }

    for (int jtheta = 0; jtheta < ntheta; jtheta++) {
      id_j = jz*ntheta + jtheta; 
      x_j = x_wall[id_j*3 + 0];
      y_j = x_wall[id_j*3 + 1];
//      z_j = x_wall[id_j*3 + 2];
      r = sqrt((x_j - x_i)*(x_j - x_i) + (y_j - y_i)*(y_j - y_i) + (z_j - z_i)*(z_j - z_i));
      if (r > LIMIT*dx_lbm) continue;
      id_wall[idNode*MAX_NEIGHWALL + num_neigh] = id_j;
      r_wall[idNode*MAX_NEIGHWALL + num_neigh] = r;
      num_neigh++;
    }
  } // store End

  if (num_neigh == 0) return;
  ShellSort_DEV(r_wall + idNode*MAX_NEIGHWALL, id_wall + idNode*MAX_NEIGHWALL, num_neigh);
}
// --------------------------------------------------------------------
#endif

__global__ void AttachmentPointAtWall_GPU
// ====================================================================
(
#if(__CUDA_ARCH__ >= 350)
#define RESTRICT __restrict__
#else 
#define RESTRICT 
#endif
        int         * RESTRICT id_wall,
        int         * attachment_point,
        bool        * is_knob,
        int         * is_attached,
        int         num_nodeALL
)
// ====================================================================
{
  int    idNode = blockDim.x*blockIdx.x + threadIdx.x,  // sequential nodeID
         id_j;
  bool   is_attached_old;

  if (idNode >= num_nodeALL) return;
  if (is_knob[idNode] == false) return;

  for (int spring = 0; spring < NUMSPRING_KNOB; spring++) {
    for (int i = 0; i < MAX_NEIGHWALL; i++) {  
      id_j = id_wall[idNode*MAX_NEIGHWALL + i];
      if (id_j == -1) continue;

      is_attached_old = (bool)atomicOr(&is_attached[id_j], 1);
      if (is_attached_old == true) continue;
      attachment_point[idNode*NUMSPRING_KNOB + spring] = id_j;
      break;
    }
  }
}
// --------------------------------------------------------------------


void adhesion_init(domain *cdo, cell *cel, Wall *wall)
// ====================================================================
{
#if(CYTOADHESION == true)
  int  i;
  char filename[256];
  FILE *fp;
  int  num_knob = 0;
  dim3 dim_grid, dim_block;

  dim_block.x = THREAD; 
  dim_grid.x = cel->adnum*cel->advertex*3/THREAD + MIN((cel->adnum*cel->advertex*3)%THREAD, 1); 

  MallocHostAdhesion(cel);
  MallocDeviceAdhesion(cel);
  Initializing_GPU<<< dim_grid, dim_block >>>(cel->f_adhD, cel->adnum*cel->advertex*3);
  printf("restart = %d\n", cdo->restart);

///*
  for (i = 0;i < cel->adnum*cel->advertex*NUMSPRING_KNOB;i++) {
  // --- initialize detach data
//    cel->konH[i] = -1.0e+10;
//  // --- initialize catch data
//    cel->koffH[i] = -1.0e+10;
  // --- initialize catch data
    cel->r_ijH[i] = -1.0e+10;
  }
//*/

  // --- initialize adhered wall point
  for (i = 0;i < cel->advertex*cel->adnum*NUMSPRING_KNOB; i++) cel->attachment_pointH[i] = -1;
  // --- intialize knob node
  for (i = 0; i < cel->adnum*cel->advertex; i++) cel->is_knobH[i] = false;

  sprintf(filename,"./input/icosa%d.dat",ICOSA2);
  fp = fopen(filename,"r");

  if (fp != NULL) { 
    printf("Read file is %s\n", filename);

    for (i = 0; i < cel->advertex*cel->adnum; i++) {
      fscanf(fp, "%*d %d", &cel->is_knobH[i]);
      if (cel->is_knobH[i] == true) num_knob ++;
    }
    fclose(fp);
  } else {
    fp = fopen(filename,"w");
    if (fp == NULL) {
      printf(" file open error %s\n", filename);
      exit(1);
    }
    for (i = 0; i < cel->advertex*cel->adnum; i++) {
      double random = ((double)(rand() + 1.0))/((double)(RAND_MAX + 2.0));
      if (random < ((double)(KNOB))/cel->advertex) {
        cel->is_knobH[i] = true;
        num_knob ++;
      }
      fprintf(fp, "%d %d\n", i, cel->is_knobH[i]);
    }
    fclose(fp);
  }
  MemcpyAdhesionHtoD(cel);

// Numerical condition ---
  printf("====================================================================\n");
  printf("  Number of knob(microvilli): %d\n",num_knob);
  printf("  Number of spring/konb     : %d\n",NUMSPRING_KNOB);
  printf("  Association rate constant : %9.3e [1/s]\n",K_ON0);
  printf("  dissociation rate constant: %9.3e [1/s]\n",K_OFF0);
  printf("  Reactive compliance       : %9.3e [m]\n",ChiBETA);
  printf("  Adhesion Spring constant  : %9.3e [N]\n",K_AD);
  printf("====================================================================\n");

// Store neighboring wall nodes
  dim_grid.x = cel->advertex*cel->adnum/THREAD + MIN(1, (cel->advertex*cel->adnum)%THREAD); 

  InitNeighWall_GPU<<< dim_grid, dim_block >>>
  (cel->id_wall, cel->r_wall, cel->advertex*cel->adnum);

  #if defined(CapsuleSHEARFLOW)
  StoreNeighWall_CUBE_GPU<<< dim_grid, dim_block >>>
  (cel->xD, cel->yD, cel->zD, cel->r_wall, cel->id_wall, cel->is_knobD, wall->x, wall->dz, cdo->dx, cdo->zmin, wall->nz, wall->nx, cel->vertex*cel->adnum);

  #elif defined(CapsuleCHANNELFLOW)
  StoreNeighWall_GPU<<< dim_grid, dim_block >>>
  (cel->xD, cel->yD, cel->zD, cel->r_wall, cel->id_wall, cel->is_knobD, wall->x, wall->dz, cdo->dx, cdo->zmin, wall->nz, wall->ntheta, wall->radius, cel->advertex*cel->adnum);
  #endif

  AttachmentPointAtWall_GPU<<< dim_grid, dim_block >>>
  (cel->id_wall, cel->attachment_pointD, cel->is_knobD, wall->is_attached, cel->advertex*cel->adnum);

#endif
}
// --------------------------------------------------------------------

void MallocHostWall(Wall *wall)
{
  MallocHost<double>(&(wall->x), wall->num_wall*3);
  MallocHost<int>(&(wall->is_attached), wall->num_wall);
}

void MallocDeviceWall(Wall *wall)
{
  MallocDevice<double>(&(wall->x), wall->num_wall*3);
  MallocDevice<int>(&(wall->is_attached), wall->num_wall);
}

void MemcpyWallHtoD(Wall *wall, Wall *wall_host)
{
  MemcpyHtoD<double>(wall->x, wall_host->x, wall->num_wall*3);
  MemcpyHtoD<int>(wall->is_attached, wall_host->is_attached, wall->num_wall);
}

void MemcpyWallDtoH(Wall *wall, Wall *wall_host)
{
  MemcpyDtoH<double>(wall->x, wall_host->x, wall->num_wall*3);
  MemcpyDtoH<int>(wall->is_attached, wall_host->is_attached, wall->num_wall);
}

void wall_init(domain *cdo, Wall *wall, Wall *wall_host)
// ====================================================================
{
  FILE  *fp;
  char filename[256];  
  int  id;

#if defined(CapsuleSHEARFLOW)
  wall_host->dz       = sqrt(1.0/RECEPTOR_DENSITY); //[m]
  wall_host->dz      /= RC;                         //[-]
  wall_host->nz       = (int)((cdo->zmax - cdo->zmin)/wall_host->dz + 1.0e-03);
  wall_host->nx       = (int)((cdo->xmax - cdo->xmin)/wall_host->dz + 1.0e-03);
  wall_host->dz       = (cdo->zmax - cdo->zmin)/(double)wall_host->nz;
  wall_host->dx       = (cdo->xmax - cdo->xmin)/(double)wall_host->nx;
  wall_host->num_wall = wall_host->nz*wall_host->nx;

  MallocHostWall(wall_host);

  wall->dz       = wall_host->dz;
  wall->dx       = wall_host->dx;
  wall->nz       = wall_host->nz;
  wall->nx       = wall_host->nx;
  wall->num_wall = wall_host->num_wall;
  printf(" Receptor density : %e [receptors/um^2]\n", 1.0/(wall->dz*wall->dx*RC*RC)*1.0e-12);

  for (int iz = 0; iz < wall->nz; iz++) {
    for (int ix = 0; ix < wall->nx; ix++) {
      id = iz*wall->nx + ix; 
      wall_host->x[id*3 + 0] = (ix - (int)(wall->nx/2))*wall->dx;
      wall_host->x[id*3 + 1] = -0.5*LY; 
      wall_host->x[id*3 + 2] = (iz - (int)(wall->nz/2))*wall->dz;
      wall_host->is_attached[id] = false;
    }
  }

#elif defined(CapsuleCHANNELFLOW)
  wall_host->radius   = R;        
//  wall_host->dz       = sqrt(1.0/RECEPTOR_DENSITY)/RC;     
  wall_host->dz       = sqrt(1.0/RECEPTOR_DENSITY);     
  wall_host->dz      /= RC;     
  wall_host->nz       = (int)((cdo->zmax - cdo->zmin)/wall_host->dz + 1.0e-03);
  wall_host->dz       = (cdo->zmax - cdo->zmin)/(double)wall_host->nz;
  wall_host->dtheta   = wall_host->dz/wall_host->radius; // R*dtheta = dz
  wall_host->ntheta   = (int)(2.0*M_PI/wall_host->dtheta);
  wall_host->num_wall = wall_host->nz*wall_host->ntheta;

  MallocHostWall(wall_host);

  wall->radius   = wall_host->radius;
  wall->dz       = wall_host->dz;
  wall->nz       = wall_host->nz;
  wall->dtheta   = wall_host->dtheta;
  wall->ntheta   = wall_host->ntheta;
  wall->num_wall = wall_host->num_wall;
  printf(" Recepotr density : %e [receptors/um^2]\n", 1.0/(wall->dz*wall->dz*RC*RC)*1.0e-12);

  for (int iz = 0; iz < wall->nz; iz++) {
    for (int itheta = 0; itheta < wall->ntheta; itheta++) {
      id = iz*wall->ntheta + itheta; 
      wall_host->x[id*3 + 0] = wall->radius*cos(itheta*wall->dtheta); 
      wall_host->x[id*3 + 1] = wall->radius*sin(itheta*wall->dtheta); 
      wall_host->x[id*3 + 2] = (iz - (int)(wall->nz/2))*wall->dz;
      wall_host->is_attached[id] = false;
    }
  }
#endif

  MallocDeviceWall(wall);
  MemcpyWallHtoD(wall, wall_host);

  sprintf(filename,"./result/wall.bin");
  fp = fopen(filename,"wb");
  if (fp == NULL) error(1);
  fwrite(wall_host->x,sizeof(double),wall_host->num_wall*3,fp);
  fclose(fp);
}
// --------------------------------------------------------------------

__global__ void AdhesionBELL_GPU
// ====================================================================
(
#if(__CUDA_ARCH__ >= 350)
#define RESTRICT __restrict__
#else 
#define RESTRICT 
#endif
        double      * RESTRICT x,
        double      * RESTRICT y,
        double      * RESTRICT z,
        double      * RESTRICT r_wall,
        int         * RESTRICT id_wall,
        int         * attachment_point,
        bool        * is_knob,
        double      * RESTRICT x_wall,
        int         * is_attached,
        double      dx_lbm,
        double      dt,
        double      shear_rate,
        int         num_nodeALL,
        double      * r_ij
)
// ====================================================================
{
  int    idNode = blockDim.x*blockIdx.x + threadIdx.x,  // sequential nodeID
         id_j;
  double rand, dr, dx, dy, dz, f_ad, k_on, k_off, p_on, p_off,
         x_i, y_i, z_i, x_j, y_j, z_j, r;
  bool   is_attached_old;

  if (idNode >= num_nodeALL) return;
  if (is_knob[idNode] == false) return;
  #if defined(WLC)
  double LL = Lcont*dx_lbm; // [m]
  #endif

  curandState s;
  clock_t time = clock();
  curand_init((long)(time + threadIdx.x), idNode*NUMSPRING_KNOB, 0, &s);

  x_i = x[idNode];
  y_i = y[idNode];
  z_i = z[idNode];
  for (int spring = 0; spring < NUMSPRING_KNOB; spring++) {
    rand = curand_uniform(&s);
    if (attachment_point[idNode*NUMSPRING_KNOB + spring] == -1) { // attachment
      for (int i = 0; i < MAX_NEIGHWALL; i++) {  
        id_j = id_wall[idNode*MAX_NEIGHWALL + i];
        if (id_j == -1) continue;
        r    = r_wall[idNode*MAX_NEIGHWALL + i];
        #if defined(LINEAR_SPRING)
        dr   = MAX(0.0, (r - L0*dx_lbm)*RC);                                   // dr   : dimentional [m] 
        f_ad = fabs(K_AD*dr);                                                  // f_ad : dimentional [N]
        #elif defined(WLC)
        dr   = MAX(0.0, 0.25/((1.0 - r/LL)*(1.0 - r/LL)) + r/LL - 0.25);       // dr   : dimentional [-] 
        f_ad = fabs(fWLC*dr);                                                  // f_ad : dimentional [N]
        #endif
        k_on = K_ON0/shear_rate*exp(f_ad*(ChiBETA - 0.5*dr)/kBT);              // k_on : non-dimentional 
        p_on = 1.0 - exp(-k_on*dt*(double)M);                        
//        p_on = 1.0;                        
        if (p_on <= rand) continue;
        is_attached_old = (bool)atomicOr(&is_attached[id_j], 1);
        if (is_attached_old == true) continue;
        attachment_point[idNode*NUMSPRING_KNOB + spring] = id_j;
        r_ij[idNode*NUMSPRING_KNOB + spring] = dr;
        break;
      }
    } else { // detachment
///*
      id_j = attachment_point[idNode*NUMSPRING_KNOB + spring];
      x_j = x_wall[id_j*3 + 0];
      y_j = x_wall[id_j*3 + 1];
      z_j = x_wall[id_j*3 + 2];
      dx = x_j - x_i;
      dy = y_j - y_i;
      dz = z_j - z_i;
      if (fabs(dz) > LZ/2.0) {
        if (z_i > z_j) {
          dz += LZ;
        } else if (z_i < z_j) {
          dz -= LZ;
        }
      }
      r = sqrt(dx*dx + dy*dy + dz*dz);
      #if defined(LINEAR_SPRING)
      dr   = MAX(0.0, (r - L0*dx_lbm)*RC);                                   // dr    : dimentional [m] 
      f_ad = fabs(K_AD*dr);                                                  // f_ad  : dimentional [N]
      #elif defined(WLC)
      dr   = MAX(0.0, 0.25/((1.0 - r/LL)*(1.0 - r/LL)) + r/LL - 0.25);       // dr   : dimentional [-] 
      f_ad = fabs(fWLC*dr);                                                  // f_ad : dimentional [N]
      #endif
      k_off = K_OFF0/shear_rate*exp(f_ad*ChiBETA/kBT);                       // k_off : non-dimentional 
      p_off = 1.0 - exp(-k_off*dt*(double)M);                        
      if (p_off <= rand) continue;
      atomicAnd(&is_attached[id_j], 0);
      attachment_point[idNode*NUMSPRING_KNOB + spring] = -1;
      r_ij[idNode*NUMSPRING_KNOB + spring] = dr;
//*/
    }
  }
}
// --------------------------------------------------------------------

__global__ void AdhesiveSpringForce_GPU
// ====================================================================
(
#if(__CUDA_ARCH__ >= 350)
#define RESTRICT __restrict__
#else 
#define RESTRICT 
#endif
        double      * RESTRICT x,
        double      * RESTRICT y,
        double      * RESTRICT z,
        double      * f_adh,
        double      * fx,
        double      * fy,
        double      * fz,
        int         * attachment_point,
        bool        * is_knob,
        double      * RESTRICT x_wall,
        double      dx_lbm,
        int         num_nodeALL
)
// ====================================================================
{
  int    idNode = blockDim.x*blockIdx.x + threadIdx.x,  // sequential nodeID
         id_j;
  double f[3] = {0.0, 0.0, 0.0}, x_i, y_i, z_i, x_j, y_j, z_j, r, dx, dy, dz, dr;
  #if defined(WLC)
  double LL = Lcont*dx_lbm, ex, repulf = 0.0; // [-]
  #endif
      
  if (idNode >= num_nodeALL) return;
  if (is_knob[idNode] == false) {
    f_adh[idNode*3 + 0] = f[0];
    f_adh[idNode*3 + 1] = f[1];
    f_adh[idNode*3 + 2] = f[2];
    return;
  }

  x_i = x[idNode];
  y_i = y[idNode];
  z_i = z[idNode];
  for (int spring = 0; spring < NUMSPRING_KNOB; spring++) {
    id_j = attachment_point[idNode*NUMSPRING_KNOB + spring];
    if (id_j == -1) continue;
    x_j = x_wall[id_j*3 + 0];
    y_j = x_wall[id_j*3 + 1];
    z_j = x_wall[id_j*3 + 2];
    dx = x_j - x_i;
    dy = y_j - y_i;
    dz = z_j - z_i;
    #if defined(CapsuleSHEARFLOW)
    if (fabs(dx) > LX/2.0) {
      if (x_i > x_j) {
        dx += LX;
      } else if (x_i < x_j) {
        dx -= LX;
      }
    }
    #endif
    if (fabs(dz) > LZ/2.0) {
      if (z_i > z_j) {
        dz += LZ;
      } else if (z_i < z_j) {
        dz -= LZ;
      }
    }
    r = sqrt(dx*dx + dy*dy + dz*dz);
    #if defined(LINEAR_SPRING)
    dr = r - L0*dx_lbm;
    f[0] += K_AD/GS*dr*dx/r;
    f[1] += K_AD/GS*dr*dy/r;
    f[2] += K_AD/GS*dr*dz/r;
    #elif defined(WLC)
    dr = 0.25/((1.0 - r/LL)*(1.0 - r/LL)) + r/LL - 0.25;
    if (fabs(dy) < L0*dx_lbm) {
      ex = exp(-ALPHA2*fabs(dy));
      repulf = ALPHA1/(GS*RA)*ALPHA2*ex/(1.0 - ex)*dy/fabs(dy); // Repulsive force [-]
    }
//    f[0] += 1.0e-06*fWLC/(GS*RA)*dr*dx/r;
//    f[1] += 1.0e-06*fWLC/(GS*RA)*dr*dy/r;
//    f[2] += 1.0e-06*fWLC/(GS*RA)*dr*dz/r;
    f[0] += fWLC/(GS*RA)*dr*dx/r;
    f[1] += fWLC/(GS*RA)*dr*dy/r + repulf;
    f[2] += fWLC/(GS*RA)*dr*dz/r;
    #endif
  }
  f_adh[idNode*3 + 0] = f[0];
  f_adh[idNode*3 + 1] = f[1];
  f_adh[idNode*3 + 2] = f[2];
}
// --------------------------------------------------------------------

void AdhesiveSpringForce (domain *cdo, cell *cel, Wall *wall)
// ====================================================================
{
  dim3 dim_grid, dim_block;

  dim_block.x = THREAD; 
  dim_grid.x = cel->advertex*cel->adnum/THREAD + MIN(1, (cel->advertex*cel->adnum)%THREAD); 

///*
  #if(N_R > 0)
  AdhesiveSpringForce_GPU <<< dim_grid, dim_block >>>
  (cel->xD, cel->yD, cel->zD, cel->f_adhD, cel->fxD, cel->fyD, cel->fzD, cel->attachment_pointD, cel->is_knobD, wall->x, cdo->dx, cel->advertex*cel->adnum);
  #endif
//*/
  #if(N_W > 0)
  AdhesiveSpringForce_GPU <<< dim_grid, dim_block >>>
  (cel->xD_w, cel->yD_w, cel->zD_w, cel->f_adhD, cel->fxD_w, cel->fyD_w, cel->fzD_w, cel->attachment_pointD, cel->is_knobD, wall->x, cdo->dx, cel->advertex*cel->adnum);
  #endif
}
// --------------------------------------------------------------------

void CytoAdhesionBELL (domain *cdo, cell *cel, Wall *wall)
// ====================================================================
{
  dim3 dim_grid, dim_block;

  dim_block.x = THREAD; 
  dim_grid.x = cel->advertex*cel->adnum/THREAD + MIN(1, (cel->advertex*cel->adnum)%THREAD); 

  InitNeighWall_GPU<<< dim_grid, dim_block >>>
  (cel->id_wall, cel->r_wall, cel->advertex*cel->adnum);

///*
#if(N_R > 0)
  #if defined(CapsuleSHEARFLOW)
  StoreNeighWall_CUBE_GPU<<< dim_grid, dim_block >>>
  (cel->xD, cel->yD, cel->zD, cel->r_wall, cel->id_wall, cel->is_knobD, wall->x, wall->dz, cdo->dx, cdo->zmin, wall->nz, wall->nx, cel->vertex*cel->adnum);
  #elif defined(CapsuleCHANNELFLOW)
  StoreNeighWall_GPU<<< dim_grid, dim_block >>>
  (cel->xD, cel->yD, cel->zD, cel->r_wall, cel->id_wall, cel->is_knobD, wall->x, wall->dz, cdo->dx, cdo->zmin, wall->nz, wall->ntheta, wall->radius, cel->advertex*cel->adnum);
  #endif

  AdhesionBELL_GPU<<< dim_grid, dim_block >>>
  (cel->xD, cel->yD, cel->zD, cel->r_wall, cel->id_wall, cel->attachment_pointD, cel->is_knobD, wall->x, wall->is_attached, cdo->dx, cdo->dt, cdo->sr, cel->advertex*cel->adnum, cel->r_ijD);
#endif
//*/
///*
#if(N_W > 0)
  #if defined(CapsuleSHEARFLOW)
  StoreNeighWall_CUBE_GPU<<< dim_grid, dim_block >>>
  (cel->xD_w, cel->yD_w, cel->zD_w, cel->r_wall, cel->id_wall, cel->is_knobD, wall->x, wall->dz, cdo->dx, cdo->zmin, wall->nz, wall->nx, cel->vertex*cel->adnum);
  #elif defined(CapsuleCHANNELFLOW)
  StoreNeighWall_GPU<<< dim_grid, dim_block >>>
  (cel->xD_w, cel->yD_w, cel->zD_w, cel->r_wall, cel->id_wall, cel->is_knobD, wall->x, wall->dz, cdo->dx, cdo->zmin, wall->nz, wall->ntheta, wall->radius, cel->advertex*cel->adnum);
  #endif

  AdhesionBELL_GPU<<< dim_grid, dim_block >>>
  (cel->xD_w, cel->yD_w, cel->zD_w, cel->r_wall, cel->id_wall, cel->attachment_pointD, cel->is_knobD, wall->x, wall->is_attached, cdo->dx, cdo->dt, cdo->sr, cel->advertex*cel->adnum, cel->r_ijD);
#endif
//*/
}
// --------------------------------------------------------------------

void Adhesion (process *prc, domain *cdo, cell *cel, Wall *wall, Wall *wall_host)
// ====================================================================
{
#if(CYTOADHESION == true)
//  if(cel->inflation_flag < NUM_FLAG){
//     if((prc->iter-1)%FILEOUTPUT == 0) cel->inflation_flag ++;
//     return;
//  }

  AdhesiveSpringForce(cdo, cel, wall);
  CytoAdhesionBELL(cdo, cel, wall);

  if ((prc->iter-1)%FILEOUTPUT_ADHESION == 0) {
    FILE       *fp;
    char       filename[256];

    cudaMemcpy(cel->f_adhH            ,cel->f_adhD            ,sizeof(double)*cel->adnum*cel->advertex*3              ,cudaMemcpyDeviceToHost);
    cudaMemcpy(cel->attachment_pointH ,cel->attachment_pointD ,sizeof(int   )*cel->adnum*cel->advertex*NUMSPRING_KNOB ,cudaMemcpyDeviceToHost);
    cudaMemcpy(cel->is_knobH          ,cel->is_knobD          ,sizeof(bool  )*cel->adnum*cel->advertex                ,cudaMemcpyDeviceToHost);
    cudaMemcpy(wall_host->is_attached ,wall->is_attached      ,sizeof(int   )*wall_host->num_wall                     ,cudaMemcpyDeviceToHost);
//    cudaMemcpy(cel->konH              ,cel->konD              ,sizeof(double)*cel->adnum*cel->advertex*NUMSPRING_KNOB ,cudaMemcpyDeviceToHost);
//    cudaMemcpy(cel->koffH             ,cel->koffD             ,sizeof(double)*cel->adnum*cel->advertex*NUMSPRING_KNOB ,cudaMemcpyDeviceToHost);
    cudaMemcpy(cel->r_ijH             ,cel->r_ijD             ,sizeof(double)*cel->adnum*cel->advertex*NUMSPRING_KNOB ,cudaMemcpyDeviceToHost);

    sprintf(filename,"./result/adhesion/adhe%05d.bin",prc->fnum_ad);
    fp = fopen(filename,"wb");
    if(fp == NULL) error(1);

    fwrite(&prc->fnum_ad         ,sizeof(int   ),1                                      ,fp);
    fwrite(&prc->iter            ,sizeof(int   ),1                                      ,fp);
    fwrite(cel->f_adhH           ,sizeof(double),cel->adnum*cel->advertex*3             ,fp);
    fwrite(cel->attachment_pointH,sizeof(int   ),cel->adnum*cel->advertex*NUMSPRING_KNOB,fp);
    fwrite(cel->is_knobH         ,sizeof(bool  ),cel->adnum*cel->advertex               ,fp);
    fwrite(wall_host->is_attached,sizeof(int   ),wall_host->num_wall                    ,fp);
//    fwrite(cel->konH             ,sizeof(double),cel->adnum*cel->advertex*NUMSPRING_KNOB,fp);
//    fwrite(cel->koffH            ,sizeof(double),cel->adnum*cel->advertex*NUMSPRING_KNOB,fp);
    fwrite(cel->r_ijH            ,sizeof(double),cel->adnum*cel->advertex*NUMSPRING_KNOB,fp);
    fclose(fp);

    //restart file
    sprintf(filename,"./result/adhesion/restart_adhe.dat");
    fp = fopen(filename,"w");
    if (fp == NULL) error(1);

    fprintf(fp,"%d\n",prc->fnum_ad);
    fprintf(fp,"%d\n",prc->iter);
    fclose(fp);

    prc->fnum_ad ++;
  }
#endif
}
// --------------------------------------------------------------------
