// Direction
// c0  ( 0  0  0 )
// c1  ( 1  0  0 )
// c2  (-1  0  0 )
// c3  ( 0  1  0 )
// c4  ( 0 -1  0 )
// c5  ( 0  0  1 )
// c6  ( 0  0 -1 )
// c7  ( 1  1  0 )
// c8  (-1 -1  0 )
// c9  (-1  1  0 )
// c10 ( 1 -1  0 )
// c11 ( 0  1  1 )
// c12 ( 0 -1 -1 )
// c13 ( 0 -1  1 )
// c14 ( 0  1 -1 )
// c15 ( 1  0  1 )
// c16 (-1  0 -1 )
// c17 ( 1  0 -1 )
// c18 (-1  0  1 )

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "thrust/device_ptr.h"
#include "thrust/transform_reduce.h"
#include "thrust/functional.h"

#include "def.h"
#include "malloc.cuh"

__device__ void GetDirectionVector_DEV (int direction, int* ix, int* iy, int* iz)
{
    if      (direction ==  0) { (*ix) =  0; (*iy) =  0; (*iz) =  0; }
    else if (direction ==  1) { (*ix) =  1; (*iy) =  0; (*iz) =  0; }
    else if (direction ==  2) { (*ix) = -1; (*iy) =  0; (*iz) =  0; }
    else if (direction ==  3) { (*ix) =  0; (*iy) =  1; (*iz) =  0; }
    else if (direction ==  4) { (*ix) =  0; (*iy) = -1; (*iz) =  0; }
    else if (direction ==  5) { (*ix) =  0; (*iy) =  0; (*iz) =  1; }
    else if (direction ==  6) { (*ix) =  0; (*iy) =  0; (*iz) = -1; }
    else if (direction ==  7) { (*ix) =  1; (*iy) =  1; (*iz) =  0; }
    else if (direction ==  8) { (*ix) = -1; (*iy) = -1; (*iz) =  0; }
    else if (direction ==  9) { (*ix) = -1; (*iy) =  1; (*iz) =  0; }
    else if (direction == 10) { (*ix) =  1; (*iy) = -1; (*iz) =  0; }
    else if (direction == 11) { (*ix) =  0; (*iy) =  1; (*iz) =  1; }
    else if (direction == 12) { (*ix) =  0; (*iy) = -1; (*iz) = -1; }
    else if (direction == 13) { (*ix) =  0; (*iy) = -1; (*iz) =  1; }
    else if (direction == 14) { (*ix) =  0; (*iy) =  1; (*iz) = -1; }
    else if (direction == 15) { (*ix) =  1; (*iy) =  0; (*iz) =  1; }
    else if (direction == 16) { (*ix) = -1; (*iy) =  0; (*iz) = -1; }
    else if (direction == 17) { (*ix) =  1; (*iy) =  0; (*iz) = -1; }
    else if (direction == 18) { (*ix) = -1; (*iy) =  0; (*iz) =  1; }
    else                      { (*ix) =  0; (*iy) =  0; (*iz) =  0; }
}

__device__ double BounceBackLinearInterpolation_DEV
//==========================================================
//
//          f2     f3   f1  wall
//         o->     <- o ->   |   o 
//
//                 || after streaming step
//                 \/
//
//          f3     f2   f1  wall
//         o<-     -> o <-   |   o 
(
    double    r,
    double    f1,
    double    f2,
    double    f3
)
//----------------------------------------------------------
{
  double f;
  if (r < 0.0) {
    f = f1;
  } else if (r < 0.5) {
    f = 2.0*r*f1 + (1.0 - 2.0*r)*f2;
//    f = f1;
  } else if (r <= 1.0) {
    f = 1.0/(2.0*r)*f1 + (2.0*r - 1.0)/(2.0*r)*f3;
//    f = f1;
  } else {
    f = f1;
  }
  return f;
}

template<typename T>
struct positive_value : public thrust::unary_function<T,T>
{
  __host__ __device__ T operator()(const T &x) const
  {
    return x < T(0) ? 0 : 1;
  }
};

template <typename T>
int NumCountPositive_Thrust(T *value, int num)
//==========================================================
{
  int result;
  thrust::device_ptr<T>start(value);
  thrust::device_ptr<T>end(value + num);
  result = thrust::transform_reduce(start, end, positive_value<T>(), 0, thrust::plus<T>());
  return result;
}

#ifdef CapsuleCHANNELFLOW

__global__ void  SetDistanceFromWall_GPU
//==========================================================
//
//  Distance from cylindrical wall
//
//
(
      int       *bc,
      int       *bblist,   // output
      double    *distance, // output
      double    dx_lbm,
      int       nx,
      int       ny,
      int       all
)
//----------------------------------------------------------
{
    int    i = blockDim.x*blockIdx.x + threadIdx.x, 
           id = i/Q, direction = i%Q, cnt,
           ix, iy, iz, id_j;
    double x_i, y_i, z_i, x_j, y_j, z_j, 
           dx, dy, dz, t, t1, t2, Det, 
           x_wall, y_wall, z_wall, r, r0;
    if(i >= all)       return;
    if(bc[id] != COMP) return;

    GetDirectionVector_DEV(direction, &ix, &iy, &iz);
    id_j = id + ix*1 + iy*nx + iz*nx*ny;
    if (bc[id_j] != WALL) return;

    iz = (int)(id/(nx*ny));
    iy = (int)(id/nx) - ny*iz;
    ix = id - nx*iy - nx*ny*iz;
    x_i = -LX/2.0 - 1.5*dx_lbm + (double)ix*dx_lbm;
    y_i = -LY/2.0 - 1.5*dx_lbm + (double)iy*dx_lbm;
    z_i = -LZ/2.0 - 1.5*dx_lbm + (double)iz*dx_lbm;

    iz = (int)(id_j/(nx*ny));
    iy = (int)(id_j/nx) - ny*iz;
    ix = id_j - nx*iy - nx*ny*iz;
    x_j = -LX/2.0 - 1.5*dx_lbm + (double)ix*dx_lbm;
    y_j = -LY/2.0 - 1.5*dx_lbm + (double)iy*dx_lbm;
    z_j = -LZ/2.0 - 1.5*dx_lbm + (double)iz*dx_lbm;

    dx = x_j - x_i;
    dy = y_j - y_i;
    dz = z_j - z_i;
    r0 = sqrt(dx*dx + dy*dy + dz*dz);
    Det = (x_i*dx + y_i*dy)*(x_i*dx + y_i*dy) 
        - (dx*dx + dy*dy)*(x_i*x_i + y_i*y_i - R*R);
    if (Det < 0.0) return;
    t1 = (-x_i*dx - y_i*dy + sqrt(Det))/(dx*dx + dy*dy);
    t2 = (-x_i*dx - y_i*dy - sqrt(Det))/(dx*dx + dy*dy);
    if (t1 > 0.0 && t1 < 1.0) {
      t = t1;
    } else if (t2 > 0.0 && t2 < 1.0){
      t = t2;
    } else {
      return;
    }
    x_wall = x_i + t*dx;
    y_wall = y_i + t*dy;
    z_wall = z_i + t*dz;
    r = (x_i - x_wall)*(x_i - x_wall) 
      + (y_i - y_wall)*(y_i - y_wall) 
      + (z_i - z_wall)*(z_i - z_wall);
    r = sqrt(r);
    
    cnt = atomicAdd(&bblist[all], 1);
    bblist[cnt+1] = i;
    distance[i] = r/r0;
//    printf("%d %d %d %d %e %e %e %e\n", cnt, id, i, direction, x_wall, y_wall, z_wall, distance[i]);
}


__global__ void lbm_BounceBack_GPU
//==========================================================
//
//
//
(
#if(__CUDA_ARCH__ >= 350)
#define RESTRICT __restrict__
#else 
#define RESTRICT 
#endif
    double    * fn,
    int       * RESTRICT bc,
    double    *distance,
    int       *bblist,
    int       nx,
    int       ny,
    int       nz,
    int       all
)
//----------------------------------------------------------
{
    int    index = blockDim.x*blockIdx.x + threadIdx.x; 
    if(index >= all) return;
    int    i = bblist[index], 
           id = i/Q, direction = i%Q,
           opposite = (direction == 0 ? 0 : direction + (((direction%2) << 1) - 1)),
           ix, iy, iz, id_j;
    double r, f1, f2, f3, f; 

    r = distance[i];
    GetDirectionVector_DEV(opposite, &ix, &iy, &iz);
    id_j = id + ix*1 + iy*nx + iz*nx*ny;
    if (bc[id_j] == BUFF) { // periodic
      id_j = id + ix*1 + iy*nx - iz*nx*ny*(nz - 5);
    }

    f1 = fn[id*Q + opposite];
    f2 = fn[id*Q + direction];
    f3 = fn[id_j*Q + opposite];
    f  = BounceBackLinearInterpolation_DEV(r, f1, f2, f3);
    fn[id*Q + opposite] = f;
//    printf(" %d %d %d %d %e\n", i, id, direction, opposite, f);
}

__global__ void  lbm_CapsuleCHANNELFLOW
//==========================================================
//
//  LATTICE BOLTZMANN METHOD ON GPU
//
//
(
#if(__CUDA_ARCH__ >= 350)
#define RESTRICT __restrict__
#else 
#define RESTRICT 
#endif
    int       iter,
    double    tau,
    double    * RESTRICT fn,
    double    *fm,
    double    *dn,
    double    *un,
    double    *vn,
    double    *wn,
    double    *dm,
    double    *um,
    double    *vm,
    double    *wm,
    double    *fx,
    double    *fy,
    double    *fz,
    double    *vf1,
//    double    *vf2,
    int       * RESTRICT bc,
    int       nx,
    int       ny,
    int       nz,
    int       n
)
//----------------------------------------------------------
{
    int       i, j;
    double    td, tu, tv, tw, tfx, tfy, tfz, tvf1, tvf2, uvw, Cr2, om, om1, om2, om3,
              f___, fp__, fm__, f_p_, f_m_, f__p, f__m,
              fpp_, fmm_, fmp_, fpm_,
              f_pp, f_mm, f_mp, f_pm,
              fp_p, fm_m, fp_m, fm_p;
    double    ui = 0.0, vi = 0.0, wi;

    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= n)        return;
    if(bc[i] != COMP) return;
    if(iter > FLOWSTART)wi = Uc;
    else                wi = 0.0;

// Distribution function @ n step ---
    f___ = fn[i*Q+ 0];
    fp__ = fn[i*Q+ 1]; fm__ = fn[i*Q+ 2];
    f_p_ = fn[i*Q+ 3]; f_m_ = fn[i*Q+ 4];
    f__p = fn[i*Q+ 5]; f__m = fn[i*Q+ 6];
    fpp_ = fn[i*Q+ 7]; fmm_ = fn[i*Q+ 8];
    fmp_ = fn[i*Q+ 9]; fpm_ = fn[i*Q+10];
    f_pp = fn[i*Q+11]; f_mm = fn[i*Q+12];
    f_mp = fn[i*Q+13]; f_pm = fn[i*Q+14];
    fp_p = fn[i*Q+15]; fm_m = fn[i*Q+16];
    fp_m = fn[i*Q+17]; fm_p = fn[i*Q+18];
// Density and Velocity ---
    if((iter-1)%M == 0){
        dm[i] = 0.0;
        um[i] = 0.0;
        vm[i] = 0.0;
        wm[i] = 0.0;
    }

    tfx  = fx[i];
    tfy  = fy[i];
    tfz  = fz[i];
    tvf1 = vf1[i];
//    tvf2 = vf2[i];

    td  = f___
        + fp__ + fm__ + f_p_ + f_m_ + f__p + f__m
        + fpp_ + fmm_ + fmp_ + fpm_
        + f_pp + f_mm + f_mp + f_pm
        + fp_p + fm_m + fp_m + fm_p;
    tu  = (fp__ - fm__ + fpp_ - fmm_ - fmp_ + fpm_ + fp_p - fm_m + fp_m - fm_p + td*P*ui + td*B*tfx)/(td*Cr);
    tv  = (f_p_ - f_m_ + fpp_ - fmm_ + fmp_ - fpm_ + f_pp - f_mm - f_mp + f_pm + td*P*vi + td*B*tfy)/(td*Cr);
    tw  = (f__p - f__m + f_pp - f_mm + f_mp - f_pm + fp_p - fm_m - fp_m + fm_p + td*P*wi + td*B*tfz)/(td*Cr);

    dn[i]  = td;
    un[i]  = tu;
    vn[i]  = tv;
    wn[i]  = tw;
    dm[i] += td;
    um[i] += tu;
    vm[i] += tv;
    wm[i] += tw;

// Collision step ---
//  om  = 1.0/tau;
//    om  = 1.0/(3.0*(1.0 + (lambda - 1.0)*tvf)*Df + 0.5);
    om  = 1.0/(3.0*(1.0 + (LAMBDA1 - 1.0)*tvf1)*Df + 0.5);
//    om  = 1.0/(3.0*(1.0 + (LAMBDA1 - 1.0)*tvf1 + (LAMBDA2 - 1.0)*tvf2)*Df + 0.5);
    om1 = 1.0 - om;
    om2 = (1.0 - om/2.0)*2.0*td*P;
    om3 = (1.0 - om/2.0)*2.0*td*B;
    Cr2 = Cr*Cr;
    uvw = -1.5*(tu*tu + tv*tv + tw*tw)*Cr2;

    f___ = om1*f___ + om*td/3.0 *(1.0                                                  + uvw)  // ___
         + om2/3.0*(3.0*-tu*Cr*ui  + 3.0*-tv*Cr*vi  + 3.0*-tw*Cr*wi)
         + om3/3.0*(3.0*-tu*Cr*tfx + 3.0*-tv*Cr*tfy + 3.0*-tw*Cr*tfz);

    fp__ = om1*fp__ + om*td/18.0*(1.0 + 3.0*tu*Cr        + 4.5*tu*tu*Cr2               + uvw)  // p__
         + om2/18.0*((3.0*( 1.0-tu*Cr)+9.0*Cr*tu)*ui  + 3.0*-tv*Cr*vi  + 3.0*-tw*Cr*wi)
         + om3/18.0*((3.0*( 1.0-tu*Cr)+9.0*Cr*tu)*tfx + 3.0*-tv*Cr*tfy + 3.0*-tw*Cr*tfz);

    fm__ = om1*fm__ + om*td/18.0*(1.0 - 3.0*tu*Cr        + 4.5*tu*tu*Cr2               + uvw)  // m__
         + om2/18.0*((3.0*(-1.0-tu*Cr)+9.0*Cr*tu)*ui  + 3.0*-tv*Cr*vi  + 3.0*-tw*Cr*wi)
         + om3/18.0*((3.0*(-1.0-tu*Cr)+9.0*Cr*tu)*tfx + 3.0*-tv*Cr*tfy + 3.0*-tw*Cr*tfz);

    f_p_ = om1*f_p_ + om*td/18.0*(1.0 + 3.0*tv*Cr        + 4.5*tv*tv*Cr2               + uvw)  // _p_
         + om2/18.0*(3.0*-tu*Cr*ui  + (3.0*( 1.0-tv*Cr)+9.0*Cr*tv)*vi  + 3.0*-tw*Cr*wi)
         + om3/18.0*(3.0*-tu*Cr*tfx + (3.0*( 1.0-tv*Cr)+9.0*Cr*tv)*tfy + 3.0*-tw*Cr*tfz);

    f_m_ = om1*f_m_ + om*td/18.0*(1.0 - 3.0*tv*Cr        + 4.5*tv*tv*Cr2               + uvw)  // _m_
         + om2/18.0*(3.0*-tu*Cr*ui  + (3.0*(-1.0-tv*Cr)+9.0*Cr*tv)*vi  + 3.0*-tw*Cr*wi)
         + om3/18.0*(3.0*-tu*Cr*tfx + (3.0*(-1.0-tv*Cr)+9.0*Cr*tv)*tfy + 3.0*-tw*Cr*tfz);

    f__p = om1*f__p + om*td/18.0*(1.0 + 3.0*tw*Cr        + 4.5*tw*tw*Cr2               + uvw)  // __p
         + om2/18.0*(3.0*-tu*Cr*ui  + 3.0*-tv*Cr*vi  + (3.0*( 1.0-tw*Cr)+9.0*Cr*tw)*wi)
         + om3/18.0*(3.0*-tu*Cr*tfx + 3.0*-tv*Cr*tfy + (3.0*( 1.0-tw*Cr)+9.0*Cr*tw)*tfz);

    f__m = om1*f__m + om*td/18.0*(1.0 - 3.0*tw*Cr        + 4.5*tw*tw*Cr2               + uvw)  // __m
         + om2/18.0*(3.0*-tu*Cr*ui  + 3.0*-tv*Cr*vi  + (3.0*(-1.0-tw*Cr)+9.0*Cr*tw)*wi)
         + om3/18.0*(3.0*-tu*Cr*tfx + 3.0*-tv*Cr*tfy + (3.0*(-1.0-tw*Cr)+9.0*Cr*tw)*tfz);

    fpp_ = om1*fpp_ + om*td/36.0*(1.0 + 3.0*(tu + tv)*Cr + 4.5*(tu + tv)*(tu + tv)*Cr2 + uvw)  // pp_
         + om2/36.0*((3.0*( 1.0-tu*Cr)+9.0*Cr*(tu+tv))*ui  + (3.0*( 1.0-tv*Cr)+9.0*Cr*(tu+tv))*vi  + 3.0*-tw*Cr*wi)
         + om3/36.0*((3.0*( 1.0-tu*Cr)+9.0*Cr*(tu+tv))*tfx + (3.0*( 1.0-tv*Cr)+9.0*Cr*(tu+tv))*tfy + 3.0*-tw*Cr*tfz);

    fmm_ = om1*fmm_ + om*td/36.0*(1.0 - 3.0*(tu + tv)*Cr + 4.5*(tu + tv)*(tu + tv)*Cr2 + uvw)  // mm_
         + om2/36.0*((3.0*(-1.0-tu*Cr)+9.0*Cr*(tu+tv))*ui  + (3.0*(-1.0-tv*Cr)+9.0*Cr*(tu+tv))*vi  + 3.0*-tw*Cr*wi)
         + om3/36.0*((3.0*(-1.0-tu*Cr)+9.0*Cr*(tu+tv))*tfx + (3.0*(-1.0-tv*Cr)+9.0*Cr*(tu+tv))*tfy + 3.0*-tw*Cr*tfz);

    fmp_ = om1*fmp_ + om*td/36.0*(1.0 - 3.0*(tu - tv)*Cr + 4.5*(tu - tv)*(tu - tv)*Cr2 + uvw)  // mp_
         + om2/36.0*((3.0*(-1.0-tu*Cr)+9.0*Cr*(tu-tv))*ui  + (3.0*( 1.0-tv*Cr)-9.0*Cr*(tu-tv))*vi  + 3.0*-tw*Cr*wi)
         + om3/36.0*((3.0*(-1.0-tu*Cr)+9.0*Cr*(tu-tv))*tfx + (3.0*( 1.0-tv*Cr)-9.0*Cr*(tu-tv))*tfy + 3.0*-tw*Cr*tfz);

    fpm_ = om1*fpm_ + om*td/36.0*(1.0 + 3.0*(tu - tv)*Cr + 4.5*(tu - tv)*(tu - tv)*Cr2 + uvw)  // pm_
         + om2/36.0*((3.0*( 1.0-tu*Cr)+9.0*Cr*(tu-tv))*ui  + (3.0*(-1.0-tv*Cr)-9.0*Cr*(tu-tv))*vi  + 3.0*-tw*Cr*wi)
         + om3/36.0*((3.0*( 1.0-tu*Cr)+9.0*Cr*(tu-tv))*tfx + (3.0*(-1.0-tv*Cr)-9.0*Cr*(tu-tv))*tfy + 3.0*-tw*Cr*tfz);

    f_pp = om1*f_pp + om*td/36.0*(1.0 + 3.0*(tv + tw)*Cr + 4.5*(tv + tw)*(tv + tw)*Cr2 + uvw)  // _pp
         + om2/36.0*(3.0*-tu*Cr*ui  + (3.0*( 1.0-tv*Cr)+9.0*Cr*(tv+tw))*vi  + (3.0*( 1.0-tw*Cr)+9.0*Cr*(tv+tw))*wi)
         + om3/36.0*(3.0*-tu*Cr*tfx + (3.0*( 1.0-tv*Cr)+9.0*Cr*(tv+tw))*tfy + (3.0*( 1.0-tw*Cr)+9.0*Cr*(tv+tw))*tfz);

    f_mm = om1*f_mm + om*td/36.0*(1.0 - 3.0*(tv + tw)*Cr + 4.5*(tv + tw)*(tv + tw)*Cr2 + uvw)  // _mm
         + om2/36.0*(3.0*-tu*Cr*ui  + (3.0*(-1.0-tv*Cr)+9.0*Cr*(tv+tw))*vi  + (3.0*(-1.0-tw*Cr)+9.0*Cr*(tv+tw))*wi)
         + om3/36.0*(3.0*-tu*Cr*tfx + (3.0*(-1.0-tv*Cr)+9.0*Cr*(tv+tw))*tfy + (3.0*(-1.0-tw*Cr)+9.0*Cr*(tv+tw))*tfz);

    f_mp = om1*f_mp + om*td/36.0*(1.0 - 3.0*(tv - tw)*Cr + 4.5*(tv - tw)*(tv - tw)*Cr2 + uvw)  // _mp
         + om2/36.0*(3.0*-tu*Cr*ui  + (3.0*(-1.0-tv*Cr)+9.0*Cr*(tv-tw))*vi  + (3.0*( 1.0-tw*Cr)-9.0*Cr*(tv-tw))*wi)
         + om3/36.0*(3.0*-tu*Cr*tfx + (3.0*(-1.0-tv*Cr)+9.0*Cr*(tv-tw))*tfy + (3.0*( 1.0-tw*Cr)-9.0*Cr*(tv-tw))*tfz);

    f_pm = om1*f_pm + om*td/36.0*(1.0 + 3.0*(tv - tw)*Cr + 4.5*(tv - tw)*(tv - tw)*Cr2 + uvw)  // _pm
         + om2/36.0*(3.0*-tu*Cr*ui  + (3.0*( 1.0-tv*Cr)+9.0*Cr*(tv-tw))*vi  + (3.0*(-1.0-tw*Cr)-9.0*Cr*(tv-tw))*wi)
         + om3/36.0*(3.0*-tu*Cr*tfx + (3.0*( 1.0-tv*Cr)+9.0*Cr*(tv-tw))*tfy + (3.0*(-1.0-tw*Cr)-9.0*Cr*(tv-tw))*tfz);

    fp_p = om1*fp_p + om*td/36.0*(1.0 + 3.0*(tw + tu)*Cr + 4.5*(tw + tu)*(tw + tu)*Cr2 + uvw)  // p_p
         + om2/36.0*((3.0*( 1.0-tu*Cr)+9.0*Cr*(tw+tu))*ui  + 3.0*-tv*Cr*vi  + (3.0*( 1.0-tw*Cr)+9.0*Cr*(tw+tu))*wi)
         + om3/36.0*((3.0*( 1.0-tu*Cr)+9.0*Cr*(tw+tu))*tfx + 3.0*-tv*Cr*tfy + (3.0*( 1.0-tw*Cr)+9.0*Cr*(tw+tu))*tfz);

    fm_m = om1*fm_m + om*td/36.0*(1.0 - 3.0*(tw + tu)*Cr + 4.5*(tw + tu)*(tw + tu)*Cr2 + uvw)  // m_m
         + om2/36.0*((3.0*(-1.0-tu*Cr)+9.0*Cr*(tw+tu))*ui  + 3.0*-tv*Cr*vi  + (3.0*(-1.0-tw*Cr)+9.0*Cr*(tw+tu))*wi)
         + om3/36.0*((3.0*(-1.0-tu*Cr)+9.0*Cr*(tw+tu))*tfx + 3.0*-tv*Cr*tfy + (3.0*(-1.0-tw*Cr)+9.0*Cr*(tw+tu))*tfz);

    fp_m = om1*fp_m + om*td/36.0*(1.0 - 3.0*(tw - tu)*Cr + 4.5*(tw - tu)*(tw - tu)*Cr2 + uvw)  // p_m
         + om2/36.0*((3.0*( 1.0-tu*Cr)-9.0*Cr*(tw-tu))*ui  + 3.0*-tv*Cr*vi  + (3.0*(-1.0-tw*Cr)+9.0*Cr*(tw-tu))*wi)
         + om3/36.0*((3.0*( 1.0-tu*Cr)-9.0*Cr*(tw-tu))*tfx + 3.0*-tv*Cr*tfy + (3.0*(-1.0-tw*Cr)+9.0*Cr*(tw-tu))*tfz);

    fm_p = om1*fm_p + om*td/36.0*(1.0 + 3.0*(tw - tu)*Cr + 4.5*(tw - tu)*(tw - tu)*Cr2 + uvw)  // m_p
         + om2/36.0*((3.0*(-1.0-tu*Cr)-9.0*Cr*(tw-tu))*ui  + 3.0*-tv*Cr*vi  + (3.0*( 1.0-tw*Cr)+9.0*Cr*(tw-tu))*wi)
         + om3/36.0*((3.0*(-1.0-tu*Cr)-9.0*Cr*(tw-tu))*tfx + 3.0*-tv*Cr*tfy + (3.0*( 1.0-tw*Cr)+9.0*Cr*(tw-tu))*tfz);

// Streaming step & Bounce back ---
    j = i*Q + 0; fm[j] = f___;                             // ___
    if(bc[i+1] == WALL) j = i*Q + 2; else j = (i+1)*Q + 1; // p__
    fm[j] = fp__;
    if(bc[i-1] == WALL) j = i*Q + 1; else j = (i-1)*Q + 2; // m__
    fm[j] = fm__;
    if(bc[i+nx] == WALL) j = i*Q + 4; else j = (i+nx)*Q + 3; // _p_
    fm[j] = f_p_;
    if(bc[i-nx] == WALL) j = i*Q + 3; else j = (i-nx)*Q + 4; // _m_
    fm[j] = f_m_;
    if(bc[i+nx*ny] == BUFF) j = (i-nx*ny*(nz-5))*Q + 5; else j = (i+nx*ny)*Q + 5; // __p
    fm[j] = f__p;
    if(bc[i-nx*ny] == BUFF) j = (i+nx*ny*(nz-5))*Q + 6; else j = (i-nx*ny)*Q + 6; // __m
    fm[j] = f__m;
    if(bc[i+1+nx] == WALL) j = i*Q + 8; else j = (i+1+nx)*Q + 7; // pp_
    fm[j] = fpp_;
    if(bc[i-1-nx] == WALL) j = i*Q + 7; else j = (i-1-nx)*Q + 8; // mm_
    fm[j] = fmm_;
    if(bc[i-1+nx] == WALL) j = i*Q +10; else j = (i-1+nx)*Q + 9; // mp_
    fm[j] = fmp_;
    if(bc[i+1-nx] == WALL) j = i*Q + 9; else j = (i+1-nx)*Q +10; // pm_
    fm[j] = fpm_;
    if(bc[i+nx+nx*ny] == WALL) j = i*Q + 12; 
    else if(bc[i+nx+nx*ny] == BUFF) j = (i+nx-nx*ny*(nz-5))*Q +11;
    else j = (i+nx+nx*ny)*Q + 11;                                // _pp
    fm[j] = f_pp;
    if(bc[i-nx-nx*ny] == WALL) j = i*Q + 11; 
    else if(bc[i-nx-nx*ny] == BUFF) j = (i-nx+nx*ny*(nz-5))*Q +12;
    else j = (i-nx-nx*ny)*Q + 12;                                // _mm
    fm[j] = f_mm;
    if(bc[i-nx+nx*ny] == WALL) j = i*Q + 14; 
    else if(bc[i-nx+nx*ny] == BUFF) j = (i-nx-nx*ny*(nz-5))*Q +13;
    else j = (i-nx+nx*ny)*Q + 13;                                // _mp
    fm[j] = f_mp;
    if(bc[i+nx-nx*ny] == WALL) j = i*Q + 13; 
    else if(bc[i+nx-nx*ny] == BUFF) j = (i+nx+nx*ny*(nz-5))*Q +14;
    else j = (i+nx-nx*ny)*Q + 14;                                // _pm
    fm[j] = f_pm;
    if(bc[i+1+nx*ny] == WALL) j = i*Q + 16; 
    else if(bc[i+1+nx*ny] == BUFF) j = (i+1-nx*ny*(nz-5))*Q +15;
    else j = (i+1+nx*ny)*Q + 15;                                 // p_p
    fm[j] = fp_p;
    if(bc[i-1-nx*ny] == WALL) j = i*Q + 15; 
    else if(bc[i-1-nx*ny] == BUFF) j = (i-1+nx*ny*(nz-5))*Q +16;
    else j = (i-1-nx*ny)*Q + 16;                                 // m_m
    fm[j] = fm_m;
    if(bc[i+1-nx*ny] == WALL) j = i*Q + 18; 
    else if(bc[i+1-nx*ny] == BUFF) j = (i+1+nx*ny*(nz-5))*Q +17;
    else j = (i+1-nx*ny)*Q + 17;                                 // p_m
    fm[j] = fp_m;
    if(bc[i-1+nx*ny] == WALL) j = i*Q + 17; 
    else if(bc[i-1+nx*ny] == BUFF) j = (i-1-nx*ny*(nz-5))*Q +18;
    else j = (i-1+nx*ny)*Q + 18;                                 // m_p
    fm[j] = fm_p;
}
#endif // CapsuleCHANNELFLOW

#if defined(CHANNELFLOW) || defined(CapsuleCHANNELFLOW)
__global__ void  lbm_CHANNELFLOW_Boundary
//==========================================================
//
//  LATTICE BOLTZMANN METHOD ON GPU
//
//
(
    double    *dn,
    double    *un,
    double    *vn,
    double    *wn,
    double    *dm,
    double    *um,
    double    *vm,
    double    *wm,
    int       *bc,
    int       nx,
    int       ny,
    int       nz,
    int       n
)
//----------------------------------------------------------
{
    int       i, iz, i1, i2;


    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= n) return;

    if(bc[i] == COMP){
        return;
    }
    else if(bc[i] == BUFF){
        iz = (int)(i/(nx*ny));

        i1 = i;
        i2 = i1;

        if(iz <= 1   ) i2 += nx*ny*(nz-4);
        if(iz >= nz-2) i2 -= nx*ny*(nz-4);

        dn[i1] = dn[i2];
        un[i1] = un[i2];
        vn[i1] = vn[i2];
        wn[i1] = wn[i2];
        dm[i1] = dm[i2];
        um[i1] = um[i2];
        vm[i1] = vm[i2];
        wm[i1] = wm[i2];
    }
    else if(bc[i] == WALL){
        return;
    }
    else return;
}
#endif // CHANNELFLOW || CapsuleCHANNELFLOW

__global__ void  lbm_AVERAGE
//==========================================================
//
//  LATTICE BOLTZMANN METHOD ON GPU
//
//
(
    double    *um,
    double    *vm,
    double    *wm,
    int       n
)
//----------------------------------------------------------
{
    int       i;

    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= n) return;

    um[i] /= (double)M;
    vm[i] /= (double)M;
    wm[i] /= (double)M;

    return;
}

void  lbm_d3q19
//==========================================================
//
//  LATTICE BOLTZMANN METHOD
//
//
(
    process   *prc,
    domain    *cdo,
    lattice   *ltc
)
//----------------------------------------------------------
{
    int         thread = 128, block = cdo->n/thread+1;
    dim3        dimB(thread,1,1), dimG(block,1,1);
    static bool is_firststep = true;

#if defined(CapsuleCHANNELFLOW)
    double    *tmp;
    if (is_firststep == true) {
      MallocDevice<double>(&(ltc->distance), cdo->n*Q);
      MallocDevice<int>(&(ltc->bblist), cdo->n*Q + 1);
      checkCudaErrors(cudaMemset(ltc->distance, 0, sizeof(double)*cdo->n*Q));
      checkCudaErrors(cudaMemset(ltc->bblist,  -1, sizeof(int   )*(cdo->n*Q + 1)));

      dim3 dim_grid, dim_block;
      dim_block.x = thread;
      dim_grid.x = cdo->n*Q/thread + MIN(cdo->n*Q%thread, 1);

      SetDistanceFromWall_GPU<<< dim_grid, dim_block >>>
      (ltc->bcD, ltc->bblist, ltc->distance, cdo->dx, cdo->nx, cdo->ny, cdo->n*Q);
      ltc->num_bb = NumCountPositive_Thrust<int>(ltc->bblist, cdo->n*Q);
      printf("num bb : %d\n", ltc->num_bb);
      is_firststep = false;
    }

    lbm_CapsuleCHANNELFLOW <<< dimG,dimB >>>
    (prc->iter,cdo->tau,ltc->fnD,ltc->fmD,
     ltc->dnD,ltc->unD,ltc->vnD,ltc->wnD,
     ltc->dmD,ltc->umD,ltc->vmD,ltc->wmD,
//     ltc->fxD,ltc->fyD,ltc->fzD,ltc->vfD,ltc->vfD2,
     ltc->fxD,ltc->fyD,ltc->fzD,ltc->vfD,
     ltc->bcD,cdo->nx,cdo->ny,cdo->nz,cdo->n);
///*
// cylinder BB 
    dim3 dim_grid, dim_block;
    dim_block.x = thread;
    dim_grid.x = ltc->num_bb/thread + MIN(ltc->num_bb%thread, 1);

    lbm_BounceBack_GPU<<< dim_grid, dim_block >>>
    (ltc->fmD, ltc->bcD, ltc->distance, ltc->bblist, cdo->nx, cdo->ny, cdo->nz, ltc->num_bb);
// cylinder BB end
//*/
    lbm_CHANNELFLOW_Boundary <<< dimG,dimB >>>
    (ltc->dnD,ltc->unD,ltc->vnD,ltc->wnD,
     ltc->dmD,ltc->umD,ltc->vmD,ltc->wmD,
     ltc->bcD,cdo->nx,cdo->ny,cdo->nz,cdo->n);

    if(prc->iter%M == 0){
        lbm_AVERAGE <<< dimG,dimB >>>
        (ltc->umD,ltc->vmD,ltc->wmD,cdo->n);
    }
    cudaThreadSynchronize();
    tmp  = ltc->fnD; ltc->fnD = ltc->fmD; ltc->fmD = tmp;
#endif
    return;
}

