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
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "def.h"

#ifdef KARMAN
__global__ void  lbm_KARMAN
//==========================================================
//
//  LATTICE BOLTZMANN METHOD ON GPU
//
//
(
    int       iter,
    double    tau,
    double    *fn,
    double    *fm,
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
    int       i, j,
              i___, ip__, im__, i_p_, i_m_, i__p, i__m,
              ipp_, imm_, imp_, ipm_,
              i_pp, i_mm, i_mp, i_pm,
              ip_p, im_m, ip_m, im_p;
    double    td, tu, tv, tw, uvw, Cr2, om, om1,
              f___, fp__, fm__, f_p_, f_m_, f__p, f__m,
              fpp_, fmm_, fmp_, fpm_,
              f_pp, f_mm, f_mp, f_pm,
              fp_p, fm_m, fp_m, fm_p;


    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= n) return;

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

// Domain connectivity ---
    if(bc[i] == COMP){
        i___ = i;
        if(bc[i+1]        == WALL) ip__ = -1; else ip__ = i+1;
        if(bc[i-1]        == WALL) im__ = -1; else im__ = i-1;
        if(bc[i+nx]       == WALL) i_p_ = -1; else i_p_ = i+nx;
        if(bc[i-nx]       == WALL) i_m_ = -1; else i_m_ = i-nx;
        if(bc[i+nx*ny]    == WALL) i__p = -1; else i__p = i+nx*ny;
        if(bc[i-nx*ny]    == WALL) i__m = -1; else i__m = i-nx*ny;
        if(bc[i+1+nx]     == WALL) ipp_ = -1; else ipp_ = i+1+nx;
        if(bc[i-1-nx]     == WALL) imm_ = -1; else imm_ = i-1-nx;
        if(bc[i-1+nx]     == WALL) imp_ = -1; else imp_ = i-1+nx;
        if(bc[i+1-nx]     == WALL) ipm_ = -1; else ipm_ = i+1-nx;
        if(bc[i+nx+nx*ny] == WALL) i_pp = -1; else i_pp = i+nx+nx*ny;
        if(bc[i-nx-nx*ny] == WALL) i_mm = -1; else i_mm = i-nx-nx*ny;
        if(bc[i-nx+nx*ny] == WALL) i_mp = -1; else i_mp = i-nx+nx*ny;
        if(bc[i+nx-nx*ny] == WALL) i_pm = -1; else i_pm = i+nx-nx*ny;
        if(bc[i+1+nx*ny]  == WALL) ip_p = -1; else ip_p = i+1+nx*ny;
        if(bc[i-1-nx*ny]  == WALL) im_m = -1; else im_m = i-1-nx*ny;
        if(bc[i+1-nx*ny]  == WALL) ip_m = -1; else ip_m = i+1-nx*ny;
        if(bc[i-1+nx*ny]  == WALL) im_p = -1; else im_p = i-1+nx*ny;
    }
    else if(bc[i] == INLET){
        i___ = i;
        ip__ = i+1;
        im__ = i-1;
        i_p_ = i+nx;
        i_m_ = i-nx;
        i__p = i+nx*ny;
        i__m = -1;
        ipp_ = i+1+nx;
        imm_ = i-1-nx;
        imp_ = i-1+nx;
        ipm_ = i+1-nx;
        i_pp = i+nx+nx*ny;
        i_mm = -1;
        i_mp = i-nx+nx*ny;
        i_pm = -1;
        ip_p = i+1+nx*ny;
        im_m = -1;
        ip_m = -1;
        im_p = i-1+nx*ny;
    }
    else if(bc[i] == OUTLET){
        i___ = i;
        ip__ = i+1;
        im__ = i-1;
        i_p_ = i+nx;
        i_m_ = i-nx;
        i__p = -1;
        i__m = i-nx*ny;
        ipp_ = i+1+nx;
        imm_ = i-1-nx;
        imp_ = i-1+nx;
        ipm_ = i+1-nx;
        i_pp = -1;
        i_mm = i-nx-nx*ny;
        i_mp = -1;
        i_pm = i+nx-nx*ny;
        ip_p = -1;
        im_m = i-1-nx*ny;
        ip_m = i+1-nx*ny;
        im_p = -1;
    }
    else if(bc[i] == WALL){
        i___ =  i;
        ip__ = -1;
        im__ = -1;
        i_p_ = -1;
        i_m_ = -1;
        i__p = -1;
        i__m = -1;
        ipp_ = -1;
        imm_ = -1;
        imp_ = -1;
        ipm_ = -1;
        i_pp = -1;
        i_mm = -1;
        i_mp = -1;
        i_pm = -1;
        ip_p = -1;
        im_m = -1;
        ip_m = -1;
        im_p = -1;
    }
    else return;

// Density and Velocity ---
    if(bc[i] == COMP){
        td = f___
           + fp__ + fm__ + f_p_ + f_m_ + f__p + f__m
           + fpp_ + fmm_ + fmp_ + fpm_
           + f_pp + f_mm + f_mp + f_pm
           + fp_p + fm_m + fp_m + fm_p;
        tu = (fp__ - fm__ + fpp_ - fmm_ - fmp_ + fpm_ + fp_p - fm_m + fp_m - fm_p)/(td*Cr);
        tv = (f_p_ - f_m_ + fpp_ - fmm_ + fmp_ - fpm_ + f_pp - f_mm - f_mp + f_pm)/(td*Cr);
        tw = (f__p - f__m + f_pp - f_mm + f_mp - f_pm + fp_p - fm_m - fp_m + fm_p)/(td*Cr);
    }
    else if(bc[i] == INLET){
        td = 1.0;
        tu = 0.0;
        tv = 0.0;
        tw = 1.0;
    }
    else if(bc[i] == OUTLET){
//      td = dn[i__m];
        td = 1.0;
        tu = un[i__m];
        tv = vn[i__m];
        tw = wn[i__m];
    }
    else if(bc[i] == WALL){
        td = 1.0;
        tu = 0.0;
        tv = 0.0;
        tw = 0.0;
    }
    else return;

    dm[i] = td;
    um[i] = tu;
    vm[i] = tv;
    wm[i] = tw;

// Collision step ---
    if     (bc[i] == COMP  ){ om  = 1.0/tau; om1 = 1.0 - om; }
    else if(bc[i] == INLET ){ om  = 1.0;     om1 = 0.0;      }
    else if(bc[i] == OUTLET){ om  = 1.0;     om1 = 0.0;      }
    else if(bc[i] == WALL  ){ om  = 0.0;     om1 = 0.0;      }

    Cr2 = Cr*Cr;
    uvw = -1.5*(tu*tu + tv*tv + tw*tw)*Cr2;

    f___ = om1*f___ + om*td/3.0 *(1.0                                                  + uvw); // ___
    fp__ = om1*fp__ + om*td/18.0*(1.0 + 3.0*tu*Cr        + 4.5*tu*tu*Cr2               + uvw); // p__
    fm__ = om1*fm__ + om*td/18.0*(1.0 - 3.0*tu*Cr        + 4.5*tu*tu*Cr2               + uvw); // m__
    f_p_ = om1*f_p_ + om*td/18.0*(1.0 + 3.0*tv*Cr        + 4.5*tv*tv*Cr2               + uvw); // _p_
    f_m_ = om1*f_m_ + om*td/18.0*(1.0 - 3.0*tv*Cr        + 4.5*tv*tv*Cr2               + uvw); // _m_
    f__p = om1*f__p + om*td/18.0*(1.0 + 3.0*tw*Cr        + 4.5*tw*tw*Cr2               + uvw); // __p
    f__m = om1*f__m + om*td/18.0*(1.0 - 3.0*tw*Cr        + 4.5*tw*tw*Cr2               + uvw); // __m
    fpp_ = om1*fpp_ + om*td/36.0*(1.0 + 3.0*(tu + tv)*Cr + 4.5*(tu + tv)*(tu + tv)*Cr2 + uvw); // pp_
    fmm_ = om1*fmm_ + om*td/36.0*(1.0 - 3.0*(tu + tv)*Cr + 4.5*(tu + tv)*(tu + tv)*Cr2 + uvw); // mm_
    fmp_ = om1*fmp_ + om*td/36.0*(1.0 - 3.0*(tu - tv)*Cr + 4.5*(tu - tv)*(tu - tv)*Cr2 + uvw); // mp_
    fpm_ = om1*fpm_ + om*td/36.0*(1.0 + 3.0*(tu - tv)*Cr + 4.5*(tu - tv)*(tu - tv)*Cr2 + uvw); // pm_
    f_pp = om1*f_pp + om*td/36.0*(1.0 + 3.0*(tv + tw)*Cr + 4.5*(tv + tw)*(tv + tw)*Cr2 + uvw); // _pp
    f_mm = om1*f_mm + om*td/36.0*(1.0 - 3.0*(tv + tw)*Cr + 4.5*(tv + tw)*(tv + tw)*Cr2 + uvw); // _mm
    f_mp = om1*f_mp + om*td/36.0*(1.0 - 3.0*(tv - tw)*Cr + 4.5*(tv - tw)*(tv - tw)*Cr2 + uvw); // _mp
    f_pm = om1*f_pm + om*td/36.0*(1.0 + 3.0*(tv - tw)*Cr + 4.5*(tv - tw)*(tv - tw)*Cr2 + uvw); // _pm
    fp_p = om1*fp_p + om*td/36.0*(1.0 + 3.0*(tw + tu)*Cr + 4.5*(tw + tu)*(tw + tu)*Cr2 + uvw); // p_p
    fm_m = om1*fm_m + om*td/36.0*(1.0 - 3.0*(tw + tu)*Cr + 4.5*(tw + tu)*(tw + tu)*Cr2 + uvw); // m_m
    fp_m = om1*fp_m + om*td/36.0*(1.0 - 3.0*(tw - tu)*Cr + 4.5*(tw - tu)*(tw - tu)*Cr2 + uvw); // p_m
    fm_p = om1*fm_p + om*td/36.0*(1.0 + 3.0*(tw - tu)*Cr + 4.5*(tw - tu)*(tw - tu)*Cr2 + uvw); // m_p

// Streaming step & Bounce back ---
    if(i___ >= 0) j = i___*Q + 0; else j = i*Q + 0; // ___
    fm[j] = f___;
    if(ip__ >= 0) j = ip__*Q + 1; else j = i*Q + 2; // p__
    fm[j] = fp__;
    if(im__ >= 0) j = im__*Q + 2; else j = i*Q + 1; // m__
    fm[j] = fm__;
    if(i_p_ >= 0) j = i_p_*Q + 3; else j = i*Q + 4; // _p_
    fm[j] = f_p_;
    if(i_m_ >= 0) j = i_m_*Q + 4; else j = i*Q + 3; // _m_
    fm[j] = f_m_;
    if(i__p >= 0) j = i__p*Q + 5; else j = i*Q + 6; // __p
    fm[j] = f__p;
    if(i__m >= 0) j = i__m*Q + 6; else j = i*Q + 5; // __m
    fm[j] = f__m;
    if(ipp_ >= 0) j = ipp_*Q + 7; else j = i*Q + 8; // pp_
    fm[j] = fpp_;
    if(imm_ >= 0) j = imm_*Q + 8; else j = i*Q + 7; // mm_
    fm[j] = fmm_;
    if(imp_ >= 0) j = imp_*Q + 9; else j = i*Q +10; // mp_
    fm[j] = fmp_;
    if(ipm_ >= 0) j = ipm_*Q +10; else j = i*Q + 9; // pm_
    fm[j] = fpm_;
    if(i_pp >= 0) j = i_pp*Q +11; else j = i*Q +12; // _pp
    fm[j] = f_pp;
    if(i_mm >= 0) j = i_mm*Q +12; else j = i*Q +11; // _mm
    fm[j] = f_mm;
    if(i_mp >= 0) j = i_mp*Q +13; else j = i*Q +14; // _mp
    fm[j] = f_mp;
    if(i_pm >= 0) j = i_pm*Q +14; else j = i*Q +13; // _pm
    fm[j] = f_pm;
    if(ip_p >= 0) j = ip_p*Q +15; else j = i*Q +16; // p_p
    fm[j] = fp_p;
    if(im_m >= 0) j = im_m*Q +16; else j = i*Q +15; // m_m
    fm[j] = fm_m;
    if(ip_m >= 0) j = ip_m*Q +17; else j = i*Q +18; // p_m
    fm[j] = fp_m;
    if(im_p >= 0) j = im_p*Q +18; else j = i*Q +17; // m_p
    fm[j] = fm_p;
}
#endif // KARMAN

#ifdef SHEARFLOW
__global__ void  lbm_SHEARFLOW
//==========================================================
//
//  LATTICE BOLTZMANN METHOD ON GPU
//
//
(
    int       iter,
    double    tau,
    double    *fn,
    double    *fm,
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
    int       i, ix, iy, iz, j,
              i___, ip__, im__, i_p_, i_m_, i__p, i__m,
              ipp_, imm_, imp_, ipm_,
              i_pp, i_mm, i_mp, i_pm,
              ip_p, im_m, ip_m, im_p;
    double    td, tu, tv, tw, uvw, Cr2, om, om1,
              f___, fp__, fm__, f_p_, f_m_, f__p, f__m,
              fpp_, fmm_, fmp_, fpm_,
              f_pp, f_mm, f_mp, f_pm,
              fp_p, fm_m, fp_m, fm_p;
    double    ub = 1.0;


    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= n)        return;
    if(bc[i] != COMP) return;

    iz = (int)(i/(nx*ny));
    iy = (int)(i/nx) - ny*iz;
    ix = i - nx*iy - nx*ny*iz;

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

// Domain connectivity ---
    i___ = i;
    if     (ix == nx-3)               ip__ = i-(nx-5);
    else                              ip__ = i+1;
    if     (ix == 2)                  im__ = i+(nx-5);
    else                              im__ = i-1;
    if     (iy == ny-3)               i_p_ = -1;
    else                              i_p_ = i+nx;
    if     (iy == 2)                  i_m_ = -1;
    else                              i_m_ = i-nx;
    if     (iz == nz-3)               i__p = i-nx*ny*(nz-5);
    else                              i__p = i+nx*ny;
    if     (iz == 2)                  i__m = i+nx*ny*(nz-5);
    else                              i__m = i-nx*ny;
    if     (iy == ny-3)               ipp_ = -1;
    else if(ix == nx-3)               ipp_ = i-(nx-5)+nx;
    else                              ipp_ = i+1+nx;
    if     (iy == 2)                  imm_ = -1;
    else if(ix == 2)                  imm_ = i+(nx-5)-nx;
    else                              imm_ = i-1-nx;
    if     (iy == ny-3)               imp_ = -1;
    else if(ix == 2)                  imp_ = i+(nx-5)+nx;
    else                              imp_ = i-1+nx;
    if     (iy == 2)                  ipm_ = -1;
    else if(ix == nx-3)               ipm_ = i-(nx-5)-nx;
    else                              ipm_ = i+1-nx;
    if     (iy == ny-3)               i_pp = -1;
    else if(iz == nz-3)               i_pp = i+nx-nx*ny*(nz-5);
    else                              i_pp = i+nx+nx*ny;
    if     (iy == 2)                  i_mm = -1;
    else if(iz == 2)                  i_mm = i-nx+nx*ny*(nz-5);
    else                              i_mm = i-nx-nx*ny;
    if     (iy == 2)                  i_mp = -1;
    else if(iz == nz-3)               i_mp = i-nx-nx*ny*(nz-5);
    else                              i_mp = i-nx+nx*ny;
    if     (iy == ny-3)               i_pm = -1;
    else if(iz == 2)                  i_pm = i+nx+nx*ny*(nz-5);
    else                              i_pm = i+nx-nx*ny;
    if     (ix == nx-3 && iz == nz-3) ip_p = i-(nx-5)-nx*ny*(nz-5);
    else if(ix == nx-3)               ip_p = i-(nx-5)+nx*ny;
    else if(iz == nz-3)               ip_p = i+1-nx*ny*(nz-5);
    else                              ip_p = i+1+nx*ny;
    if     (ix == 2 && iz == 2)       im_m = i+(nx-5)+nx*ny*(nz-5);
    else if(ix == 2)                  im_m = i+(nx-5)-nx*ny;
    else if(iz == 2)                  im_m = i-1+nx*ny*(nz-5);
    else                              im_m = i-1-nx*ny;
    if     (ix == nx-3 && iz == 2)    ip_m = i-(nx-5)+nx*ny*(nz-5);
    else if(ix == nx-3)               ip_m = i-(nx-5)-nx*ny;
    else if(iz == 2)                  ip_m = i+1+nx*ny*(nz-5);
    else                              ip_m = i+1-nx*ny;
    if     (ix == 2 && iz == nz-3)    im_p = i+(nx-5)-nx*ny*(nz-5);
    else if(ix == 2)                  im_p = i+(nx-5)+nx*ny;
    else if(iz == nz-3)               im_p = i-1-nx*ny*(nz-5);
    else                              im_p = i-1+nx*ny;

// Density and Velocity ---
    td = f___
       + fp__ + fm__ + f_p_ + f_m_ + f__p + f__m
       + fpp_ + fmm_ + fmp_ + fpm_
       + f_pp + f_mm + f_mp + f_pm
       + fp_p + fm_m + fp_m + fm_p;
    tu = (fp__ - fm__ + fpp_ - fmm_ - fmp_ + fpm_ + fp_p - fm_m + fp_m - fm_p)/(td*Cr);
    tv = (f_p_ - f_m_ + fpp_ - fmm_ + fmp_ - fpm_ + f_pp - f_mm - f_mp + f_pm)/(td*Cr);
    tw = (f__p - f__m + f_pp - f_mm + f_mp - f_pm + fp_p - fm_m - fp_m + fm_p)/(td*Cr);

    dn[i] = td;
    un[i] = tu;
    vn[i] = tv;
    wn[i] = tw;

// Collision step ---
    om  = 1.0/tau;
    om1 = 1.0 - om;
    Cr2 = Cr*Cr;
    uvw = -1.5*(tu*tu + tv*tv + tw*tw)*Cr2;

    f___ = om1*f___ + om*td/3.0 *(1.0                                                  + uvw); // ___
    fp__ = om1*fp__ + om*td/18.0*(1.0 + 3.0*tu*Cr        + 4.5*tu*tu*Cr2               + uvw); // p__
    fm__ = om1*fm__ + om*td/18.0*(1.0 - 3.0*tu*Cr        + 4.5*tu*tu*Cr2               + uvw); // m__
    f_p_ = om1*f_p_ + om*td/18.0*(1.0 + 3.0*tv*Cr        + 4.5*tv*tv*Cr2               + uvw); // _p_
    f_m_ = om1*f_m_ + om*td/18.0*(1.0 - 3.0*tv*Cr        + 4.5*tv*tv*Cr2               + uvw); // _m_
    f__p = om1*f__p + om*td/18.0*(1.0 + 3.0*tw*Cr        + 4.5*tw*tw*Cr2               + uvw); // __p
    f__m = om1*f__m + om*td/18.0*(1.0 - 3.0*tw*Cr        + 4.5*tw*tw*Cr2               + uvw); // __m
    fpp_ = om1*fpp_ + om*td/36.0*(1.0 + 3.0*(tu + tv)*Cr + 4.5*(tu + tv)*(tu + tv)*Cr2 + uvw); // pp_
    fmm_ = om1*fmm_ + om*td/36.0*(1.0 - 3.0*(tu + tv)*Cr + 4.5*(tu + tv)*(tu + tv)*Cr2 + uvw); // mm_
    fmp_ = om1*fmp_ + om*td/36.0*(1.0 - 3.0*(tu - tv)*Cr + 4.5*(tu - tv)*(tu - tv)*Cr2 + uvw); // mp_
    fpm_ = om1*fpm_ + om*td/36.0*(1.0 + 3.0*(tu - tv)*Cr + 4.5*(tu - tv)*(tu - tv)*Cr2 + uvw); // pm_
    f_pp = om1*f_pp + om*td/36.0*(1.0 + 3.0*(tv + tw)*Cr + 4.5*(tv + tw)*(tv + tw)*Cr2 + uvw); // _pp
    f_mm = om1*f_mm + om*td/36.0*(1.0 - 3.0*(tv + tw)*Cr + 4.5*(tv + tw)*(tv + tw)*Cr2 + uvw); // _mm
    f_mp = om1*f_mp + om*td/36.0*(1.0 - 3.0*(tv - tw)*Cr + 4.5*(tv - tw)*(tv - tw)*Cr2 + uvw); // _mp
    f_pm = om1*f_pm + om*td/36.0*(1.0 + 3.0*(tv - tw)*Cr + 4.5*(tv - tw)*(tv - tw)*Cr2 + uvw); // _pm
    fp_p = om1*fp_p + om*td/36.0*(1.0 + 3.0*(tw + tu)*Cr + 4.5*(tw + tu)*(tw + tu)*Cr2 + uvw); // p_p
    fm_m = om1*fm_m + om*td/36.0*(1.0 - 3.0*(tw + tu)*Cr + 4.5*(tw + tu)*(tw + tu)*Cr2 + uvw); // m_m
    fp_m = om1*fp_m + om*td/36.0*(1.0 - 3.0*(tw - tu)*Cr + 4.5*(tw - tu)*(tw - tu)*Cr2 + uvw); // p_m
    fm_p = om1*fm_p + om*td/36.0*(1.0 + 3.0*(tw - tu)*Cr + 4.5*(tw - tu)*(tw - tu)*Cr2 + uvw); // m_p

// Streaming step & Bounce back ---
    if(i___ >= 0) j = i___*Q + 0; else j = i*Q + 0;                    // ___
    fm[j] = f___;
    if(ip__ >= 0) j = ip__*Q + 1; else j = i*Q + 2;                    // p__
    fm[j] = fp__;
    if(im__ >= 0) j = im__*Q + 2; else j = i*Q + 1;                    // m__
    fm[j] = fm__;
    if(i_p_ >= 0) j = i_p_*Q + 3; else j = i*Q + 4;                    // _p_
    fm[j] = f_p_;
    if(i_m_ >= 0) j = i_m_*Q + 4; else j = i*Q + 3;                    // _m_
    fm[j] = f_m_;
    if(i__p >= 0) j = i__p*Q + 5; else j = i*Q + 6;                    // __p
    fm[j] = f__p;
    if(i__m >= 0) j = i__m*Q + 6; else j = i*Q + 5;                    // __m
    fm[j] = f__m;
    if(ipp_ >= 0) j = ipp_*Q + 7; else j = i*Q + 8;                    // pp_
    fm[j] = fpp_;
    if(imm_ >= 0) j = imm_*Q + 8; else j = i*Q + 7;                    // mm_
    fm[j] = fmm_;
    if(imp_ >= 0) j = imp_*Q + 9; else j = i*Q +10;                    // mp_
    fm[j] = fmp_;
    if(ipm_ >= 0) j = ipm_*Q +10; else j = i*Q + 9;                    // pm_
    fm[j] = fpm_;
    if(i_pp >= 0){ j = i_pp*Q +11; fm[j] = f_pp;                     } // _pp
    else         { j = i   *Q +12; fm[j] = f_pp - 6.0*td*Cr/36.0*ub; }
    if(i_mm >= 0){ j = i_mm*Q +12; fm[j] = f_mm;                     } // _mm
    else         { j = i   *Q +11; fm[j] = f_mm - 6.0*td*Cr/36.0*ub; }
    if(i_mp >= 0){ j = i_mp*Q +13; fm[j] = f_mp;                     } // _mp
    else         { j = i   *Q +14; fm[j] = f_mp + 6.0*td*Cr/36.0*ub; }
    if(i_pm >= 0){ j = i_pm*Q +14; fm[j] = f_pm;                     } // _pm
    else         { j = i   *Q +13; fm[j] = f_pm + 6.0*td*Cr/36.0*ub; }
    if(ip_p >= 0) j = ip_p*Q +15; else j = i*Q +16;                    // p_p
    fm[j] = fp_p;
    if(im_m >= 0) j = im_m*Q +16; else j = i*Q +15;                    // m_m
    fm[j] = fm_m;
    if(ip_m >= 0) j = ip_m*Q +17; else j = i*Q +18;                    // p_m
    fm[j] = fp_m;
    if(im_p >= 0) j = im_p*Q +18; else j = i*Q +17;                    // m_p
    fm[j] = fm_p;
}
#endif // SHEARFLOW

#ifdef CHANNELFLOW
__global__ void  lbm_CHANNELFLOW
//==========================================================
//
//  LATTICE BOLTZMANN METHOD ON GPU
//
//
(
    int       iter,
    double    tau,
    double    *fn,
    double    *fm,
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
    int       i, j,
              i___, ip__, im__, i_p_, i_m_, i__p, i__m,
              ipp_, imm_, imp_, ipm_,
              i_pp, i_mm, i_mp, i_pm,
              ip_p, im_m, ip_m, im_p;
    double    td, tu, tv, tw, uvw, Cr2, om, om1, om2,
              f___, fp__, fm__, f_p_, f_m_, f__p, f__m,
              fpp_, fmm_, fmp_, fpm_,
              f_pp, f_mm, f_mp, f_pm,
              fp_p, fm_m, fp_m, fm_p;
    double    ui = 0.0, vi = 0.0, wi = 1.0;


    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= n)        return;
    if(bc[i] != COMP) return;

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

// Domain connectivity ---
    i___ = i;
    if     (bc[i+1]        == WALL) ip__ = -1;
    else                            ip__ = i+1;
    if     (bc[i-1]        == WALL) im__ = -1;
    else                            im__ = i-1;
    if     (bc[i+nx]       == WALL) i_p_ = -1;
    else                            i_p_ = i+nx;
    if     (bc[i-nx]       == WALL) i_m_ = -1;
    else                            i_m_ = i-nx;
    if     (bc[i+nx*ny]    == BUFF) i__p = i-nx*ny*(nz-5);
    else                            i__p = i+nx*ny;
    if     (bc[i-nx*ny]    == BUFF) i__m = i+nx*ny*(nz-5);
    else                            i__m = i-nx*ny;
    if     (bc[i+1+nx]     == WALL) ipp_ = -1;
    else                            ipp_ = i+1+nx;
    if     (bc[i-1-nx]     == WALL) imm_ = -1;
    else                            imm_ = i-1-nx;
    if     (bc[i-1+nx]     == WALL) imp_ = -1;
    else                            imp_ = i-1+nx;
    if     (bc[i+1-nx]     == WALL) ipm_ = -1;
    else                            ipm_ = i+1-nx;
    if     (bc[i+nx+nx*ny] == WALL) i_pp = -1;
    else if(bc[i+nx+nx*ny] == BUFF) i_pp = i+nx-nx*ny*(nz-5);
    else                            i_pp = i+nx+nx*ny;
    if     (bc[i-nx-nx*ny] == WALL) i_mm = -1;
    else if(bc[i-nx-nx*ny] == BUFF) i_mm = i-nx+nx*ny*(nz-5);
    else                            i_mm = i-nx-nx*ny;
    if     (bc[i-nx+nx*ny] == WALL) i_mp = -1;
    else if(bc[i-nx+nx*ny] == BUFF) i_mp = i-nx-nx*ny*(nz-5);
    else                            i_mp = i-nx+nx*ny;
    if     (bc[i+nx-nx*ny] == WALL) i_pm = -1;
    else if(bc[i+nx-nx*ny] == BUFF) i_pm = i+nx+nx*ny*(nz-5);
    else                            i_pm = i+nx-nx*ny;
    if     (bc[i+1+nx*ny]  == WALL) ip_p = -1;
    else if(bc[i+1+nx*ny]  == BUFF) ip_p = i+1-nx*ny*(nz-5);
    else                            ip_p = i+1+nx*ny;
    if     (bc[i-1-nx*ny]  == WALL) im_m = -1;
    else if(bc[i-1-nx*ny]  == BUFF) im_m = i-1+nx*ny*(nz-5);
    else                            im_m = i-1-nx*ny;
    if     (bc[i+1-nx*ny]  == WALL) ip_m = -1;
    else if(bc[i+1-nx*ny]  == BUFF) ip_m = i+1+nx*ny*(nz-5);
    else                            ip_m = i+1-nx*ny;
    if     (bc[i-1+nx*ny]  == WALL) im_p = -1;
    else if(bc[i-1+nx*ny]  == BUFF) im_p = i-1-nx*ny*(nz-5);
    else                            im_p = i-1+nx*ny;

// Density and Velocity ---
    td  = f___
        + fp__ + fm__ + f_p_ + f_m_ + f__p + f__m
        + fpp_ + fmm_ + fmp_ + fpm_
        + f_pp + f_mm + f_mp + f_pm
        + fp_p + fm_m + fp_m + fm_p;
    tu  = (fp__ - fm__ + fpp_ - fmm_ - fmp_ + fpm_ + fp_p - fm_m + fp_m - fm_p + td*P*ui)/(td*Cr);
    tv  = (f_p_ - f_m_ + fpp_ - fmm_ + fmp_ - fpm_ + f_pp - f_mm - f_mp + f_pm + td*P*vi)/(td*Cr);
    tw  = (f__p - f__m + f_pp - f_mm + f_mp - f_pm + fp_p - fm_m - fp_m + fm_p + td*P*wi)/(td*Cr);

    dn[i]  = td;
    un[i]  = tu;
    vn[i]  = tv;
    wn[i]  = tw;

// Collision step ---
    om  = 1.0/tau;
    om1 = 1.0 - om;
    om2 = (1.0 - om/2.0)*2.0*td*P;
    Cr2 = Cr*Cr;
    uvw = -1.5*(tu*tu + tv*tv + tw*tw)*Cr2;

    f___ = om1*f___ + om*td/3.0 *(1.0                                                  + uvw)  // ___
         + om2/3.0*(3.0*-tu*Cr*ui + 3.0*-tv*Cr*vi + 3.0*-tw*Cr*wi);

    fp__ = om1*fp__ + om*td/18.0*(1.0 + 3.0*tu*Cr        + 4.5*tu*tu*Cr2               + uvw)  // p__
         + om2/18.0*((3.0*( 1.0-tu*Cr)+9.0*Cr*tu)*ui + 3.0*-tv*Cr*vi + 3.0*-tw*Cr*wi);

    fm__ = om1*fm__ + om*td/18.0*(1.0 - 3.0*tu*Cr        + 4.5*tu*tu*Cr2               + uvw)  // m__
         + om2/18.0*((3.0*(-1.0-tu*Cr)+9.0*Cr*tu)*ui + 3.0*-tv*Cr*vi + 3.0*-tw*Cr*wi);

    f_p_ = om1*f_p_ + om*td/18.0*(1.0 + 3.0*tv*Cr        + 4.5*tv*tv*Cr2               + uvw)  // _p_
         + om2/18.0*(3.0*-tu*Cr*ui + (3.0*( 1.0-tv*Cr)+9.0*Cr*tv)*vi + 3.0*-tw*Cr*wi);

    f_m_ = om1*f_m_ + om*td/18.0*(1.0 - 3.0*tv*Cr        + 4.5*tv*tv*Cr2               + uvw)  // _m_
         + om2/18.0*(3.0*-tu*Cr*ui + (3.0*(-1.0-tv*Cr)+9.0*Cr*tv)*vi + 3.0*-tw*Cr*wi);

    f__p = om1*f__p + om*td/18.0*(1.0 + 3.0*tw*Cr        + 4.5*tw*tw*Cr2               + uvw)  // __p
         + om2/18.0*(3.0*-tu*Cr*ui + 3.0*-tv*Cr*vi + (3.0*( 1.0-tw*Cr)+9.0*Cr*tw)*wi);

    f__m = om1*f__m + om*td/18.0*(1.0 - 3.0*tw*Cr        + 4.5*tw*tw*Cr2               + uvw)  // __m
         + om2/18.0*(3.0*-tu*Cr*ui + 3.0*-tv*Cr*vi + (3.0*(-1.0-tw*Cr)+9.0*Cr*tw)*wi);

    fpp_ = om1*fpp_ + om*td/36.0*(1.0 + 3.0*(tu + tv)*Cr + 4.5*(tu + tv)*(tu + tv)*Cr2 + uvw)  // pp_
         + om2/36.0*((3.0*( 1.0-tu*Cr)+9.0*Cr*(tu+tv))*ui + (3.0*( 1.0-tv*Cr)+9.0*Cr*(tu+tv))*vi + 3.0*-tw*Cr*wi);

    fmm_ = om1*fmm_ + om*td/36.0*(1.0 - 3.0*(tu + tv)*Cr + 4.5*(tu + tv)*(tu + tv)*Cr2 + uvw)  // mm_
         + om2/36.0*((3.0*(-1.0-tu*Cr)+9.0*Cr*(tu+tv))*ui + (3.0*(-1.0-tv*Cr)+9.0*Cr*(tu+tv))*vi + 3.0*-tw*Cr*wi);

    fmp_ = om1*fmp_ + om*td/36.0*(1.0 - 3.0*(tu - tv)*Cr + 4.5*(tu - tv)*(tu - tv)*Cr2 + uvw)  // mp_
         + om2/36.0*((3.0*(-1.0-tu*Cr)+9.0*Cr*(tu-tv))*ui + (3.0*( 1.0-tv*Cr)-9.0*Cr*(tu-tv))*vi + 3.0*-tw*Cr*wi);

    fpm_ = om1*fpm_ + om*td/36.0*(1.0 + 3.0*(tu - tv)*Cr + 4.5*(tu - tv)*(tu - tv)*Cr2 + uvw)  // pm_
         + om2/36.0*((3.0*( 1.0-tu*Cr)+9.0*Cr*(tu-tv))*ui + (3.0*(-1.0-tv*Cr)-9.0*Cr*(tu-tv))*vi + 3.0*-tw*Cr*wi);

    f_pp = om1*f_pp + om*td/36.0*(1.0 + 3.0*(tv + tw)*Cr + 4.5*(tv + tw)*(tv + tw)*Cr2 + uvw)  // _pp
         + om2/36.0*(3.0*-tu*Cr*ui + (3.0*( 1.0-tv*Cr)+9.0*Cr*(tv+tw))*vi + (3.0*( 1.0-tw*Cr)+9.0*Cr*(tv+tw))*wi);

    f_mm = om1*f_mm + om*td/36.0*(1.0 - 3.0*(tv + tw)*Cr + 4.5*(tv + tw)*(tv + tw)*Cr2 + uvw)  // _mm
         + om2/36.0*(3.0*-tu*Cr*ui + (3.0*(-1.0-tv*Cr)+9.0*Cr*(tv+tw))*vi + (3.0*(-1.0-tw*Cr)+9.0*Cr*(tv+tw))*wi);

    f_mp = om1*f_mp + om*td/36.0*(1.0 - 3.0*(tv - tw)*Cr + 4.5*(tv - tw)*(tv - tw)*Cr2 + uvw)  // _mp
         + om2/36.0*(3.0*-tu*Cr*ui + (3.0*(-1.0-tv*Cr)+9.0*Cr*(tv-tw))*vi + (3.0*( 1.0-tw*Cr)-9.0*Cr*(tv-tw))*wi);

    f_pm = om1*f_pm + om*td/36.0*(1.0 + 3.0*(tv - tw)*Cr + 4.5*(tv - tw)*(tv - tw)*Cr2 + uvw)  // _pm
         + om2/36.0*(3.0*-tu*Cr*ui + (3.0*( 1.0-tv*Cr)+9.0*Cr*(tv-tw))*vi + (3.0*(-1.0-tw*Cr)-9.0*Cr*(tv-tw))*wi);

    fp_p = om1*fp_p + om*td/36.0*(1.0 + 3.0*(tw + tu)*Cr + 4.5*(tw + tu)*(tw + tu)*Cr2 + uvw)  // p_p
         + om2/36.0*((3.0*( 1.0-tu*Cr)+9.0*Cr*(tw+tu))*ui + 3.0*-tv*Cr*vi + (3.0*( 1.0-tw*Cr)+9.0*Cr*(tw+tu))*wi);

    fm_m = om1*fm_m + om*td/36.0*(1.0 - 3.0*(tw + tu)*Cr + 4.5*(tw + tu)*(tw + tu)*Cr2 + uvw)  // m_m
         + om2/36.0*((3.0*(-1.0-tu*Cr)+9.0*Cr*(tw+tu))*ui + 3.0*-tv*Cr*vi + (3.0*(-1.0-tw*Cr)+9.0*Cr*(tw+tu))*wi);

    fp_m = om1*fp_m + om*td/36.0*(1.0 - 3.0*(tw - tu)*Cr + 4.5*(tw - tu)*(tw - tu)*Cr2 + uvw)  // p_m
         + om2/36.0*((3.0*( 1.0-tu*Cr)-9.0*Cr*(tw-tu))*ui + 3.0*-tv*Cr*vi + (3.0*(-1.0-tw*Cr)+9.0*Cr*(tw-tu))*wi);

    fm_p = om1*fm_p + om*td/36.0*(1.0 + 3.0*(tw - tu)*Cr + 4.5*(tw - tu)*(tw - tu)*Cr2 + uvw)  // m_p
         + om2/36.0*((3.0*(-1.0-tu*Cr)-9.0*Cr*(tw-tu))*ui + 3.0*-tv*Cr*vi + (3.0*( 1.0-tw*Cr)+9.0*Cr*(tw-tu))*wi);

// Streaming step & Bounce back ---
    if(i___ >= 0) j = i___*Q + 0; else j = i*Q + 0; // ___
    fm[j] = f___;
    if(ip__ >= 0) j = ip__*Q + 1; else j = i*Q + 2; // p__
    fm[j] = fp__;
    if(im__ >= 0) j = im__*Q + 2; else j = i*Q + 1; // m__
    fm[j] = fm__;
    if(i_p_ >= 0) j = i_p_*Q + 3; else j = i*Q + 4; // _p_
    fm[j] = f_p_;
    if(i_m_ >= 0) j = i_m_*Q + 4; else j = i*Q + 3; // _m_
    fm[j] = f_m_;
    if(i__p >= 0) j = i__p*Q + 5; else j = i*Q + 6; // __p
    fm[j] = f__p;
    if(i__m >= 0) j = i__m*Q + 6; else j = i*Q + 5; // __m
    fm[j] = f__m;
    if(ipp_ >= 0) j = ipp_*Q + 7; else j = i*Q + 8; // pp_
    fm[j] = fpp_;
    if(imm_ >= 0) j = imm_*Q + 8; else j = i*Q + 7; // mm_
    fm[j] = fmm_;
    if(imp_ >= 0) j = imp_*Q + 9; else j = i*Q +10; // mp_
    fm[j] = fmp_;
    if(ipm_ >= 0) j = ipm_*Q +10; else j = i*Q + 9; // pm_
    fm[j] = fpm_;
    if(i_pp >= 0) j = i_pp*Q +11; else j = i*Q +12; // _pp
    fm[j] = f_pp;
    if(i_mm >= 0) j = i_mm*Q +12; else j = i*Q +11; // _mm
    fm[j] = f_mm;
    if(i_mp >= 0) j = i_mp*Q +13; else j = i*Q +14; // _mp
    fm[j] = f_mp;
    if(i_pm >= 0) j = i_pm*Q +14; else j = i*Q +13; // _pm
    fm[j] = f_pm;
    if(ip_p >= 0) j = ip_p*Q +15; else j = i*Q +16; // p_p
    fm[j] = fp_p;
    if(im_m >= 0) j = im_m*Q +16; else j = i*Q +15; // m_m
    fm[j] = fm_m;
    if(ip_m >= 0) j = ip_m*Q +17; else j = i*Q +18; // p_m
    fm[j] = fp_m;
    if(im_p >= 0) j = im_p*Q +18; else j = i*Q +17; // m_p
    fm[j] = fm_p;
}
#endif // CHANNELFLOW

#ifdef CapsuleSHEARFLOW
__global__ void  lbm_CapsuleSHEARFLOW
//==========================================================
//
//  LATTICE BOLTZMANN METHOD ON GPU
//
//
(
    int       iter,
    double    tau,
    double    sr,
    double    *fn,
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
    double    *vf2,
    int       *bc,
    int       nx,
    int       ny,
    int       nz,
    int       n
)
//----------------------------------------------------------
{
    int       i, ix, iy, iz, j,
              i___, ip__, im__, i_p_, i_m_, i__p, i__m,
              ipp_, imm_, imp_, ipm_,
              i_pp, i_mm, i_mp, i_pm,
              ip_p, im_m, ip_m, im_p;
    double    td, tu, tv, tw, tfx, tfy, tfz, uvw, Cr2, om, om1, om2, /* tvf1, tvf2 */
              f___, fp__, fm__, f_p_, f_m_, f__p, f__m,
              fpp_, fmm_, fmp_, fpm_,
              f_pp, f_mm, f_mp, f_pm,
              fp_p, fm_m, fp_m, fm_p, sra;


    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= n)        return;
    if(bc[i] != COMP) return;

    if((iter) < FLOWSTART) sra = 0.0; else sra = sr;

    iz = (int)(i/(nx*ny));
    iy = (int)(i/nx) - ny*iz;
    ix = i - nx*iy - nx*ny*iz;

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

// Domain connectivity ---
    i___ = i;
    if     (ix == nx-3)               ip__ = i-(nx-5);
    else                              ip__ = i+1;
    if     (ix == 2)                  im__ = i+(nx-5);
    else                              im__ = i-1;
    if     (iy == ny-3)               i_p_ = -1;
    else                              i_p_ = i+nx;
    if     (iy == 2)                  i_m_ = -1;
    else                              i_m_ = i-nx;
    if     (iz == nz-3)               i__p = i-nx*ny*(nz-5);
    else                              i__p = i+nx*ny;
    if     (iz == 2)                  i__m = i+nx*ny*(nz-5);
    else                              i__m = i-nx*ny;
    if     (iy == ny-3)               ipp_ = -1;
    else if(ix == nx-3)               ipp_ = i-(nx-5)+nx;
    else                              ipp_ = i+1+nx;
    if     (iy == 2)                  imm_ = -1;
    else if(ix == 2)                  imm_ = i+(nx-5)-nx;
    else                              imm_ = i-1-nx;
    if     (iy == ny-3)               imp_ = -1;
    else if(ix == 2)                  imp_ = i+(nx-5)+nx;
    else                              imp_ = i-1+nx;
    if     (iy == 2)                  ipm_ = -1;
    else if(ix == nx-3)               ipm_ = i-(nx-5)-nx;
    else                              ipm_ = i+1-nx;
    if     (iy == ny-3)               i_pp = -1;
    else if(iz == nz-3)               i_pp = i+nx-nx*ny*(nz-5);
    else                              i_pp = i+nx+nx*ny;
    if     (iy == 2)                  i_mm = -1;
    else if(iz == 2)                  i_mm = i-nx+nx*ny*(nz-5);
    else                              i_mm = i-nx-nx*ny;
    if     (iy == 2)                  i_mp = -1;
    else if(iz == nz-3)               i_mp = i-nx-nx*ny*(nz-5);
    else                              i_mp = i-nx+nx*ny;
    if     (iy == ny-3)               i_pm = -1;
    else if(iz == 2)                  i_pm = i+nx+nx*ny*(nz-5);
    else                              i_pm = i+nx-nx*ny;
    if     (ix == nx-3 && iz == nz-3) ip_p = i-(nx-5)-nx*ny*(nz-5);
    else if(ix == nx-3)               ip_p = i-(nx-5)+nx*ny;
    else if(iz == nz-3)               ip_p = i+1-nx*ny*(nz-5);
    else                              ip_p = i+1+nx*ny;
    if     (ix == 2 && iz == 2)       im_m = i+(nx-5)+nx*ny*(nz-5);
    else if(ix == 2)                  im_m = i+(nx-5)-nx*ny;
    else if(iz == 2)                  im_m = i-1+nx*ny*(nz-5);
    else                              im_m = i-1-nx*ny;
    if     (ix == nx-3 && iz == 2)    ip_m = i-(nx-5)+nx*ny*(nz-5);
    else if(ix == nx-3)               ip_m = i-(nx-5)-nx*ny;
    else if(iz == 2)                  ip_m = i+1+nx*ny*(nz-5);
    else                              ip_m = i+1-nx*ny;
    if     (ix == 2 && iz == nz-3)    im_p = i+(nx-5)-nx*ny*(nz-5);
    else if(ix == 2)                  im_p = i+(nx-5)+nx*ny;
    else if(iz == nz-3)               im_p = i-1-nx*ny*(nz-5);
    else                              im_p = i-1+nx*ny;

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

    double tvf1 = vf1[i];
    //double tvf2 = vf2[i];

    td  = f___
        + fp__ + fm__ + f_p_ + f_m_ + f__p + f__m
        + fpp_ + fmm_ + fmp_ + fpm_
        + f_pp + f_mm + f_mp + f_pm
        + fp_p + fm_m + fp_m + fm_p;
    tu  = (fp__ - fm__ + fpp_ - fmm_ - fmp_ + fpm_ + fp_p - fm_m + fp_m - fm_p + td*B*tfx)/(td*Cr);
    tv  = (f_p_ - f_m_ + fpp_ - fmm_ + fmp_ - fpm_ + f_pp - f_mm - f_mp + f_pm + td*B*tfy)/(td*Cr);
    tw  = (f__p - f__m + f_pp - f_mm + f_mp - f_pm + fp_p - fm_m - fp_m + fm_p + td*B*tfz)/(td*Cr);

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
    om  = 1.0/(3.0*(1.0 + (LAMBDA1 - 1.0)*tvf1)*Df + 0.5);
//    om  = 1.0/(3.0*(1.0 + (LAMBDA1 - 1.0)*tvf1 + (LAMBDA2 - 1.0)*tvf2)*Df + 0.5);
    om1 = 1.0 - om;
    om2 = (1.0 - om/2.0)*2.0*td*B;
    Cr2 = Cr*Cr;
    uvw = -1.5*(tu*tu + tv*tv + tw*tw)*Cr2;

    f___ = om1*f___ + om*td/3.0 *(1.0                                                  + uvw)  // ___
         + om2/3.0*(3.0*-tu*Cr*tfx + 3.0*-tv*Cr*tfy + 3.0*-tw*Cr*tfz);

    fp__ = om1*fp__ + om*td/18.0*(1.0 + 3.0*tu*Cr        + 4.5*tu*tu*Cr2               + uvw)  // p__
         + om2/18.0*((3.0*( 1.0-tu*Cr)+9.0*Cr*tu)*tfx + 3.0*-tv*Cr*tfy + 3.0*-tw*Cr*tfz);

    fm__ = om1*fm__ + om*td/18.0*(1.0 - 3.0*tu*Cr        + 4.5*tu*tu*Cr2               + uvw)  // m__
         + om2/18.0*((3.0*(-1.0-tu*Cr)+9.0*Cr*tu)*tfx + 3.0*-tv*Cr*tfy + 3.0*-tw*Cr*tfz);

    f_p_ = om1*f_p_ + om*td/18.0*(1.0 + 3.0*tv*Cr        + 4.5*tv*tv*Cr2               + uvw)  // _p_
         + om2/18.0*(3.0*-tu*Cr*tfx + (3.0*( 1.0-tv*Cr)+9.0*Cr*tv)*tfy + 3.0*-tw*Cr*tfz);

    f_m_ = om1*f_m_ + om*td/18.0*(1.0 - 3.0*tv*Cr        + 4.5*tv*tv*Cr2               + uvw)  // _m_
         + om2/18.0*(3.0*-tu*Cr*tfx + (3.0*(-1.0-tv*Cr)+9.0*Cr*tv)*tfy + 3.0*-tw*Cr*tfz);

    f__p = om1*f__p + om*td/18.0*(1.0 + 3.0*tw*Cr        + 4.5*tw*tw*Cr2               + uvw)  // __p
         + om2/18.0*(3.0*-tu*Cr*tfx + 3.0*-tv*Cr*tfy + (3.0*( 1.0-tw*Cr)+9.0*Cr*tw)*tfz);

    f__m = om1*f__m + om*td/18.0*(1.0 - 3.0*tw*Cr        + 4.5*tw*tw*Cr2               + uvw)  // __m
         + om2/18.0*(3.0*-tu*Cr*tfx + 3.0*-tv*Cr*tfy + (3.0*(-1.0-tw*Cr)+9.0*Cr*tw)*tfz);

    fpp_ = om1*fpp_ + om*td/36.0*(1.0 + 3.0*(tu + tv)*Cr + 4.5*(tu + tv)*(tu + tv)*Cr2 + uvw)  // pp_
         + om2/36.0*((3.0*( 1.0-tu*Cr)+9.0*Cr*(tu+tv))*tfx + (3.0*( 1.0-tv*Cr)+9.0*Cr*(tu+tv))*tfy + 3.0*-tw*Cr*tfz);

    fmm_ = om1*fmm_ + om*td/36.0*(1.0 - 3.0*(tu + tv)*Cr + 4.5*(tu + tv)*(tu + tv)*Cr2 + uvw)  // mm_
         + om2/36.0*((3.0*(-1.0-tu*Cr)+9.0*Cr*(tu+tv))*tfx + (3.0*(-1.0-tv*Cr)+9.0*Cr*(tu+tv))*tfy + 3.0*-tw*Cr*tfz);

    fmp_ = om1*fmp_ + om*td/36.0*(1.0 - 3.0*(tu - tv)*Cr + 4.5*(tu - tv)*(tu - tv)*Cr2 + uvw)  // mp_
         + om2/36.0*((3.0*(-1.0-tu*Cr)+9.0*Cr*(tu-tv))*tfx + (3.0*( 1.0-tv*Cr)-9.0*Cr*(tu-tv))*tfy + 3.0*-tw*Cr*tfz);

    fpm_ = om1*fpm_ + om*td/36.0*(1.0 + 3.0*(tu - tv)*Cr + 4.5*(tu - tv)*(tu - tv)*Cr2 + uvw)  // pm_
         + om2/36.0*((3.0*( 1.0-tu*Cr)+9.0*Cr*(tu-tv))*tfx + (3.0*(-1.0-tv*Cr)-9.0*Cr*(tu-tv))*tfy + 3.0*-tw*Cr*tfz);

    f_pp = om1*f_pp + om*td/36.0*(1.0 + 3.0*(tv + tw)*Cr + 4.5*(tv + tw)*(tv + tw)*Cr2 + uvw)  // _pp
         + om2/36.0*(3.0*-tu*Cr*tfx + (3.0*( 1.0-tv*Cr)+9.0*Cr*(tv+tw))*tfy + (3.0*( 1.0-tw*Cr)+9.0*Cr*(tv+tw))*tfz);

    f_mm = om1*f_mm + om*td/36.0*(1.0 - 3.0*(tv + tw)*Cr + 4.5*(tv + tw)*(tv + tw)*Cr2 + uvw)  // _mm
         + om2/36.0*(3.0*-tu*Cr*tfx + (3.0*(-1.0-tv*Cr)+9.0*Cr*(tv+tw))*tfy + (3.0*(-1.0-tw*Cr)+9.0*Cr*(tv+tw))*tfz);

    f_mp = om1*f_mp + om*td/36.0*(1.0 - 3.0*(tv - tw)*Cr + 4.5*(tv - tw)*(tv - tw)*Cr2 + uvw)  // _mp
         + om2/36.0*(3.0*-tu*Cr*tfx + (3.0*(-1.0-tv*Cr)+9.0*Cr*(tv-tw))*tfy + (3.0*( 1.0-tw*Cr)-9.0*Cr*(tv-tw))*tfz);

    f_pm = om1*f_pm + om*td/36.0*(1.0 + 3.0*(tv - tw)*Cr + 4.5*(tv - tw)*(tv - tw)*Cr2 + uvw)  // _pm
         + om2/36.0*(3.0*-tu*Cr*tfx + (3.0*( 1.0-tv*Cr)+9.0*Cr*(tv-tw))*tfy + (3.0*(-1.0-tw*Cr)-9.0*Cr*(tv-tw))*tfz);

    fp_p = om1*fp_p + om*td/36.0*(1.0 + 3.0*(tw + tu)*Cr + 4.5*(tw + tu)*(tw + tu)*Cr2 + uvw)  // p_p
         + om2/36.0*((3.0*( 1.0-tu*Cr)+9.0*Cr*(tw+tu))*tfx + 3.0*-tv*Cr*tfy + (3.0*( 1.0-tw*Cr)+9.0*Cr*(tw+tu))*tfz);

    fm_m = om1*fm_m + om*td/36.0*(1.0 - 3.0*(tw + tu)*Cr + 4.5*(tw + tu)*(tw + tu)*Cr2 + uvw)  // m_m
         + om2/36.0*((3.0*(-1.0-tu*Cr)+9.0*Cr*(tw+tu))*tfx + 3.0*-tv*Cr*tfy + (3.0*(-1.0-tw*Cr)+9.0*Cr*(tw+tu))*tfz);

    fp_m = om1*fp_m + om*td/36.0*(1.0 - 3.0*(tw - tu)*Cr + 4.5*(tw - tu)*(tw - tu)*Cr2 + uvw)  // p_m
         + om2/36.0*((3.0*( 1.0-tu*Cr)-9.0*Cr*(tw-tu))*tfx + 3.0*-tv*Cr*tfy + (3.0*(-1.0-tw*Cr)+9.0*Cr*(tw-tu))*tfz);

    fm_p = om1*fm_p + om*td/36.0*(1.0 + 3.0*(tw - tu)*Cr + 4.5*(tw - tu)*(tw - tu)*Cr2 + uvw)  // m_p
         + om2/36.0*((3.0*(-1.0-tu*Cr)-9.0*Cr*(tw-tu))*tfx + 3.0*-tv*Cr*tfy + (3.0*( 1.0-tw*Cr)+9.0*Cr*(tw-tu))*tfz);

// Streaming step & Bounce back ---
    if(i___ >= 0) j = i___*Q + 0; else j = i*Q + 0;                              // ___
    fm[j] = f___;
    if(ip__ >= 0) j = ip__*Q + 1; else j = i*Q + 2;                              // p__
    fm[j] = fp__;
    if(im__ >= 0) j = im__*Q + 2; else j = i*Q + 1;                              // m__
    fm[j] = fm__;
    if(i_p_ >= 0) j = i_p_*Q + 3; else j = i*Q + 4;                              // _p_
    fm[j] = f_p_;
    if(i_m_ >= 0) j = i_m_*Q + 4; else j = i*Q + 3;                              // _m_
    fm[j] = f_m_;
    if(i__p >= 0) j = i__p*Q + 5; else j = i*Q + 6;                              // __p
    fm[j] = f__p;
    if(i__m >= 0) j = i__m*Q + 6; else j = i*Q + 5;                              // __m
    fm[j] = f__m;
    if(ipp_ >= 0) j = ipp_*Q + 7; else j = i*Q + 8;                              // pp_
    fm[j] = fpp_;
    if(imm_ >= 0) j = imm_*Q + 8; else j = i*Q + 7;                              // mm_
    fm[j] = fmm_;
    if(imp_ >= 0) j = imp_*Q + 9; else j = i*Q +10;                              // mp_
    fm[j] = fmp_;
    if(ipm_ >= 0) j = ipm_*Q +10; else j = i*Q + 9;                              // pm_
    fm[j] = fpm_;
//wall move top and under
    #if (allwall == true)
    if(i_pp >= 0){ j = i_pp*Q +11; fm[j] = f_pp;                               } // _pp
    else         { j = i   *Q +12; fm[j] = f_pp - 6.0*td*Cr/36.0*sra*(LY/2.0); }
    if(i_mm >= 0){ j = i_mm*Q +12; fm[j] = f_mm;                               } // _mm
    else         { j = i   *Q +11; fm[j] = f_mm - 6.0*td*Cr/36.0*sra*(LY/2.0); }
    if(i_mp >= 0){ j = i_mp*Q +13; fm[j] = f_mp;                               } // _mp
    else         { j = i   *Q +14; fm[j] = f_mp + 6.0*td*Cr/36.0*sra*(LY/2.0); }
    if(i_pm >= 0){ j = i_pm*Q +14; fm[j] = f_pm;                               } // _pm
    else         { j = i   *Q +13; fm[j] = f_pm + 6.0*td*Cr/36.0*sra*(LY/2.0); }
// wall move top
    #elif (topwall == true)
    if(i_pp >= 0){ j = i_pp*Q +11; fm[j] = f_pp;                               } // _pp
    else         { j = i   *Q +12; fm[j] = f_pp - 6.0*td*Cr/36.0*sra*LY;       }
    if(i_mm >= 0){ j = i_mm*Q +12; fm[j] = f_mm;                               } // _mm
    else           j = i   *Q +11; fm[j] = f_mm;  
    if(i_mp >= 0){ j = i_mp*Q +13; fm[j] = f_mp;                               } // _mp
    else           j = i   *Q +14; fm[j] = f_mp;  
    if(i_pm >= 0){ j = i_pm*Q +14; fm[j] = f_pm;                               } // _pm
    else         { j = i   *Q +13; fm[j] = f_pm + 6.0*td*Cr/36.0*sra*LY;       }
    #endif
    if(ip_p >= 0) j = ip_p*Q +15; else j = i*Q +16;                              // p_p
    fm[j] = fp_p;
    if(im_m >= 0) j = im_m*Q +16; else j = i*Q +15;                              // m_m
    fm[j] = fm_m;
    if(ip_m >= 0) j = ip_m*Q +17; else j = i*Q +18;                              // p_m
    fm[j] = fp_m;
    if(im_p >= 0) j = im_p*Q +18; else j = i*Q +17;                              // m_p
    fm[j] = fm_p;
}
#endif // CapsuelSHEARFLOW

#ifdef CapsuleCHANNELFLOW
__global__ void  lbm_CapsuleCHANNELFLOW
//==========================================================
//
//  LATTICE BOLTZMANN METHOD ON GPU
//
//
(
    int       iter,
    double    tau,
    double    dt,
    double    *fn,
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
    double    *vf2,
    int       *bc,
    int       nx,
    int       ny,
    int       nz,
    int       n
)
//----------------------------------------------------------
{
    int       i, j,
              i___, ip__, im__, i_p_, i_m_, i__p, i__m,
              ipp_, imm_, imp_, ipm_,
              i_pp, i_mm, i_mp, i_pm,
              ip_p, im_m, ip_m, im_p;
    double    td, tu, tv, tw, tfx, tfy, tfz, uvw, Cr2, om, om1, om2, om3, /* tvf1, tvf2 */
              f___, fp__, fm__, f_p_, f_m_, f__p, f__m,
              fpp_, fmm_, fmp_, fpm_,
              f_pp, f_mm, f_mp, f_pm,
              fp_p, fm_m, fp_m, fm_p;
    double    ui = 0.0, vi = 0.0, wi;


    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= n)        return;
    if(bc[i] != COMP) return;
    double t   = dt*(double)iter;
    //if(iter > FLOWSTART) wi = 1.0;
    if(iter > FLOWSTART) wi = 1.0 + 0.5 * sin(M_PI*t/FREQ);
    else                 wi = 0.0;

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
///*
// Domain connectivity ---
    i___ = i;
    if     (bc[i+1]        == WALL) ip__ = -1;
    else                            ip__ = i+1;
    if     (bc[i-1]        == WALL) im__ = -1;
    else                            im__ = i-1;
    if     (bc[i+nx]       == WALL) i_p_ = -1;
    else                            i_p_ = i+nx;
    if     (bc[i-nx]       == WALL) i_m_ = -1;
    else                            i_m_ = i-nx;
    if     (bc[i+nx*ny]    == BUFF) i__p = i-nx*ny*(nz-5);
    else                            i__p = i+nx*ny;
    if     (bc[i-nx*ny]    == BUFF) i__m = i+nx*ny*(nz-5);
    else                            i__m = i-nx*ny;
    if     (bc[i+1+nx]     == WALL) ipp_ = -1;
    else                            ipp_ = i+1+nx;
    if     (bc[i-1-nx]     == WALL) imm_ = -1;
    else                            imm_ = i-1-nx;
    if     (bc[i-1+nx]     == WALL) imp_ = -1;
    else                            imp_ = i-1+nx;
    if     (bc[i+1-nx]     == WALL) ipm_ = -1;
    else                            ipm_ = i+1-nx;
    if     (bc[i+nx+nx*ny] == WALL) i_pp = -1;
    else if(bc[i+nx+nx*ny] == BUFF) i_pp = i+nx-nx*ny*(nz-5);
    else                            i_pp = i+nx+nx*ny;
    if     (bc[i-nx-nx*ny] == WALL) i_mm = -1;
    else if(bc[i-nx-nx*ny] == BUFF) i_mm = i-nx+nx*ny*(nz-5);
    else                            i_mm = i-nx-nx*ny;
    if     (bc[i-nx+nx*ny] == WALL) i_mp = -1;
    else if(bc[i-nx+nx*ny] == BUFF) i_mp = i-nx-nx*ny*(nz-5);
    else                            i_mp = i-nx+nx*ny;
    if     (bc[i+nx-nx*ny] == WALL) i_pm = -1;
    else if(bc[i+nx-nx*ny] == BUFF) i_pm = i+nx+nx*ny*(nz-5);
    else                            i_pm = i+nx-nx*ny;
    if     (bc[i+1+nx*ny]  == WALL) ip_p = -1;
    else if(bc[i+1+nx*ny]  == BUFF) ip_p = i+1-nx*ny*(nz-5);
    else                            ip_p = i+1+nx*ny;
    if     (bc[i-1-nx*ny]  == WALL) im_m = -1;
    else if(bc[i-1-nx*ny]  == BUFF) im_m = i-1+nx*ny*(nz-5);
    else                            im_m = i-1-nx*ny;
    if     (bc[i+1-nx*ny]  == WALL) ip_m = -1;
    else if(bc[i+1-nx*ny]  == BUFF) ip_m = i+1+nx*ny*(nz-5);
    else                            ip_m = i+1-nx*ny;
    if     (bc[i-1+nx*ny]  == WALL) im_p = -1;
    else if(bc[i-1+nx*ny]  == BUFF) im_p = i-1-nx*ny*(nz-5);
    else                            im_p = i-1+nx*ny;
//*/
// Density and Velocity ---
    if((iter-1)%M == 0){
        dm[i] = 0.0;
        um[i] = 0.0;
        vm[i] = 0.0;
        wm[i] = 0.0;
    }

    tfx = fx[i];
    tfy = fy[i];
    tfz = fz[i];

    double tvf1 = vf1[i];
    //double tvf2 = vf2[i];

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
    om  = 1.0/(3.0*(1.0 + (LAMBDA1 - 1.0)*tvf1)*Df + 0.5);
//  om  = 1.0/(3.0*(1.0 + (LAMBDA1 - 1.0)*tvf1 + (LAMBDA2 - 1.0)*tvf2)*Df + 0.5);
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
/*
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
    else j = (i+1+nx*ny)*Q + 15;                                // p_p
    fm[j] = fp_p;
    if(bc[i-1-nx*ny] == WALL) j = i*Q + 15; 
    else if(bc[i-1-nx*ny] == BUFF) j = (i-1+nx*ny*(nz-5))*Q +16;
    else j = (i-1-nx*ny)*Q + 16;                                // m_m
    fm[j] = fm_m;
    if(bc[i+1-nx*ny] == WALL) j = i*Q + 18; 
    else if(bc[i+1-nx*ny] == BUFF) j = (i+1+nx*ny*(nz-5))*Q +17;
    else j = (i+1-nx*ny)*Q + 17;                                // p_m
    fm[j] = fp_m;
    if(bc[i-1+nx*ny] == WALL) j = i*Q + 17; 
    else if(bc[i-1+nx*ny] == BUFF) j = (i-1-nx*ny*(nz-5))*Q +18;
    else j = (i-1+nx*ny)*Q + 18;                                // m_p
    fm[j] = fm_p;
*/
///*
// Streaming step & Bounce back ---
    if(i___ >= 0) j = i___*Q + 0; else j = i*Q + 0; // ___
    fm[j] = f___;
    if(ip__ >= 0) j = ip__*Q + 1; else j = i*Q + 2; // p__
    fm[j] = fp__;
    if(im__ >= 0) j = im__*Q + 2; else j = i*Q + 1; // m__
    fm[j] = fm__;
    if(i_p_ >= 0) j = i_p_*Q + 3; else j = i*Q + 4; // _p_
    fm[j] = f_p_;
    if(i_m_ >= 0) j = i_m_*Q + 4; else j = i*Q + 3; // _m_
    fm[j] = f_m_;
    if(i__p >= 0) j = i__p*Q + 5; else j = i*Q + 6; // __p
    fm[j] = f__p;
    if(i__m >= 0) j = i__m*Q + 6; else j = i*Q + 5; // __m
    fm[j] = f__m;
    if(ipp_ >= 0) j = ipp_*Q + 7; else j = i*Q + 8; // pp_
    fm[j] = fpp_;
    if(imm_ >= 0) j = imm_*Q + 8; else j = i*Q + 7; // mm_
    fm[j] = fmm_;
    if(imp_ >= 0) j = imp_*Q + 9; else j = i*Q +10; // mp_
    fm[j] = fmp_;
    if(ipm_ >= 0) j = ipm_*Q +10; else j = i*Q + 9; // pm_
    fm[j] = fpm_;
    if(i_pp >= 0) j = i_pp*Q +11; else j = i*Q +12; // _pp
    fm[j] = f_pp;
    if(i_mm >= 0) j = i_mm*Q +12; else j = i*Q +11; // _mm
    fm[j] = f_mm;
    if(i_mp >= 0) j = i_mp*Q +13; else j = i*Q +14; // _mp
    fm[j] = f_mp;
    if(i_pm >= 0) j = i_pm*Q +14; else j = i*Q +13; // _pm
    fm[j] = f_pm;
    if(ip_p >= 0) j = ip_p*Q +15; else j = i*Q +16; // p_p
    fm[j] = fp_p;
    if(im_m >= 0) j = im_m*Q +16; else j = i*Q +15; // m_m
    fm[j] = fm_m;
    if(ip_m >= 0) j = ip_m*Q +17; else j = i*Q +18; // p_m
    fm[j] = fp_m;
    if(im_p >= 0) j = im_p*Q +18; else j = i*Q +17; // m_p
    fm[j] = fm_p;
//*/
}
#endif // CapsuleCHANNELFLOW

#if defined(SHEARFLOW) || defined(CapsuleSHEARFLOW)
__global__ void  lbm_SHEARFLOW_Boundary
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
    int       i, ix, iy, iz, i1, i2;


    i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i >= n) return;

    if(bc[i] == COMP){
        return;
    }
    else if(bc[i] == BUFF){
        iz = (int)(i/(nx*ny));
        iy = (int)(i/nx) - ny*iz;
        ix = i - nx*iy - nx*ny*iz;

        i1 = i;
        i2 = i1;

        if(ix <= 1   ) i2 += (nx-4);
        if(ix >= nx-2) i2 -= (nx-4);
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
#endif // SHEARFLOW || CapsuleSHEARFLOW

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
    int       thread = 128, block = cdo->n/thread+1;
    dim3      dimB(thread,1,1), dimG(block,1,1);


#if defined(KARMAN)
    double    *tmp, *tmpd, *tmpu, *tmpv, *tmpw;

    lbm_KARMAN <<< dimG,dimB >>>
    (prc->iter,cdo->tau,ltc->fnD,ltc->fmD,
     ltc->dnD,ltc->unD,ltc->vnD,ltc->wnD,
     ltc->dmD,ltc->umD,ltc->vmD,ltc->wmD,
     ltc->bcD,cdo->nx,cdo->ny,cdo->nz,cdo->n);

    cudaThreadSynchronize();

    tmp  = ltc->fnD; ltc->fnD = ltc->fmD; ltc->fmD = tmp;
    tmpd = ltc->dnD; ltc->dnD = ltc->dmD; ltc->dmD = tmpd;
    tmpu = ltc->unD; ltc->unD = ltc->umD; ltc->umD = tmpu;
    tmpv = ltc->vnD; ltc->vnD = ltc->vmD; ltc->vmD = tmpv;
    tmpw = ltc->wnD; ltc->wnD = ltc->wmD; ltc->wmD = tmpw;
#elif defined(SHEARFLOW)
    double    *tmp;

    lbm_SHEARFLOW <<< dimG,dimB >>>
    (prc->iter,cdo->tau,ltc->fnD,ltc->fmD,
     ltc->dnD,ltc->unD,ltc->vnD,ltc->wnD,
     ltc->dmD,ltc->umD,ltc->vmD,ltc->wmD,
     ltc->bcD,cdo->nx,cdo->ny,cdo->nz,cdo->n);

    lbm_SHEARFLOW_Boundary <<< dimG,dimB >>>
    (ltc->dnD,ltc->unD,ltc->vnD,ltc->wnD,
     ltc->dmD,ltc->umD,ltc->vmD,ltc->wmD,
     ltc->bcD,cdo->nx,cdo->ny,cdo->nz,cdo->n);

    cudaThreadSynchronize();

    tmp  = ltc->fnD; ltc->fnD = ltc->fmD; ltc->fmD = tmp;
#elif defined(CHANNELFLOW)
    double    *tmp;

    lbm_CHANNELFLOW <<< dimG,dimB >>>
    (prc->iter,cdo->tau,ltc->fnD,ltc->fmD,
     ltc->dnD,ltc->unD,ltc->vnD,ltc->wnD,
     ltc->dmD,ltc->umD,ltc->vmD,ltc->wmD,
     ltc->bcD,cdo->nx,cdo->ny,cdo->nz,cdo->n);

    lbm_CHANNELFLOW_Boundary <<< dimG,dimB >>>
    (ltc->dnD,ltc->unD,ltc->vnD,ltc->wnD,
     ltc->dmD,ltc->umD,ltc->vmD,ltc->wmD,
     ltc->bcD,cdo->nx,cdo->ny,cdo->nz,cdo->n);

    cudaThreadSynchronize();

    tmp  = ltc->fnD; ltc->fnD = ltc->fmD; ltc->fmD = tmp;
#elif defined(CapsuleSHEARFLOW)
    double    *tmp;

    lbm_CapsuleSHEARFLOW <<< dimG,dimB >>>
    (prc->iter,cdo->tau,cdo->sr,ltc->fnD,ltc->fmD,
     ltc->dnD,ltc->unD,ltc->vnD,ltc->wnD,
     ltc->dmD,ltc->umD,ltc->vmD,ltc->wmD,
     ltc->fxD,ltc->fyD,ltc->fzD,ltc->vfD,ltc->vfD2,
     ltc->bcD,cdo->nx,cdo->ny,cdo->nz,cdo->n);

    lbm_SHEARFLOW_Boundary <<< dimG,dimB >>>
    (ltc->dnD,ltc->unD,ltc->vnD,ltc->wnD,
     ltc->dmD,ltc->umD,ltc->vmD,ltc->wmD,
     ltc->bcD,cdo->nx,cdo->ny,cdo->nz,cdo->n);

    if(prc->iter%M == 0){
        lbm_AVERAGE <<< dimG,dimB >>>
        (ltc->umD,ltc->vmD,ltc->wmD,cdo->n);
    }

    cudaThreadSynchronize();

    tmp  = ltc->fnD; ltc->fnD = ltc->fmD; ltc->fmD = tmp;
#elif defined(CapsuleCHANNELFLOW)
    double    *tmp;

    lbm_CapsuleCHANNELFLOW <<< dimG,dimB >>>
    (prc->iter,cdo->tau,cdo->dt,ltc->fnD,ltc->fmD,
     ltc->dnD,ltc->unD,ltc->vnD,ltc->wnD,
     ltc->dmD,ltc->umD,ltc->vmD,ltc->wmD,
     ltc->fxD,ltc->fyD,ltc->fzD,ltc->vfD,ltc->vfD2,
     ltc->bcD,cdo->nx,cdo->ny,cdo->nz,cdo->n);

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

