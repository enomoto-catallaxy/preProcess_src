#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "def.h"


void  cdo_init
//==========================================================
//
//  INITIAL SETTING OF COMPUTATIONAL DOMAIN
//
//
(
 domain    *cdo,
 lattice   *ltc
 )
//----------------------------------------------------------
{
  // Initialize computational domain ---
  printf("Initializing computational domain ----------------------------------\n");

  // Generate domain ---
#if defined(KARMAN)
  int       i, ix, iy, iz;
  double    x, y, z, r;
  double    Cy = 0.25, Cz = -LZ/4.0;

  cdo->dx  = Lc/KL;
  cdo->dt  = Cr*cdo->dx/Uc;
  cdo->tau = (6.0*Df + 1.0)/2.0;

  cdo->nx = (int)floor(LX/cdo->dx + 1.0e-03) + 2;
  cdo->ny = (int)floor(LY/cdo->dx + 1.0e-03) + 2;
  cdo->nz = (int)floor(LZ/cdo->dx + 1.0e-03) + 0;
  cdo->n  = cdo->nx*cdo->ny*cdo->nz;

  cdo->xmin = -LX/2.0; cdo->xmax = LX/2.0;
  cdo->ymin = -LY/2.0; cdo->ymax = LY/2.0;
  cdo->zmin = -LZ/2.0; cdo->zmax = LZ/2.0;

  ltc->bcH = (int *)malloc(sizeof(int)*cdo->n); if(ltc->bcH == NULL) error(0);

  for(i = 0;i < cdo->n;i++){
    iz = (int)(i/(cdo->nx*cdo->ny));
    iy = (int)(i/cdo->nx) - cdo->ny*iz;
    ix = i - cdo->nx*iy - cdo->nx*cdo->ny*iz;

    x = -LX/2.0 - cdo->dx/2.0 + (double)ix*cdo->dx;
    y = -LY/2.0 - cdo->dx/2.0 + (double)iy*cdo->dx;
    z = -LZ/2.0 + cdo->dx/2.0 + (double)iz*cdo->dx;
    r = sqrt((Cy-y)*(Cy-y) + (Cz-z)*(Cz-z));

    if(x < cdo->xmin || x > cdo->xmax
        || y < cdo->ymin || y > cdo->ymax) ltc->bcH[i] = WALL;
    else if(r  <  Lc       )           ltc->bcH[i] = WALL;
    else if(iz == 0        )           ltc->bcH[i] = INLET;
    else if(iz == cdo->nz-1)           ltc->bcH[i] = OUTLET;
    else                               ltc->bcH[i] = COMP;
  }
#elif defined(SHEARFLOW)
  int       i, ix, iy, iz;
  double    x, y, z;

  cdo->dx  = Lc/KL;
  cdo->dt  = Cr*cdo->dx/Uc;
  cdo->tau = (6.0*Df + 1.0)/2.0;

  cdo->nx = (int)floor(LX/cdo->dx + 1.0e-03) + 4;
  cdo->ny = (int)floor(LY/cdo->dx + 1.0e-03) + 4;
  cdo->nz = (int)floor(LZ/cdo->dx + 1.0e-03) + 4;
  cdo->n  = cdo->nx*cdo->ny*cdo->nz;

  cdo->xmin = -LX/2.0; cdo->xmax = LX/2.0;
  cdo->ymin = -LY/2.0; cdo->ymax = LY/2.0;
  cdo->zmin = -LZ/2.0; cdo->zmax = LZ/2.0;

  ltc->bcH = (int *)malloc(sizeof(int)*cdo->n); if(ltc->bcH == NULL) error(0);

  for(i = 0;i < cdo->n;i++){
    iz = (int)(i/(cdo->nx*cdo->ny));
    iy = (int)(i/cdo->nx) - cdo->ny*iz;
    ix = i - cdo->nx*iy - cdo->nx*cdo->ny*iz;

    x = -LX/2.0 - 1.5*cdo->dx + (double)ix*cdo->dx;
    y = -LY/2.0 - 1.5*cdo->dx + (double)iy*cdo->dx;
    z = -LZ/2.0 - 1.5*cdo->dx + (double)iz*cdo->dx;

    if     (x > cdo->xmin && x < cdo->xmax
        && y > cdo->ymin && y < cdo->ymax
        && z > cdo->zmin && z < cdo->zmax) ltc->bcH[i] = COMP;
    else if(y > cdo->ymin && y < cdo->ymax) ltc->bcH[i] = BUFF;
    else                                    ltc->bcH[i] = WALL;
  }
#elif defined(CHANNELFLOW)
  int       i, ix, iy, iz;
  double    x, y, z, r;

  cdo->dx  = Lc/KL;
  cdo->dt  = Cr*cdo->dx/Uc;
  cdo->tau = (6.0*Df + 1.0)/2.0;

  cdo->nx = (int)floor(LX/cdo->dx + 1.0e-03) + 4;
  cdo->ny = (int)floor(LY/cdo->dx + 1.0e-03) + 4;
  cdo->nz = (int)floor(LZ/cdo->dx + 1.0e-03) + 4;
  cdo->n  = cdo->nx*cdo->ny*cdo->nz;

  cdo->xmin = -LX/2.0; cdo->xmax = LX/2.0;
  cdo->ymin = -LY/2.0; cdo->ymax = LY/2.0;
  cdo->zmin = -LZ/2.0; cdo->zmax = LZ/2.0;

  ltc->bcH = (int *)malloc(sizeof(int)*cdo->n); if(ltc->bcH == NULL) error(0);

  for(i = 0;i < cdo->n;i++){
    iz = (int)(i/(cdo->nx*cdo->ny));
    iy = (int)(i/cdo->nx) - cdo->ny*iz;
    ix = i - cdo->nx*iy - cdo->nx*cdo->ny*iz;

    x = -LX/2.0 - 1.5*cdo->dx + (double)ix*cdo->dx;
    y = -LY/2.0 - 1.5*cdo->dx + (double)iy*cdo->dx;
    z = -LZ/2.0 - 1.5*cdo->dx + (double)iz*cdo->dx;

    r = sqrt(x*x + y*y);
    if     (r <= Lc/2.0 &&  z > cdo->zmin && z < cdo->zmax ) ltc->bcH[i] = COMP;
    else if(r <= Lc/2.0 && (z < cdo->zmin || z > cdo->zmax)) ltc->bcH[i] = BUFF;
    else                                                     ltc->bcH[i] = WALL;
  }
#elif defined(CapsuleSHEARFLOW)
  int       i, ix, iy, iz;
  double    x, y, z;

  if(MAX(Df,LAMBDA1*Df) > DfLIMIT) error(7);

  cdo->dx     = Lc/KL;
  cdo->dt     = Cr*cdo->dx/Uc;
  cdo->tau    = (6.0*Df + 1.0)/2.0;
  cdo->sr     = Uc/Lc;
  cdo->lambda = LAMBDA1;

  cdo->nx = (int)floor(LX/cdo->dx + 1.0e-03) + 4;
  cdo->ny = (int)floor(LY/cdo->dx + 1.0e-03) + 4;
  cdo->nz = (int)floor(LZ/cdo->dx + 1.0e-03) + 4;
  cdo->n  = cdo->nx*cdo->ny*cdo->nz;
  //adhesion wall domain ---
  cdo->nxw = cdo->nx;
  cdo->nyw = 2;
  cdo->nzw = cdo->nz;
  cdo->nw  = cdo->nxw*cdo->nyw*cdo->nzw;

  cdo->xmin = -LX/2.0; cdo->xmax = LX/2.0;
  cdo->ymin = -LY/2.0; cdo->ymax = LY/2.0;
  cdo->zmin = -LZ/2.0; cdo->zmax = LZ/2.0;

  ltc->bcH = (int   *)malloc(sizeof(int   )*cdo->n)      ;if(ltc->bcH == NULL) error(0);
  ltc->xwH = (double*)malloc(sizeof(double)*cdo->nw     );if(ltc->xwH == NULL) error(0);
  ltc->ywH = (double*)malloc(sizeof(double)*cdo->nw     );if(ltc->ywH == NULL) error(0);
  ltc->zwH = (double*)malloc(sizeof(double)*cdo->nw     );if(ltc->zwH == NULL) error(0);

  for(i = 0;i < cdo->n;i++){
    iz = (int)(i/(cdo->nx*cdo->ny));
    iy = (int)(i/cdo->nx) - cdo->ny*iz;
    ix = i - cdo->nx*iy - cdo->nx*cdo->ny*iz;

    x = -LX/2.0 - 1.5*cdo->dx + (double)ix*cdo->dx;
    y = -LY/2.0 - 1.5*cdo->dx + (double)iy*cdo->dx;
    z = -LZ/2.0 - 1.5*cdo->dx + (double)iz*cdo->dx;

    if     (x > cdo->xmin && x < cdo->xmax
        && y > cdo->ymin && y < cdo->ymax
        && z > cdo->zmin && z < cdo->zmax) ltc->bcH[i] = COMP;
    else if(y > cdo->ymin && y < cdo->ymax) ltc->bcH[i] = BUFF;
    else                                    ltc->bcH[i] = WALL;
  }

  //adhesion wall domain ---
  for(i = 0;i < cdo->nw;i++){
    iy = (int)(i/(cdo->nx*cdo->nz));
    iz = (int)(i/cdo->nx) - cdo->nz*iy;
    ix = i - cdo->nx*iz - cdo->nx*cdo->nz*iy;
    ltc->xwH[i] = -LX/2.0 - 1.5*cdo->dx + (double)ix*cdo->dx;
    ltc->ywH[i] = -LY/2.0 + LY*iy;
    ltc->zwH[i] = -LZ/2.0 - 1.5*cdo->dx + (double)iz*cdo->dx;
  }

#elif defined(CapsuleCHANNELFLOW)
  int       i, ix, iy, iz;
  double    x, y, z, r, lambda = 0.0;

  if(N_R > 0 && N_W == 0 && N_P == 0 && N_C == 0){
    if(MAX(Df,LAMBDA1*Df) > DfLIMIT) error(7);
  }else{
    lambda = MAX(LAMBDA1,LAMBDA2);
    if(MAX(Df,lambda*Df) > DfLIMIT) error(7);
  }

  cdo->dx     = Lc/KL;
  cdo->dt     = Cr*cdo->dx/Uc;
  cdo->tau    = (6.0*Df + 1.0)/2.0;
  cdo->sr     = SR;
  cdo->lambda = LAMBDA1;

  cdo->nx = (int)floor(LX/cdo->dx + 1.0e-03) + 4;
  cdo->ny = (int)floor(LY/cdo->dx + 1.0e-03) + 4;
  cdo->nz = (int)floor(LZ/cdo->dx + 1.0e-03) + 4;
  cdo->n  = cdo->nx*cdo->ny*cdo->nz;

  cdo->xmin = -LX/2.0; cdo->xmax = LX/2.0;
  cdo->ymin = -LY/2.0; cdo->ymax = LY/2.0;
  cdo->zmin = -LZ/2.0; cdo->zmax = LZ/2.0;
  ltc->bcH = (int *)malloc(sizeof(int)*cdo->n); if(ltc->bcH == NULL) error(0);

//  int num_wall = 0;
//  int num_buff = 0;
  for(i = 0;i < cdo->n;i++){
    iz = (int)(i/(cdo->nx*cdo->ny));
    iy = (int)(i/cdo->nx) - cdo->ny*iz;
    ix = i - cdo->nx*iy - cdo->nx*cdo->ny*iz;

    x = -LX/2.0 - 1.5*cdo->dx + (double)ix*cdo->dx;
    y = -LY/2.0 - 1.5*cdo->dx + (double)iy*cdo->dx;
    z = -LZ/2.0 - 1.5*cdo->dx + (double)iz*cdo->dx;

    r = sqrt(x*x + y*y);
    if     (r <= Lc/2.0 &&  z > cdo->zmin && z < cdo->zmax ) ltc->bcH[i] = COMP;
    else if(r <= Lc/2.0 && (z < cdo->zmin || z > cdo->zmax)) ltc->bcH[i] = BUFF;
    else                                                     ltc->bcH[i] = WALL;
//    if (ltc->bcH[i] == WALL) num_wall ++;
//    else if (ltc->bcH[i] == BUFF) num_buff ++;
  }
//    printf(" all, comp, wall : %d %d %d\n", cdo->n, cdo->n - num_wall - num_buff, num_wall);
  //adhesion wall domain ---
/*
  for(i = 0;i < cdo->nw;i++){
    iz = (int)(i/cdo->nxw);
    ix = i - cdo->nxw*iz;
    ltc->xwH[i] =  LX/2.0*cos(2.0*M_PI*(double)ix/(double)cdo->nxw);
    ltc->ywH[i] =  LY/2.0*sin(2.0*M_PI*(double)ix/(double)cdo->nxw);
    ltc->zwH[i] = -LZ/2.0 - 1.5*cdo->dx + (double)iz*cdo->dx;
    //if(iz%2 == 0 && ix%2 == 0)     ltc->reonH[i] = 1;
    //else                           ltc->reonH[i] = 0;
//    ltc->reonH[i] = 1;
//    for(j = 0;j < nure;j++)ltc->reH[i + j*cdo->nw] = 0;

  }
*/
#endif
  // Numerical condition ---
#if defined(KARMAN) || defined(SHEARFLOW) || defined(CHANNELFLOW)
  printf("  Reynolds number           : %9.3e\n",Re);
  printf("  Spatial resolution        : %9.3e\n",KL);
  printf("  Diffusion number          : %9.3e\n",Df);
  printf("  Courant number            : %9.3e\n",Cr);
  printf("  time interval             : %9.3e\n",cdo->dt);
  printf("  lattice spacing           : %9.3e\n",cdo->dx);
  printf("  relaxation time           : %9.3e\n",cdo->tau);
  printf("  domain size               : (x)%9.3e (y)%9.3e (z)%9.3e\n",LX,LY,LZ);
  printf("  lattice node              : (x)%-9d (y)%-9d (z)%-9d\n",cdo->nx,cdo->ny,cdo->nz);
#elif defined(CapsuleSHEARFLOW) || defined(CapsuleCHANNELFLOW)
  printf("  Capillary number          : %9.3e\n",Ca);
  printf("  Reynolds number           : %9.3e\n",Re);
  printf("  Spatial resolution        : %9.3e\n",KL);
  printf("  Diffusion number          : %9.3e\n",Df);
  printf("  Courant number            : %9.3e\n",Cr);
  printf("  Undisturbed Courant number: %9.3e\n",B );
  printf("  time interval             : %9.3e\n",cdo->dt);
  printf("  lattice spacing           : %9.3e\n",cdo->dx);
  printf("  relaxation time           : %9.3e\n",cdo->tau);
  printf("  domain size               : (x)%9.3e (y)%9.3e (z)%9.3e\n",LX,LY,LZ);
  printf("  lattice node              : (x)%-9d (y)%-9d (z)%-9d\n",cdo->nx,cdo->ny,cdo->nz);
#endif

  return;
}


void  flu_init
//==========================================================
//
//  INITIAL SETTING OF FLUID
//
//
(
 domain    *cdo,
 lattice   *ltc
 )
//----------------------------------------------------------
{
  double    td, tu, tv, tw, tfx, tfy, tfz, tvf, uvw, Cr2;
#if defined(KARMAN)
  int       i;
#elif defined(SHEARFLOW)
  int       i, iy, iz;
#elif defined(CHANNELFLOW)
  int       i, ix, iy, iz;
#elif defined(CapsuleSHEARFLOW)
  int       i, iy, iz;
#elif defined(CapsuleCHANNELFLOW)
  int       i;
#endif


  // Initialize fluid ---
  printf("Initializing fluid -------------------------------------------------\n");

  // Allocate ---
  ltc->fnH   = (double *)malloc(sizeof(double)*cdo->n*Q          ); if(ltc->fnH   == NULL) error(0);
  ltc->fmH   = (double *)malloc(sizeof(double)*cdo->n*Q          ); if(ltc->fmH   == NULL) error(0);
  ltc->dnH   = (double *)malloc(sizeof(double)*cdo->n            ); if(ltc->dnH   == NULL) error(0);
  ltc->unH   = (double *)malloc(sizeof(double)*cdo->n            ); if(ltc->unH   == NULL) error(0);
  ltc->vnH   = (double *)malloc(sizeof(double)*cdo->n            ); if(ltc->vnH   == NULL) error(0);
  ltc->wnH   = (double *)malloc(sizeof(double)*cdo->n            ); if(ltc->wnH   == NULL) error(0);
  ltc->dmH   = (double *)malloc(sizeof(double)*cdo->n            ); if(ltc->dmH   == NULL) error(0);
  ltc->umH   = (double *)malloc(sizeof(double)*cdo->n            ); if(ltc->umH   == NULL) error(0);
  ltc->vmH   = (double *)malloc(sizeof(double)*cdo->n            ); if(ltc->vmH   == NULL) error(0);
  ltc->wmH   = (double *)malloc(sizeof(double)*cdo->n            ); if(ltc->wmH   == NULL) error(0);
  ltc->fxH   = (double *)malloc(sizeof(double)*cdo->n            ); if(ltc->fxH   == NULL) error(0);
  ltc->fyH   = (double *)malloc(sizeof(double)*cdo->n            ); if(ltc->fyH   == NULL) error(0);
  ltc->fzH   = (double *)malloc(sizeof(double)*cdo->n            ); if(ltc->fzH   == NULL) error(0);
  ltc->vfH   = (double *)malloc(sizeof(double)*cdo->n            ); if(ltc->vfH   == NULL) error(0);
  ltc->vfH2  = (double *)malloc(sizeof(double)*cdo->n            ); if(ltc->vfH2  == NULL) error(0);

  // Initial condition ---
  for(i = 0;i < cdo->n;i++){

#if defined(KARMAN)
    td  = 1.0;
    tu  = 0.0;
    tv  = 0.0;
    if(ltc->bcH[i] == INLET) tw = 1.0;
    else                     tw = 0.0;
    tfx = 0.0;
    tfy = 0.0;
    tfz = 0.0;
    tvf = 0.0;
#elif defined(SHEARFLOW)
    iz  = (int)(i/(cdo->nx*cdo->ny));
    iy  = (int)(i/cdo->nx) - cdo->ny*iz;
    td  = 1.0;
    tu  = 0.0;
    tv  = 0.0;
    tw  = (1.0/(LY/2.0))*(-LY/2.0 - 1.5*cdo->dx + (double)iy*cdo->dx);
    tfx = 0.0;
    tfy = 0.0;
    tfz = 0.0;
    tvf = 0.0;
#elif defined(CHANNELFLOW)
    iz  = (int)(i/(cdo->nx*cdo->ny));
    iy  = (int)(i/cdo->nx) - cdo->ny*iz;
    ix  = i - cdo->nx*iy - cdo->nx*cdo->ny*iz;
    //      x   = -LX/2.0 - 1.5*cdo->dx + (double)ix*cdo->dx;
    //      y   = -LY/2.0 - 1.5*cdo->dx + (double)iy*cdo->dx;
    //      r   = sqrt(x*x + y*y);
    td  = 1.0;
    tu  = 0.0;
    tv  = 0.0;
    tw  = 0.0;
    //      tw  = -cdo->pgrad*(cdo->ymax*cdo->ymax - rr*rr)/(4.0*cdo->d*cdo->dvout);
    tfx = 0.0;
    tfy = 0.0;
    tfz = 0.0;
    tvf = 0.0;
#elif defined(CapsuleSHEARFLOW)
    iz  = (int)(i/(cdo->nx*cdo->ny));
    iy  = (int)(i/cdo->nx) - cdo->ny*iz;
    td  = 1.0;
    tu  = 0.0;
    tv  = 0.0;
    #if(FLOWSTART == 0) 
      //wall move top and under
      #if(allwall == true)
      tw  = cdo->sr*(-LY/2.0 - 1.5*cdo->dx + (double)iy*cdo->dx);
      // wall move top
      #elif(topwall == true)
      tw  = cdo->sr*(- 1.5*cdo->dx + (double)iy*cdo->dx);
      #endif
    #elif(FLOWSTART > 0)
      tw  = 0.0;
    #endif
    tfx = 0.0;
    tfy = 0.0;
    tfz = 0.0;
    tvf = 0.0;
#elif defined(CapsuleCHANNELFLOW)
    td  = 1.0;
    tu  = 0.0;
    tv  = 0.0;
    tw  = 0.0;
    tfx = 0.0;
    tfy = 0.0;
    tfz = 0.0;
    tvf = 0.0;
#endif

    ltc->dnH[i]  = td;
    ltc->unH[i]  = tu;
    ltc->vnH[i]  = tv;
    ltc->wnH[i]  = tw;
    ltc->dmH[i]  = td;
    ltc->umH[i]  = tu;
    ltc->vmH[i]  = tv;
    ltc->wmH[i]  = tw;
    ltc->fxH[i]  = tfx;
    ltc->fyH[i]  = tfy;
    ltc->fzH[i]  = tfz;
    ltc->vfH[i]  = tvf;
    ltc->vfH2[i] = tvf;

    Cr2 = Cr*Cr;
    uvw = -1.5*(tu*tu + tv*tv + tw*tw)*Cr2;

    ltc->fnH[i*Q+ 0] = td/3.0 *(1.0                                                  + uvw); // ___
    ltc->fnH[i*Q+ 1] = td/18.0*(1.0 + 3.0*tu*Cr        + 4.5*tu*tu*Cr2               + uvw); // p__
    ltc->fnH[i*Q+ 2] = td/18.0*(1.0 - 3.0*tu*Cr        + 4.5*tu*tu*Cr2               + uvw); // m__
    ltc->fnH[i*Q+ 3] = td/18.0*(1.0 + 3.0*tv*Cr        + 4.5*tv*tv*Cr2               + uvw); // _p_
    ltc->fnH[i*Q+ 4] = td/18.0*(1.0 - 3.0*tv*Cr        + 4.5*tv*tv*Cr2               + uvw); // _m_
    ltc->fnH[i*Q+ 5] = td/18.0*(1.0 + 3.0*tw*Cr        + 4.5*tw*tw*Cr2               + uvw); // __p
    ltc->fnH[i*Q+ 6] = td/18.0*(1.0 - 3.0*tw*Cr        + 4.5*tw*tw*Cr2               + uvw); // __m
    ltc->fnH[i*Q+ 7] = td/36.0*(1.0 + 3.0*(tu + tv)*Cr + 4.5*(tu + tv)*(tu + tv)*Cr2 + uvw); // pp_
    ltc->fnH[i*Q+ 8] = td/36.0*(1.0 - 3.0*(tu + tv)*Cr + 4.5*(tu + tv)*(tu + tv)*Cr2 + uvw); // mm_
    ltc->fnH[i*Q+ 9] = td/36.0*(1.0 - 3.0*(tu - tv)*Cr + 4.5*(tu - tv)*(tu - tv)*Cr2 + uvw); // mp_
    ltc->fnH[i*Q+10] = td/36.0*(1.0 + 3.0*(tu - tv)*Cr + 4.5*(tu - tv)*(tu - tv)*Cr2 + uvw); // pm_
    ltc->fnH[i*Q+11] = td/36.0*(1.0 + 3.0*(tv + tw)*Cr + 4.5*(tv + tw)*(tv + tw)*Cr2 + uvw); // _pp
    ltc->fnH[i*Q+12] = td/36.0*(1.0 - 3.0*(tv + tw)*Cr + 4.5*(tv + tw)*(tv + tw)*Cr2 + uvw); // _mm
    ltc->fnH[i*Q+13] = td/36.0*(1.0 - 3.0*(tv - tw)*Cr + 4.5*(tv - tw)*(tv - tw)*Cr2 + uvw); // _mp
    ltc->fnH[i*Q+14] = td/36.0*(1.0 + 3.0*(tv - tw)*Cr + 4.5*(tv - tw)*(tv - tw)*Cr2 + uvw); // _pm
    ltc->fnH[i*Q+15] = td/36.0*(1.0 + 3.0*(tw + tu)*Cr + 4.5*(tw + tu)*(tw + tu)*Cr2 + uvw); // p_p
    ltc->fnH[i*Q+16] = td/36.0*(1.0 - 3.0*(tw + tu)*Cr + 4.5*(tw + tu)*(tw + tu)*Cr2 + uvw); // m_m
    ltc->fnH[i*Q+17] = td/36.0*(1.0 - 3.0*(tw - tu)*Cr + 4.5*(tw - tu)*(tw - tu)*Cr2 + uvw); // p_m
    ltc->fnH[i*Q+18] = td/36.0*(1.0 + 3.0*(tw - tu)*Cr + 4.5*(tw - tu)*(tw - tu)*Cr2 + uvw); // m_p

    ltc->fmH[i*Q+ 0] = 0.0;
    ltc->fmH[i*Q+ 1] = 0.0; ltc->fmH[i*Q+ 2] = 0.0;
    ltc->fmH[i*Q+ 3] = 0.0; ltc->fmH[i*Q+ 4] = 0.0;
    ltc->fmH[i*Q+ 5] = 0.0; ltc->fmH[i*Q+ 6] = 0.0;
    ltc->fmH[i*Q+ 7] = 0.0; ltc->fmH[i*Q+ 8] = 0.0;
    ltc->fmH[i*Q+ 9] = 0.0; ltc->fmH[i*Q+10] = 0.0;
    ltc->fmH[i*Q+11] = 0.0; ltc->fmH[i*Q+12] = 0.0;
    ltc->fmH[i*Q+13] = 0.0; ltc->fmH[i*Q+14] = 0.0;
    ltc->fmH[i*Q+15] = 0.0; ltc->fmH[i*Q+16] = 0.0;
    ltc->fmH[i*Q+17] = 0.0; ltc->fmH[i*Q+18] = 0.0;
  }

  return;
}


void  cel_init
//==========================================================
//
//  INITIAL SETTING OF CELL
//
//
(
 domain    *cdo,
 cell      *cel
)
//----------------------------------------------------------
{
#if defined(KARMAN) || defined(SHEARFLOW) || defined(CHANNELFLOW)
  return;
#elif defined(CapsuleSHEARFLOW) || defined(CapsuleCHANNELFLOW)
  #if(N_R == 0)
  return;
  #elif(N_R > 0)
  FILE      *fp;
  char      filename[256];
  int       i, j, k, e, ic, jid, eid, ecn,
            e1, e2, i1, i2, i3, j1, j2, j3, k1, k2, k3, k4;
  int       *sph_ele, *bic_ele, *shape;
  double    tx, ty, tz,
            *sph_x, *sph_y, *sph_z,
            *bic_x, *bic_y, *bic_z,
            *rp, *gxp, *gyp, *gzp, *alpha, *beta, *gamma;
  double    cost, cost2, sint,
            a1x, a2x, a3x, a4x, a21x, a31x, a34x, a24x, t1x, t2x, xix, zex,
            a1y, a2y, a3y, a4y, a21y, a31y, a34y, a24y, t1y, t2y, xiy, zey,
            a1z, a2z, a3z, a4z, a21z, a31z, a34z, a24z, t1z, t2z, xiz, zez;


  // Initialize cell ---
  printf("Initializing cell --------------------------------------------------\n");

  // Input cell file ---
  // read sphere shape
  sprintf(filename,"./cell/icosa_div/icosa_div%d_cal.dat",ICOSA);
  fp = fopen(filename,"r");
  if(fp == NULL) error(1);

  fscanf(fp,"%d %d",&cel->vertex,&cel->element);

  sph_x   = (double *)malloc(sizeof(double)*cel->vertex     ); if(sph_x   == NULL) error(0);
  sph_y   = (double *)malloc(sizeof(double)*cel->vertex     ); if(sph_y   == NULL) error(0);
  sph_z   = (double *)malloc(sizeof(double)*cel->vertex     ); if(sph_z   == NULL) error(0);
  sph_ele = (int    *)malloc(sizeof(int   )*cel->element*TRI); if(sph_ele == NULL) error(0);

  for(i = 0;i < cel->vertex;     i++) fscanf(fp,"%lf %lf %lf",&sph_x[i],&sph_y[i],&sph_z[i]);
  for(i = 0;i < cel->element*TRI;i++) fscanf(fp,"%d",&sph_ele[i]);
  fclose(fp);

  // read biconcave shape
  sprintf(filename,"./cell/rbc_div/rbc_div%d_cal.dat",ICOSA);
  fp = fopen(filename,"r");
  if(fp == NULL) error(1);

  fscanf(fp,"%d %d",&cel->vertex,&cel->element);

  bic_x   = (double *)malloc(sizeof(double)*cel->vertex     ); if(bic_x   == NULL) error(0);
  bic_y   = (double *)malloc(sizeof(double)*cel->vertex     ); if(bic_y   == NULL) error(0);
  bic_z   = (double *)malloc(sizeof(double)*cel->vertex     ); if(bic_z   == NULL) error(0);
  bic_ele = (int    *)malloc(sizeof(int   )*cel->element*TRI); if(bic_ele == NULL) error(0);

  for(i = 0;i < cel->vertex;     i++) fscanf(fp,"%lf %lf %lf",&bic_x[i],&bic_y[i],&bic_z[i]);
  for(i = 0;i < cel->element*TRI;i++) fscanf(fp,"%d",&bic_ele[i]);
  fclose(fp);

  // Allocate ---
  cel->n  = N_R;
  cel->gs = GS;
  cel->rg = 1.0;
  cel->br = BR;
  cel->inflation_flag = NUM_FLAG;
  #if(CYTOADHESION == true)
  if (N_W == 0 && N_P == 0 && N_C == 0) {
    cel->adnum = cel->n;
    cel->advertex = cel->vertex;
  }
  #endif

  shape = (int    *)malloc(sizeof(int   )*cel->n); if(shape == NULL) error(0);
  rp    = (double *)malloc(sizeof(double)*cel->n); if(rp    == NULL) error(0);
  gxp   = (double *)malloc(sizeof(double)*cel->n); if(gxp   == NULL) error(0);
  gyp   = (double *)malloc(sizeof(double)*cel->n); if(gyp   == NULL) error(0);
  gzp   = (double *)malloc(sizeof(double)*cel->n); if(gzp   == NULL) error(0);
  alpha = (double *)malloc(sizeof(double)*cel->n); if(alpha == NULL) error(0);
  beta  = (double *)malloc(sizeof(double)*cel->n); if(beta  == NULL) error(0);
  gamma = (double *)malloc(sizeof(double)*cel->n); if(gamma == NULL) error(0);

  cel->rH   = (double *)malloc(sizeof(double)*cel->n                 ); if(cel->rH   == NULL) error(0);
  cel->xH   = (double *)malloc(sizeof(double)*cel->n*cel->vertex     ); if(cel->xH   == NULL) error(0);
  cel->yH   = (double *)malloc(sizeof(double)*cel->n*cel->vertex     ); if(cel->yH   == NULL) error(0);
  cel->zH   = (double *)malloc(sizeof(double)*cel->n*cel->vertex     ); if(cel->zH   == NULL) error(0);
  cel->rxH  = (double *)malloc(sizeof(double)*cel->n                 ); if(cel->rxH  == NULL) error(0);
  cel->ryH  = (double *)malloc(sizeof(double)*cel->n                 ); if(cel->ryH  == NULL) error(0);
  cel->rzH  = (double *)malloc(sizeof(double)*cel->n                 ); if(cel->rzH  == NULL) error(0);
  cel->uH   = (double *)malloc(sizeof(double)*cel->n*cel->vertex     ); if(cel->uH   == NULL) error(0);
  cel->vH   = (double *)malloc(sizeof(double)*cel->n*cel->vertex     ); if(cel->vH   == NULL) error(0);
  cel->wH   = (double *)malloc(sizeof(double)*cel->n*cel->vertex     ); if(cel->wH   == NULL) error(0);
  cel->fxH  = (double *)malloc(sizeof(double)*cel->n*cel->vertex     ); if(cel->fxH  == NULL) error(0);
  cel->fyH  = (double *)malloc(sizeof(double)*cel->n*cel->vertex     ); if(cel->fyH  == NULL) error(0);
  cel->fzH  = (double *)malloc(sizeof(double)*cel->n*cel->vertex     ); if(cel->fzH  == NULL) error(0);
  cel->c0H  = (double *)malloc(sizeof(double)*cel->n*cel->element*TRI); if(cel->c0H  == NULL) error(0);
  cel->s0H  = (double *)malloc(sizeof(double)*cel->n*cel->element*TRI); if(cel->s0H  == NULL) error(0);
  cel->eleH = (int    *)malloc(sizeof(int   )*cel->n*cel->element*TRI); if(cel->eleH == NULL) error(0);
  cel->jcH  = (int    *)malloc(sizeof(int   )*cel->n*cel->vertex*HEX ); if(cel->jcH  == NULL) error(0);
  cel->jcnH = (int    *)malloc(sizeof(int   )*cel->n*cel->vertex     ); if(cel->jcnH == NULL) error(0);
  cel->ecH  = (int    *)malloc(sizeof(int   )*cel->n*cel->element*TRI); if(cel->ecH  == NULL) error(0);

// Initial displacement of RBCs
///*
  srand((unsigned int)time(NULL));
  double tminp = 0.0;          // tminp = 0.0 [deg]
  double tmaxp = 2.0*M_PI;     // tmaxp = 360 [deg]

  shape[0] = SPHERE;  
  rp[0]    = CR;
  gxp[0]   = CX;
  gyp[0]   = CY;
  gzp[0]   = CZ;   
//  alpha[0] = CA;      
//  beta[0]  = CB;      
//  gamma[0] = CG;       
  alpha[0] = (double)(rand()*(tmaxp-tminp)/RAND_MAX);
  beta[0]  = (double)(rand()*(tmaxp-tminp)/RAND_MAX);
  gamma[0] = (double)(rand()*(tmaxp-tminp)/RAND_MAX);
//*/
/*
  srand((unsigned int)time(NULL));
  int    flag1, flag2;
  double dx0, dx1, randr, randt, rminp, rmaxp, tminp, tmaxp;

  rminp = cdo->dx*24.0; // rminp = 6.00[um]
  rmaxp = cdo->dx*82.0; // rmaxp = 20.5[um]
  tminp = 0.0;          // tminp = 0.0 [deg]
  tmaxp = 2.0*M_PI;     // tmaxp = 360 [deg]

// Initial displacement of RBCs
  for (ic = 0; ic < cel->n; ic++) {

    flag1 = ic%4;
    flag2 = ic/4;
    dx0   = cdo->dx*32.0; // 8.00 [um]
    dx1   = cdo->dx*64.0; // 16.0 [um]
    randr = rminp + (double)(rand()*(rmaxp-rminp)/RAND_MAX);

    shape[ic] = BICONCAVE;
    rp[ic]    = CR;
    alpha[ic] = (double)(rand()*(tmaxp-tminp)/RAND_MAX);
    beta[ic]  = (double)(rand()*(tmaxp-tminp)/RAND_MAX);
    gamma[ic] = (double)(rand()*(tmaxp-tminp)/RAND_MAX);

    if (flag1 == 0) {
      randt   = tminp + (double)(rand()*(tmaxp-tminp)/RAND_MAX);
      gxp[ic] = randr*cos(randt + (double)flag1*M_PI_4);
      gyp[ic] = randr*sin(randt + (double)flag1*M_PI_4);
      gzp[ic] = -LZ/2.0 + dx0 + (double)flag2*dx1;
      if (cdo->restart == 1) continue;

    } else {
      gxp[ic] = randr*cos(randt + (double)flag1*M_PI_4);
      gyp[ic] = randr*sin(randt + (double)flag1*M_PI_4);
      gzp[ic] = -LZ/2.0 + dx0 + (double)flag2*dx1;
      if (cdo->restart == 1) continue;
    }

    //printf("RBC %4d: position %10.3e\n",ic,gzp[ic]);
    if (gzp[ic] > LZ/2.0) error(8);
  }
*/

  // Radius, coordinate, velocity, stress, vertex number of triangular element ---
  for(ic = 0;ic < cel->n;ic++){
    cel->rH[ic] = rp[ic];

    for(j = 0;j < cel->vertex;j++){
      if(shape[ic] == SPHERE){
        tx = sph_x[j];
        ty = sph_y[j];
        tz = sph_z[j];
      }
      else if(shape[ic] == BICONCAVE){
        tx = bic_x[j];
        ty = bic_y[j];
        tz = bic_z[j];
      }
      else break;

      jid = ic*cel->vertex + j;

      cel->xH[jid] = tx;
      cel->yH[jid] = ty*cos(alpha[ic]) - tz*sin(alpha[ic]);
      cel->zH[jid] = ty*sin(alpha[ic]) + tz*cos(alpha[ic]);
      tz = cel->zH[jid];
      tx = cel->xH[jid];
      ty = cel->yH[jid];
      cel->zH[jid] = tz*cos(beta[ic]) - tx*sin(beta[ic]);
      cel->xH[jid] = tz*sin(beta[ic]) + tx*cos(beta[ic]);
      cel->yH[jid] = ty;
      tx = cel->xH[jid];
      ty = cel->yH[jid];
      tz = cel->zH[jid];
      cel->xH[jid] = tx*cos(gamma[ic]) - ty*sin(gamma[ic]);
      cel->yH[jid] = tx*sin(gamma[ic]) + ty*cos(gamma[ic]);
      cel->zH[jid] = tz;
      cel->xH[jid] = cel->xH[jid]*rp[ic] + gxp[ic];
      cel->yH[jid] = cel->yH[jid]*rp[ic] + gyp[ic];
      cel->zH[jid] = cel->zH[jid]*rp[ic] + gzp[ic];

      cel->uH[jid]  = 0.0;
      cel->vH[jid]  = 0.0;
      cel->wH[jid]  = 0.0;
      cel->fxH[jid] = 0.0;
      cel->fyH[jid] = 0.0;
      cel->fzH[jid] = 0.0;
    }

    for(e = 0;e < cel->element;e++){
      if(shape[ic] == SPHERE){
        j1 = sph_ele[e*TRI  ] + ic*cel->vertex;
        j2 = sph_ele[e*TRI+1] + ic*cel->vertex;
        j3 = sph_ele[e*TRI+2] + ic*cel->vertex;
      }
      else if(shape[ic] == BICONCAVE){
        j1 = bic_ele[e*TRI  ] + ic*cel->vertex;
        j2 = bic_ele[e*TRI+1] + ic*cel->vertex;
        j3 = bic_ele[e*TRI+2] + ic*cel->vertex;
      }
      else break;

      eid = ic*cel->element + e;

      cel->eleH[eid*TRI  ] = j1;
      cel->eleH[eid*TRI+1] = j2;
      cel->eleH[eid*TRI+2] = j3;
    }
  }
  // Vertex connectivity ---
  for(ic = 0;ic < cel->n;ic++){
    for(j = 0;j < cel->vertex;j++){
      jid = ic*cel->vertex + j;
      cel->jcnH[jid]      = 0;
      cel->jcH[jid*HEX  ] = -1;
      cel->jcH[jid*HEX+1] = -1;
      cel->jcH[jid*HEX+2] = -1;
      cel->jcH[jid*HEX+3] = -1;
      cel->jcH[jid*HEX+4] = -1;
      cel->jcH[jid*HEX+5] = -1;

      for(e = 0;e < cel->element;e++){
        eid = ic*cel->element + e;
        j1 = cel->eleH[eid*TRI  ];
        j2 = cel->eleH[eid*TRI+1];
        j3 = cel->eleH[eid*TRI+2];

        if(jid == j1){
          for(k = 0;k < HEX;k++){
            if(j2 == cel->jcH[jid*HEX+k]) j2 = -1;
            if(j3 == cel->jcH[jid*HEX+k]) j3 = -1;
          }
          if(j2 != -1){
            cel->jcH[jid*HEX + cel->jcnH[jid]] = j2;
            cel->jcnH[jid]++;
          }
          if(j3 != -1){
            cel->jcH[jid*HEX + cel->jcnH[jid]] = j3;
            cel->jcnH[jid]++;
          }
        }
        else if(jid == j2){
          for(k = 0;k < HEX;k++){
            if(j3 == cel->jcH[jid*HEX+k]) j3 = -1;
            if(j1 == cel->jcH[jid*HEX+k]) j1 = -1;
          }
          if(j3 != -1){
            cel->jcH[jid*HEX + cel->jcnH[jid]] = j3;
            cel->jcnH[jid]++;
          }
          if(j1 != -1){
            cel->jcH[jid*HEX + cel->jcnH[jid]] = j1;
            cel->jcnH[jid]++;
          }
        }
        else if(jid == j3){
          for(k = 0;k < HEX;k++){
            if(j1 == cel->jcH[jid*HEX+k]) j1 = -1;
            if(j2 == cel->jcH[jid*HEX+k]) j2 = -1;
          }
          if(j1 != -1){
            cel->jcH[jid*HEX + cel->jcnH[jid]] = j1;
            cel->jcnH[jid]++;
          }
          if(j2 != -1){
            cel->jcH[jid*HEX + cel->jcnH[jid]] = j2;
            cel->jcnH[jid]++;
          }
        }
        else continue;
      }
    }
  }
  // Element connectivity ---
  for(ic = 0;ic < cel->n;ic++){
    for(e = 0;e < cel->element;e++){
      ecn = 0;
      e1 = ic*cel->element + e;
      i1 = cel->eleH[e1*TRI  ];
      i2 = cel->eleH[e1*TRI+1];
      i3 = cel->eleH[e1*TRI+2];

      for(k = 0;k < cel->element;k++){
        if(e == k) continue;
        e2 = ic*cel->element + k;
        j1 = cel->eleH[e2*TRI  ];
        j2 = cel->eleH[e2*TRI+1];
        j3 = cel->eleH[e2*TRI+2];

        if((i1 == j1 && i2 == j2)
            || (i1 == j1 && i2 == j3)
            || (i1 == j2 && i2 == j3)
            || (i1 == j2 && i2 == j1)
            || (i1 == j3 && i2 == j1)
            || (i1 == j3 && i2 == j2)
            || (i2 == j1 && i3 == j2)
            || (i2 == j1 && i3 == j3)
            || (i2 == j2 && i3 == j3)
            || (i2 == j2 && i3 == j1)
            || (i2 == j3 && i3 == j1)
            || (i2 == j3 && i3 == j2)
            || (i3 == j1 && i1 == j2)
            || (i3 == j1 && i1 == j3)
            || (i3 == j2 && i1 == j3)
            || (i3 == j2 && i1 == j1)
            || (i3 == j3 && i1 == j1)
            || (i3 == j3 && i1 == j2)){
          cel->ecH[e1*TRI+ecn] = e2;
          ecn++;
        }
        else continue;
      }
    }
  }

  // Reference of bending ---
  for(ic = 0;ic < cel->n;ic++){
    for(e = 0;e < cel->element;e++){
      e1 = ic*cel->element + e;
      i1 = cel->eleH[e1*TRI  ];
      i2 = cel->eleH[e1*TRI+1];
      i3 = cel->eleH[e1*TRI+2];

      for(k = 0;k < TRI;k++){
        e2 = cel->ecH[e1*TRI+k];
        j1 = cel->eleH[e2*TRI  ];
        j2 = cel->eleH[e2*TRI+1];
        j3 = cel->eleH[e2*TRI+2];

        if     ((i1==j1 && i2==j2)||(i1==j2 && i2==j1)){ k1 = i3; k2 = i1; k3 = i2; k4 = j3; }
        else if((i1==j2 && i2==j3)||(i1==j3 && i2==j2)){ k1 = i3; k2 = i1; k3 = i2; k4 = j1; }
        else if((i1==j3 && i2==j1)||(i1==j1 && i2==j3)){ k1 = i3; k2 = i1; k3 = i2; k4 = j2; }
        else if((i2==j1 && i3==j2)||(i2==j2 && i3==j1)){ k1 = i1; k2 = i2; k3 = i3; k4 = j3; }
        else if((i2==j2 && i3==j3)||(i2==j3 && i3==j2)){ k1 = i1; k2 = i2; k3 = i3; k4 = j1; }
        else if((i2==j3 && i3==j1)||(i2==j1 && i3==j3)){ k1 = i1; k2 = i2; k3 = i3; k4 = j2; }
        else if((i3==j1 && i1==j2)||(i3==j2 && i1==j1)){ k1 = i2; k2 = i3; k3 = i1; k4 = j3; }
        else if((i3==j2 && i1==j3)||(i3==j3 && i1==j2)){ k1 = i2; k2 = i3; k3 = i1; k4 = j1; }
        else if((i3==j3 && i1==j1)||(i3==j1 && i1==j3)){ k1 = i2; k2 = i3; k3 = i1; k4 = j2; }

        a1x = cel->xH[k1]; a2x = cel->xH[k2]; a3x = cel->xH[k3]; a4x = cel->xH[k4];
        a1y = cel->yH[k1]; a2y = cel->yH[k2]; a3y = cel->yH[k3]; a4y = cel->yH[k4];
        a1z = cel->zH[k1]; a2z = cel->zH[k2]; a3z = cel->zH[k3]; a4z = cel->zH[k4];

        a21x = a2x - a1x; a31x = a3x - a1x; a34x = a3x - a4x; a24x = a2x - a4x;
        a21y = a2y - a1y; a31y = a3y - a1y; a34y = a3y - a4y; a24y = a2y - a4y;
        a21z = a2z - a1z; a31z = a3z - a1z; a34z = a3z - a4z; a24z = a2z - a4z;

        xix = a21y*a31z - a21z*a31y; zex = a34y*a24z - a34z*a24y;
        xiy = a21z*a31x - a21x*a31z; zey = a34z*a24x - a34x*a24z;
        xiz = a21x*a31y - a21y*a31x; zez = a34x*a24y - a34y*a24x;

        t1x = (a1x + a2x + a3x)/3.0; t2x = (a4x + a2x + a3x)/3.0;
        t1y = (a1y + a2y + a3y)/3.0; t2y = (a4y + a2y + a3y)/3.0;
        t1z = (a1z + a2z + a3z)/3.0; t2z = (a4z + a2z + a3z)/3.0;

        cost = (xix*zex + xiy*zey + xiz*zez) 
          /sqrt(xix*xix + xiy*xiy + xiz*xiz)
          /sqrt(zex*zex + zey*zey + zez*zez);
        cost2 = 1.0 - cost*cost; if(cost2 < 0.0) cost2 = 0.0;

        if(((xix-zex)*(t1x-t2x)
              +(xiy-zey)*(t1y-t2y)
              +(xiz-zez)*(t1z-t2z)) >= 0.0) sint =  sqrt(cost2);
        else                                sint = -sqrt(cost2);

        cel->c0H[e1*TRI + k] = cost;
        cel->s0H[e1*TRI + k] = sint;
        cel->c0H[e1*TRI + k] = 1.0;
        cel->s0H[e1*TRI + k] = 0.0;
      }
    }
  }

  free(sph_x); free(sph_y); free(sph_z); free(sph_ele);
  free(bic_x); free(bic_y); free(bic_z); free(bic_ele);
  free(shape); free(rp); free(gxp); free(gyp); free(gzp);
  free(alpha); free(beta); free(gamma);

  // Numerical condition ---
  printf("====================================================================\n");
  printf("  Number of RBC             : %d\n",N_R);
  printf("  Radius of RBC             : %9.3e [um]\n",RA*1.0e+06);
  printf("  Viscosity ratio           : %9.3e [-]\n",LAMBDA1);
  printf("  Surface shear elasticity  : %9.3e [N/m]\n",GS);
  printf("  Average bending stiffness : %9.3e [J]\n",KC);
  printf("====================================================================\n");

  return;
  #endif // N_R
#endif // CapsuleSHEARFLOW || CapsuleCHANNELFLOW
}


void  mem_init
//==========================================================
//
//  INITIAL SETTING OF DEVICE MEMORY
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
  // Initialize device memory ---
  printf("Initializing device memory -----------------------------------------\n");

  // Allocate ---
  printf("  Allocating...\n");

  checkCudaErrors(cudaMalloc((void **)&(ltc->fnD  ),sizeof(double)*cdo->n*Q               ));
  checkCudaErrors(cudaMalloc((void **)&(ltc->fmD  ),sizeof(double)*cdo->n*Q               ));
  checkCudaErrors(cudaMalloc((void **)&(ltc->dnD  ),sizeof(double)*cdo->n                 ));
  checkCudaErrors(cudaMalloc((void **)&(ltc->unD  ),sizeof(double)*cdo->n                 ));
  checkCudaErrors(cudaMalloc((void **)&(ltc->vnD  ),sizeof(double)*cdo->n                 ));
  checkCudaErrors(cudaMalloc((void **)&(ltc->wnD  ),sizeof(double)*cdo->n                 ));
  checkCudaErrors(cudaMalloc((void **)&(ltc->dmD  ),sizeof(double)*cdo->n                 ));
  checkCudaErrors(cudaMalloc((void **)&(ltc->umD  ),sizeof(double)*cdo->n                 ));
  checkCudaErrors(cudaMalloc((void **)&(ltc->vmD  ),sizeof(double)*cdo->n                 ));
  checkCudaErrors(cudaMalloc((void **)&(ltc->wmD  ),sizeof(double)*cdo->n                 ));
  checkCudaErrors(cudaMalloc((void **)&(ltc->fxD  ),sizeof(double)*cdo->n                 ));
  checkCudaErrors(cudaMalloc((void **)&(ltc->fyD  ),sizeof(double)*cdo->n                 ));
  checkCudaErrors(cudaMalloc((void **)&(ltc->fzD  ),sizeof(double)*cdo->n                 ));
  checkCudaErrors(cudaMalloc((void **)&(ltc->vfD  ),sizeof(double)*cdo->n                 ));
  checkCudaErrors(cudaMalloc((void **)&(ltc->vfD2  ),sizeof(double)*cdo->n                 ));
  checkCudaErrors(cudaMalloc((void **)&(ltc->bcD  ),sizeof(int   )*cdo->n                 ));

  // Memcpy for variables of fluid ---
  checkCudaErrors(cudaMemcpy(ltc->fnD,  ltc->fnH,  sizeof(double)*cdo->n*Q, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->fmD,  ltc->fmH,  sizeof(double)*cdo->n*Q, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->dnD,  ltc->dnH,  sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->unD,  ltc->unH,  sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->vnD,  ltc->vnH,  sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->wnD,  ltc->wnH,  sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->dmD,  ltc->dmH,  sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->umD,  ltc->umH,  sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->vmD,  ltc->vmH,  sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->wmD,  ltc->wmH,  sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->fxD,  ltc->fxH,  sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->fyD,  ltc->fyH,  sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->fzD,  ltc->fzH,  sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->vfD,  ltc->vfH,  sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->vfD2, ltc->vfH2, sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->bcD,  ltc->bcH,  sizeof(int   )*cdo->n,   cudaMemcpyHostToDevice));


#if defined(CapsuleSHEARFLOW) || defined(CapsuleCHANNELFLOW)
  #if(N_R > 0)
  checkCudaErrors(cudaMalloc((void **)&(cel->rD     ),sizeof(double)*cel->n                        ));
  checkCudaErrors(cudaMalloc((void **)&(cel->xD     ),sizeof(double)*cel->n*cel->vertex            ));
  checkCudaErrors(cudaMalloc((void **)&(cel->yD     ),sizeof(double)*cel->n*cel->vertex            ));
  checkCudaErrors(cudaMalloc((void **)&(cel->zD     ),sizeof(double)*cel->n*cel->vertex            ));
  checkCudaErrors(cudaMalloc((void **)&(cel->rxD    ),sizeof(double)*cel->n                        ));
  checkCudaErrors(cudaMalloc((void **)&(cel->ryD    ),sizeof(double)*cel->n                        ));
  checkCudaErrors(cudaMalloc((void **)&(cel->rzD    ),sizeof(double)*cel->n                        ));
  checkCudaErrors(cudaMalloc((void **)&(cel->uD     ),sizeof(double)*cel->n*cel->vertex            ));
  checkCudaErrors(cudaMalloc((void **)&(cel->vD     ),sizeof(double)*cel->n*cel->vertex            ));
  checkCudaErrors(cudaMalloc((void **)&(cel->wD     ),sizeof(double)*cel->n*cel->vertex            ));
  checkCudaErrors(cudaMalloc((void **)&(cel->fxD    ),sizeof(double)*cel->n*cel->vertex            ));
  checkCudaErrors(cudaMalloc((void **)&(cel->fyD    ),sizeof(double)*cel->n*cel->vertex            ));
  checkCudaErrors(cudaMalloc((void **)&(cel->fzD    ),sizeof(double)*cel->n*cel->vertex            ));
  checkCudaErrors(cudaMalloc((void **)&(cel->c0D    ),sizeof(double)*cel->n*cel->element*TRI       ));
  checkCudaErrors(cudaMalloc((void **)&(cel->s0D    ),sizeof(double)*cel->n*cel->element*TRI       ));
  checkCudaErrors(cudaMalloc((void **)&(cel->eleD   ),sizeof(int   )*cel->n*cel->element*TRI       ));
  checkCudaErrors(cudaMalloc((void **)&(cel->jcD    ),sizeof(int   )*cel->n*cel->vertex*HEX        ));
  checkCudaErrors(cudaMalloc((void **)&(cel->jcnD   ),sizeof(int   )*cel->n*cel->vertex            ));
  checkCudaErrors(cudaMalloc((void **)&(cel->ecD    ),sizeof(int   )*cel->n*cel->element*TRI       ));

  checkCudaErrors(cudaMalloc((void **)&(fem->xd     ),sizeof(double)*cel->n_th*3                   ));
  checkCudaErrors(cudaMalloc((void **)&(fem->xrd    ),sizeof(double)*cel->n_th*3                   ));
  checkCudaErrors(cudaMalloc((void **)&(fem->qd     ),sizeof(double)*cel->n*cel->mn                ));
  checkCudaErrors(cudaMalloc((void **)&(fem->qid    ),sizeof(double)*cel->n*cel->mn                ));
  checkCudaErrors(cudaMalloc((void **)&(fem->td     ),sizeof(double)*cel->n*cel->element*4         ));
  checkCudaErrors(cudaMalloc((void **)&(fem->tpd    ),sizeof(double)*cel->n*cel->element*2         ));
  checkCudaErrors(cudaMalloc((void **)&(fem->rgd    ),sizeof(double)*cel->n*cel->element           ));
  checkCudaErrors(cudaMalloc((void **)&(fem->nd     ),sizeof(double)*cel->n*cel->element*4         ));
  checkCudaErrors(cudaMalloc((void **)&(fem->eled   ),sizeof(int   )*cel->n*cel->element*6         ));
  checkCudaErrors(cudaMalloc((void **)&(fem->nlnd   ),sizeof(int   )*cel->n_th                     ));
  checkCudaErrors(cudaMalloc((void **)&(fem->lnd    ),sizeof(int   )*cel->n_th*7                   ));
  checkCudaErrors(cudaMalloc((void **)&(fem->lod    ),sizeof(int   )*cel->n_th*7                   ));
  checkCudaErrors(cudaMalloc((void **)&(fem->nled   ),sizeof(int   )*cel->n_th                     ));
  checkCudaErrors(cudaMalloc((void **)&(fem->led    ),sizeof(int   )*cel->n_th*6                   ));
  checkCudaErrors(cudaMalloc((void **)&(fem->ptrd   ),sizeof(int   )*cel->mn                       ));
  checkCudaErrors(cudaMalloc((void **)&(fem->indexd ),sizeof(int   )*cel->mn*7                     ));
  checkCudaErrors(cudaMalloc((void **)&(fem->valued ),sizeof(double)*cel->mn*7                     ));
  checkCudaErrors(cudaMalloc((void **)&(fem->bd     ),sizeof(double)*cel->mn                       ));

  // Memcpy for variables of RBC ---
  checkCudaErrors(cudaMemcpy(cel->rD     ,cel->rH     ,sizeof(double)*cel->n                        ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->xD     ,cel->xH     ,sizeof(double)*cel->n*cel->vertex            ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->yD     ,cel->yH     ,sizeof(double)*cel->n*cel->vertex            ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->zD     ,cel->zH     ,sizeof(double)*cel->n*cel->vertex            ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->rxD    ,cel->rxH    ,sizeof(double)*cel->n                        ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->ryD    ,cel->ryH    ,sizeof(double)*cel->n                        ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->rzD    ,cel->rzH    ,sizeof(double)*cel->n                        ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->uD     ,cel->uH     ,sizeof(double)*cel->n*cel->vertex            ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->vD     ,cel->vH     ,sizeof(double)*cel->n*cel->vertex            ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->wD     ,cel->wH     ,sizeof(double)*cel->n*cel->vertex            ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->fxD    ,cel->fxH    ,sizeof(double)*cel->n*cel->vertex            ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->fyD    ,cel->fyH    ,sizeof(double)*cel->n*cel->vertex            ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->fzD    ,cel->fzH    ,sizeof(double)*cel->n*cel->vertex            ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->c0D    ,cel->c0H    ,sizeof(double)*cel->n*cel->element*TRI       ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->s0D    ,cel->s0H    ,sizeof(double)*cel->n*cel->element*TRI       ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->eleD   ,cel->eleH   ,sizeof(int   )*cel->n*cel->element*TRI       ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->jcD    ,cel->jcH    ,sizeof(int   )*cel->n*cel->vertex*HEX        ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->jcnD   ,cel->jcnH   ,sizeof(int   )*cel->n*cel->vertex            ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->ecD    ,cel->ecH    ,sizeof(int   )*cel->n*cel->element*TRI       ,cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(fem->xd     ,fem->x      ,sizeof(double)*cel->n_th*3                   ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->xrd     ,fem->xr    ,sizeof(double)*cel->n_th*3                   ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->qd     ,fem->q      ,sizeof(double)*cel->n*cel->mn                ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->qid    ,fem->qi     ,sizeof(double)*cel->n*cel->mn                ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->td     ,fem->t      ,sizeof(double)*cel->n*cel->element*4         ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->tpd    ,fem->tp     ,sizeof(double)*cel->n*cel->element*2         ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->nd     ,fem->n      ,sizeof(double)*cel->n*cel->element*4         ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->eled   ,fem->ele    ,sizeof(int   )*cel->n*cel->element*6         ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->nlnd   ,fem->nln    ,sizeof(int   )*cel->n_th                     ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->lnd    ,fem->ln     ,sizeof(int   )*cel->n_th*7                   ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->lod    ,fem->lo     ,sizeof(int   )*cel->n_th*7                   ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->nled   ,fem->nle    ,sizeof(int   )*cel->n_th                     ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->led    ,fem->le     ,sizeof(int   )*cel->n_th*6                   ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->ptrd   ,fem->ptr    ,sizeof(int   )*cel->mn                       ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->indexd ,fem->index  ,sizeof(int   )*cel->mn*7                     ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->valued ,fem->value  ,sizeof(double)*cel->mn*7                     ,cudaMemcpyHostToDevice));
  #endif // N_R

  #if (N_W > 0)
  checkCudaErrors(cudaMalloc((void **)&(cel->rD_w  ),sizeof(double)*cel->n_w                   ));
  checkCudaErrors(cudaMalloc((void **)&(cel->xD_w  ),sizeof(double)*cel->n_w*cel->vertex_w     ));
  checkCudaErrors(cudaMalloc((void **)&(cel->yD_w  ),sizeof(double)*cel->n_w*cel->vertex_w     ));
  checkCudaErrors(cudaMalloc((void **)&(cel->zD_w  ),sizeof(double)*cel->n_w*cel->vertex_w     ));
  checkCudaErrors(cudaMalloc((void **)&(cel->rxD_w ),sizeof(double)*cel->n_w                   ));
  checkCudaErrors(cudaMalloc((void **)&(cel->ryD_w ),sizeof(double)*cel->n_w                   ));
  checkCudaErrors(cudaMalloc((void **)&(cel->rzD_w ),sizeof(double)*cel->n_w                   ));
  checkCudaErrors(cudaMalloc((void **)&(cel->uD_w  ),sizeof(double)*cel->n_w*cel->vertex_w     ));
  checkCudaErrors(cudaMalloc((void **)&(cel->vD_w  ),sizeof(double)*cel->n_w*cel->vertex_w     ));
  checkCudaErrors(cudaMalloc((void **)&(cel->wD_w  ),sizeof(double)*cel->n_w*cel->vertex_w     ));
  checkCudaErrors(cudaMalloc((void **)&(cel->fxD_w ),sizeof(double)*cel->n_w*cel->vertex_w     ));
  checkCudaErrors(cudaMalloc((void **)&(cel->fyD_w ),sizeof(double)*cel->n_w*cel->vertex_w     ));
  checkCudaErrors(cudaMalloc((void **)&(cel->fzD_w ),sizeof(double)*cel->n_w*cel->vertex_w     ));
  checkCudaErrors(cudaMalloc((void **)&(cel->c0D_w ),sizeof(double)*cel->n_w*cel->element_w*TRI));
  checkCudaErrors(cudaMalloc((void **)&(cel->s0D_w ),sizeof(double)*cel->n_w*cel->element_w*TRI));
  checkCudaErrors(cudaMalloc((void **)&(cel->eleD_w),sizeof(int   )*cel->n_w*cel->element_w*TRI));
  checkCudaErrors(cudaMalloc((void **)&(cel->jcD_w ),sizeof(int   )*cel->n_w*cel->vertex_w*HEX ));
  checkCudaErrors(cudaMalloc((void **)&(cel->jcnD_w),sizeof(int   )*cel->n_w*cel->vertex_w     ));
  checkCudaErrors(cudaMalloc((void **)&(cel->ecD_w ),sizeof(int   )*cel->n_w*cel->element_w*TRI));

  checkCudaErrors(cudaMalloc((void **)&(fem->xd_w    ),sizeof(double)*cel->n_th_w*3            ));
  checkCudaErrors(cudaMalloc((void **)&(fem->xrd_w   ),sizeof(double)*cel->n_th_w*3            ));
  checkCudaErrors(cudaMalloc((void **)&(fem->qd_w    ),sizeof(double)*cel->n_w*cel->mn_w       ));
  checkCudaErrors(cudaMalloc((void **)&(fem->qid_w   ),sizeof(double)*cel->n_w*cel->mn_w       ));
  checkCudaErrors(cudaMalloc((void **)&(fem->td_w    ),sizeof(double)*cel->n_w*cel->element_w*4));
  checkCudaErrors(cudaMalloc((void **)&(fem->tpd_w   ),sizeof(double)*cel->n_w*cel->element_w*2));
  checkCudaErrors(cudaMalloc((void **)&(fem->rgd_w   ),sizeof(double)*cel->n_w*cel->element_w  ));
  checkCudaErrors(cudaMalloc((void **)&(fem->nd_w    ),sizeof(double)*cel->n_w*cel->element_w*4));
  checkCudaErrors(cudaMalloc((void **)&(fem->eled_w  ),sizeof(int   )*cel->n_w*cel->element_w*6));
  checkCudaErrors(cudaMalloc((void **)&(fem->nlnd_w  ),sizeof(int   )*cel->n_th_w              ));
  checkCudaErrors(cudaMalloc((void **)&(fem->lnd_w   ),sizeof(int   )*cel->n_th_w*7            ));
  checkCudaErrors(cudaMalloc((void **)&(fem->lod_w   ),sizeof(int   )*cel->n_th_w*7            ));
  checkCudaErrors(cudaMalloc((void **)&(fem->nled_w  ),sizeof(int   )*cel->n_th_w              ));
  checkCudaErrors(cudaMalloc((void **)&(fem->led_w   ),sizeof(int   )*cel->n_th_w*6            ));
  checkCudaErrors(cudaMalloc((void **)&(fem->ptrd_w  ),sizeof(int   )*cel->mn_w                ));
  checkCudaErrors(cudaMalloc((void **)&(fem->indexd_w),sizeof(int   )*cel->mn_w*7              ));
  checkCudaErrors(cudaMalloc((void **)&(fem->valued_w),sizeof(double)*cel->mn_w*7              ));
  checkCudaErrors(cudaMalloc((void **)&(fem->bd_w    ),sizeof(double)*cel->mn_w                ));

  // Memcpy for variables of WBC ---
  checkCudaErrors(cudaMemcpy(cel->rD_w  ,cel->rH_w  ,sizeof(double)*cel->n_w                   ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->xD_w  ,cel->xH_w  ,sizeof(double)*cel->n_w*cel->vertex_w     ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->yD_w  ,cel->yH_w  ,sizeof(double)*cel->n_w*cel->vertex_w     ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->zD_w  ,cel->zH_w  ,sizeof(double)*cel->n_w*cel->vertex_w     ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->rxD_w ,cel->rxH_w ,sizeof(double)*cel->n_w                   ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->ryD_w ,cel->ryH_w ,sizeof(double)*cel->n_w                   ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->rzD_w ,cel->rzH_w ,sizeof(double)*cel->n_w                   ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->uD_w  ,cel->uH_w  ,sizeof(double)*cel->n_w*cel->vertex_w     ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->vD_w  ,cel->vH_w  ,sizeof(double)*cel->n_w*cel->vertex_w     ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->wD_w  ,cel->wH_w  ,sizeof(double)*cel->n_w*cel->vertex_w     ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->fxD_w ,cel->fxH_w ,sizeof(double)*cel->n_w*cel->vertex_w     ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->fyD_w ,cel->fyH_w ,sizeof(double)*cel->n_w*cel->vertex_w     ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->fzD_w ,cel->fzH_w ,sizeof(double)*cel->n_w*cel->vertex_w     ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->c0D_w ,cel->c0H_w ,sizeof(double)*cel->n_w*cel->element_w*TRI,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->s0D_w ,cel->s0H_w ,sizeof(double)*cel->n_w*cel->element_w*TRI,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->eleD_w,cel->eleH_w,sizeof(int   )*cel->n_w*cel->element_w*TRI,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->jcD_w ,cel->jcH_w ,sizeof(int   )*cel->n_w*cel->vertex_w*HEX ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->jcnD_w,cel->jcnH_w,sizeof(int   )*cel->n_w*cel->vertex_w     ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->ecD_w ,cel->ecH_w ,sizeof(int   )*cel->n_w*cel->element_w*TRI,cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMemcpy(fem->xd_w    ,fem->x_w    ,sizeof(double)*cel->n_th_w*3            ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->xrd_w   ,fem->xr_w   ,sizeof(double)*cel->n_th_w*3            ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->qd_w    ,fem->q_w    ,sizeof(double)*cel->n_w*cel->mn_w       ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->qid_w   ,fem->qi_w   ,sizeof(double)*cel->n_w*cel->mn_w       ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->td_w    ,fem->t_w    ,sizeof(double)*cel->n_w*cel->element_w*4,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->tpd_w   ,fem->tp_w   ,sizeof(double)*cel->n_w*cel->element_w*2,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->nd_w    ,fem->n_w    ,sizeof(double)*cel->n_w*cel->element_w*4,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->eled_w  ,fem->ele_w  ,sizeof(int   )*cel->n_w*cel->element_w*6,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->nlnd_w  ,fem->nln_w  ,sizeof(int   )*cel->n_th_w              ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->lnd_w   ,fem->ln_w   ,sizeof(int   )*cel->n_th_w*7            ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->lod_w   ,fem->lo_w   ,sizeof(int   )*cel->n_th_w*7            ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->nled_w  ,fem->nle_w  ,sizeof(int   )*cel->n_th_w              ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->led_w   ,fem->le_w   ,sizeof(int   )*cel->n_th_w*6            ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->ptrd_w  ,fem->ptr_w  ,sizeof(int   )*cel->mn_w                ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->indexd_w,fem->index_w,sizeof(int   )*cel->mn_w*7              ,cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->valued_w,fem->value_w,sizeof(double)*cel->mn_w*7              ,cudaMemcpyHostToDevice));
  #endif // N_W
#endif // CapsuleSHEARFLOW || CapsuleCHANNELFLOW

  return;
}


void  vof_init
//==========================================================
//
//  INITIAL SETTING OF VOLUME FRACTION
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
  #if (CN == 0)
  return;
  #else
  // Initialize volume fraction ---
  printf("Initializing volume fraction ---------------------------------------\n");

  // front-tracking method ---
  front_tracking(cdo,ltc,cel);

  return;
  #endif // CN
#endif // CapsuleSHEARFLOW || CapsuleCHANNELFLOW
}

