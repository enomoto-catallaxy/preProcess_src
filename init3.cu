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

void  cel_init_wbc
//==========================================================
//
//  INITIAL SETTING OF PLATELET CELL
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
  #if(N_W == 0)
  cel->n_w = 0;
  cel->vertex_w = 0;
  cel->element_w = 0;
  return;
  #elif(N_W > 0)

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
  printf("\n");
  printf("Initializing WBC ---------------------------------------------------\n");

// Input cell file ---
//--------------------------------------------------------------------------------------
  sprintf(filename,"./cell/icosa_div/icosa_div%d_cal.dat",ICOSA2); // Read sphere shape
  fp = fopen(filename,"r");
  if(fp == NULL) error(1);

  fscanf(fp,"%d %d",&cel->vertex_w,&cel->element_w);

  sph_x   = (double *)malloc(sizeof(double)*cel->vertex_w     ); if(sph_x   == NULL) error(0);
  sph_y   = (double *)malloc(sizeof(double)*cel->vertex_w     ); if(sph_y   == NULL) error(0);
  sph_z   = (double *)malloc(sizeof(double)*cel->vertex_w     ); if(sph_z   == NULL) error(0);
  sph_ele = (int    *)malloc(sizeof(int   )*cel->element_w*TRI); if(sph_ele == NULL) error(0);

  for(i = 0;i < cel->vertex_w;     i++) fscanf(fp,"%lf %lf %lf",&sph_x[i],&sph_y[i],&sph_z[i]);
  for(i = 0;i < cel->element_w*TRI;i++) fscanf(fp,"%d",&sph_ele[i]);
  fclose(fp);
//--------------------------------------------------------------------------------------

//--------------------------------------------------------------------------------------
  sprintf(filename,"./cell/rbc_div/rbc_div%d_cal.dat",ICOSA2); // Read biconcave shape
  fp = fopen(filename,"r");
  if(fp == NULL) error(1);

  fscanf(fp,"%d %d",&cel->vertex_w,&cel->element_w);

  bic_x   = (double *)malloc(sizeof(double)*cel->vertex_w     ); if(bic_x   == NULL) error(0);
  bic_y   = (double *)malloc(sizeof(double)*cel->vertex_w     ); if(bic_y   == NULL) error(0);
  bic_z   = (double *)malloc(sizeof(double)*cel->vertex_w     ); if(bic_z   == NULL) error(0);
  bic_ele = (int    *)malloc(sizeof(int   )*cel->element_w*TRI); if(bic_ele == NULL) error(0);

  for(i = 0;i < cel->vertex_w;     i++) fscanf(fp,"%lf %lf %lf",&bic_x[i],&bic_y[i],&bic_z[i]);
  for(i = 0;i < cel->element_w*TRI;i++) fscanf(fp,"%d",&bic_ele[i]);
  fclose(fp);
//--------------------------------------------------------------------------------------

// Allocate ---
  cel->n_w  = N_W;
  cel->gs_w = GSw;
  cel->rg_w = GSw/GS;
  cel->br_w = BRw;
  #if(CYTOADHESION == true)
  cel->adnum = cel->n_w;
  cel->advertex = cel->vertex_w;
  #endif

  shape = (int    *)malloc(sizeof(int   )*cel->n_w); if(shape == NULL) error(0);
  rp    = (double *)malloc(sizeof(double)*cel->n_w); if(rp    == NULL) error(0);
  gxp   = (double *)malloc(sizeof(double)*cel->n_w); if(gxp   == NULL) error(0);
  gyp   = (double *)malloc(sizeof(double)*cel->n_w); if(gyp   == NULL) error(0);
  gzp   = (double *)malloc(sizeof(double)*cel->n_w); if(gzp   == NULL) error(0);
  alpha = (double *)malloc(sizeof(double)*cel->n_w); if(alpha == NULL) error(0);
  beta  = (double *)malloc(sizeof(double)*cel->n_w); if(beta  == NULL) error(0);
  gamma = (double *)malloc(sizeof(double)*cel->n_w); if(gamma == NULL) error(0);

  cel->rH_w   = (double *)malloc(sizeof(double)*cel->n_w                   ); if(cel->rH_w   == NULL) error(0);
  cel->xH_w   = (double *)malloc(sizeof(double)*cel->n_w*cel->vertex_w     ); if(cel->xH_w   == NULL) error(0);
  cel->yH_w   = (double *)malloc(sizeof(double)*cel->n_w*cel->vertex_w     ); if(cel->yH_w   == NULL) error(0);
  cel->zH_w   = (double *)malloc(sizeof(double)*cel->n_w*cel->vertex_w     ); if(cel->zH_w   == NULL) error(0);
  cel->rxH_w  = (double *)malloc(sizeof(double)*cel->n_w                   ); if(cel->rxH_w  == NULL) error(0);
  cel->ryH_w  = (double *)malloc(sizeof(double)*cel->n_w                   ); if(cel->ryH_w  == NULL) error(0);
  cel->rzH_w  = (double *)malloc(sizeof(double)*cel->n_w                   ); if(cel->rzH_w  == NULL) error(0);
  cel->uH_w   = (double *)malloc(sizeof(double)*cel->n_w*cel->vertex_w     ); if(cel->uH_w   == NULL) error(0);
  cel->vH_w   = (double *)malloc(sizeof(double)*cel->n_w*cel->vertex_w     ); if(cel->vH_w   == NULL) error(0);
  cel->wH_w   = (double *)malloc(sizeof(double)*cel->n_w*cel->vertex_w     ); if(cel->wH_w   == NULL) error(0);
  cel->fxH_w  = (double *)malloc(sizeof(double)*cel->n_w*cel->vertex_w     ); if(cel->fxH_w  == NULL) error(0);
  cel->fyH_w  = (double *)malloc(sizeof(double)*cel->n_w*cel->vertex_w     ); if(cel->fyH_w  == NULL) error(0);
  cel->fzH_w  = (double *)malloc(sizeof(double)*cel->n_w*cel->vertex_w     ); if(cel->fzH_w  == NULL) error(0);
  cel->c0H_w  = (double *)malloc(sizeof(double)*cel->n_w*cel->element_w*TRI); if(cel->c0H_w  == NULL) error(0);
  cel->s0H_w  = (double *)malloc(sizeof(double)*cel->n_w*cel->element_w*TRI); if(cel->s0H_w  == NULL) error(0);
  cel->eleH_w = (int    *)malloc(sizeof(int   )*cel->n_w*cel->element_w*TRI); if(cel->eleH_w == NULL) error(0);
  cel->jcH_w  = (int    *)malloc(sizeof(int   )*cel->n_w*cel->vertex_w*HEX ); if(cel->jcH_w  == NULL) error(0);
  cel->jcnH_w = (int    *)malloc(sizeof(int   )*cel->n_w*cel->vertex_w     ); if(cel->jcnH_w == NULL) error(0);
  cel->ecH_w  = (int    *)malloc(sizeof(int   )*cel->n_w*cel->element_w*TRI); if(cel->ecH_w  == NULL) error(0);

// Capsule properties ---
  shape[0] = SPHERE;  
//  if     (cdo->restart == 0) rp[0] = CW/RATIO; 
//  else if(cdo->restart == 1) rp[0] = CW; 
  rp[0]    = CW;
  gxp[0]   = CX;
  gyp[0]   = -0.5*LY + CW + cdo->dx;
  gzp[0]   = CZ;   
  alpha[0] = CA;      
  beta[0]  = CB;      
  gamma[0] = CG;       

// Radius, coordinate, velocity, stress, vertex number of triangular element ---
  for(ic = 0;ic < cel->n_w;ic++){
      cel->rH_w[ic] = rp[ic];

      for(j = 0;j < cel->vertex_w;j++){
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

          jid = ic*cel->vertex_w + j;

          cel->xH_w[jid] = tx;
          cel->yH_w[jid] = ty*cos(alpha[ic]) - tz*sin(alpha[ic]);
          cel->zH_w[jid] = ty*sin(alpha[ic]) + tz*cos(alpha[ic]);
          tz = cel->zH_w[jid];
          tx = cel->xH_w[jid];
          ty = cel->yH_w[jid];
          cel->zH_w[jid] = tz*cos(beta[ic]) - tx*sin(beta[ic]);
          cel->xH_w[jid] = tz*sin(beta[ic]) + tx*cos(beta[ic]);
          cel->yH_w[jid] = ty;
          tx = cel->xH_w[jid];
          ty = cel->yH_w[jid];
          tz = cel->zH_w[jid];
          cel->xH_w[jid] = tx*cos(gamma[ic]) - ty*sin(gamma[ic]);
          cel->yH_w[jid] = tx*sin(gamma[ic]) + ty*cos(gamma[ic]);
          cel->zH_w[jid] = tz;
          cel->xH_w[jid] = cel->xH_w[jid]*rp[ic] + gxp[ic];
          cel->yH_w[jid] = cel->yH_w[jid]*rp[ic] + gyp[ic];
          cel->zH_w[jid] = cel->zH_w[jid]*rp[ic] + gzp[ic];

          cel->uH_w[jid]  = 0.0;
          cel->vH_w[jid]  = 0.0;
          cel->wH_w[jid]  = 0.0;
          cel->fxH_w[jid] = 0.0;
          cel->fyH_w[jid] = 0.0;
          cel->fzH_w[jid] = 0.0;
      }

      for(e = 0;e < cel->element_w;e++){
          if(shape[ic] == SPHERE){
              j1 = sph_ele[e*TRI  ] + ic*cel->vertex_w;
              j2 = sph_ele[e*TRI+1] + ic*cel->vertex_w;
              j3 = sph_ele[e*TRI+2] + ic*cel->vertex_w;
          }
          else if(shape[ic] == BICONCAVE){
              j1 = bic_ele[e*TRI  ] + ic*cel->vertex_w;
              j2 = bic_ele[e*TRI+1] + ic*cel->vertex_w;
              j3 = bic_ele[e*TRI+2] + ic*cel->vertex_w;
          }
          else break;

          eid = ic*cel->element_w + e;

          cel->eleH_w[eid*TRI  ] = j1;
          cel->eleH_w[eid*TRI+1] = j2;
          cel->eleH_w[eid*TRI+2] = j3;
      }
  }

// Vertex connectivity ---
  for(ic = 0;ic < cel->n_w;ic++){
      for(j = 0;j < cel->vertex_w;j++){
          jid = ic*cel->vertex_w + j;
          cel->jcnH_w[jid]      = 0;
          cel->jcH_w[jid*HEX+0] = -1;
          cel->jcH_w[jid*HEX+1] = -1;
          cel->jcH_w[jid*HEX+2] = -1;
          cel->jcH_w[jid*HEX+3] = -1;
          cel->jcH_w[jid*HEX+4] = -1;
          cel->jcH_w[jid*HEX+5] = -1;

          for(e = 0;e < cel->element_w;e++){
              eid = ic*cel->element_w + e;
              j1 = cel->eleH_w[eid*TRI+0];
              j2 = cel->eleH_w[eid*TRI+1];
              j3 = cel->eleH_w[eid*TRI+2];

              if(jid == j1){
                  for(k = 0;k < HEX;k++){
                      if(j2 == cel->jcH_w[jid*HEX+k]) j2 = -1;
                      if(j3 == cel->jcH_w[jid*HEX+k]) j3 = -1;
                  }
                  if(j2 != -1){
                      cel->jcH_w[jid*HEX + cel->jcnH_w[jid]] = j2;
                      cel->jcnH_w[jid]++;
                  }
                  if(j3 != -1){
                      cel->jcH_w[jid*HEX + cel->jcnH_w[jid]] = j3;
                      cel->jcnH_w[jid]++;
                  }
              }
              else if(jid == j2){
                  for(k = 0;k < HEX;k++){
                      if(j3 == cel->jcH_w[jid*HEX+k]) j3 = -1;
                      if(j1 == cel->jcH_w[jid*HEX+k]) j1 = -1;
                  }
                  if(j3 != -1){
                      cel->jcH_w[jid*HEX + cel->jcnH_w[jid]] = j3;
                      cel->jcnH_w[jid]++;
                  }
                  if(j1 != -1){
                      cel->jcH_w[jid*HEX + cel->jcnH_w[jid]] = j1;
                      cel->jcnH_w[jid]++;
                  }
              }
              else if(jid == j3){
                  for(k = 0;k < HEX;k++){
                      if(j1 == cel->jcH_w[jid*HEX+k]) j1 = -1;
                      if(j2 == cel->jcH_w[jid*HEX+k]) j2 = -1;
                  }
                  if(j1 != -1){
                      cel->jcH_w[jid*HEX + cel->jcnH_w[jid]] = j1;
                      cel->jcnH_w[jid]++;
                  }
                  if(j2 != -1){
                      cel->jcH_w[jid*HEX + cel->jcnH_w[jid]] = j2;
                      cel->jcnH_w[jid]++;
                  }
              }
              else continue;
          }
      }
  }

// Element connectivity ---
  for(ic = 0;ic < cel->n_w;ic++){
      for(e = 0;e < cel->element_w;e++){
          ecn = 0;
          e1 = ic*cel->element_w + e;
          i1 = cel->eleH_w[e1*TRI+0];
          i2 = cel->eleH_w[e1*TRI+1];
          i3 = cel->eleH_w[e1*TRI+2];

          for(k = 0;k < cel->element_w;k++){
              if(e == k) continue;
              e2 = ic*cel->element_w + k;
              j1 = cel->eleH_w[e2*TRI  ];
              j2 = cel->eleH_w[e2*TRI+1];
              j3 = cel->eleH_w[e2*TRI+2];

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
                  cel->ecH_w[e1*TRI+ecn] = e2;
                  ecn++;
              }
              else continue;
          }
      }
  }

// Reference of bending ---
  for(ic = 0;ic < cel->n_w;ic++){
      for(e = 0;e < cel->element_w;e++){
          e1 = ic*cel->element_w + e;
          i1 = cel->eleH_w[e1*TRI+0];
          i2 = cel->eleH_w[e1*TRI+1];
          i3 = cel->eleH_w[e1*TRI+2];

          for(k = 0;k < TRI;k++){
              e2 = cel->ecH_w[e1*TRI+k];
              j1 = cel->eleH_w[e2*TRI+0];
              j2 = cel->eleH_w[e2*TRI+1];
              j3 = cel->eleH_w[e2*TRI+2];

              if     ((i1==j1 && i2==j2)||(i1==j2 && i2==j1)){ k1 = i3; k2 = i1; k3 = i2; k4 = j3; }
              else if((i1==j2 && i2==j3)||(i1==j3 && i2==j2)){ k1 = i3; k2 = i1; k3 = i2; k4 = j1; }
              else if((i1==j3 && i2==j1)||(i1==j1 && i2==j3)){ k1 = i3; k2 = i1; k3 = i2; k4 = j2; }
              else if((i2==j1 && i3==j2)||(i2==j2 && i3==j1)){ k1 = i1; k2 = i2; k3 = i3; k4 = j3; }
              else if((i2==j2 && i3==j3)||(i2==j3 && i3==j2)){ k1 = i1; k2 = i2; k3 = i3; k4 = j1; }
              else if((i2==j3 && i3==j1)||(i2==j1 && i3==j3)){ k1 = i1; k2 = i2; k3 = i3; k4 = j2; }
              else if((i3==j1 && i1==j2)||(i3==j2 && i1==j1)){ k1 = i2; k2 = i3; k3 = i1; k4 = j3; }
              else if((i3==j2 && i1==j3)||(i3==j3 && i1==j2)){ k1 = i2; k2 = i3; k3 = i1; k4 = j1; }
              else if((i3==j3 && i1==j1)||(i3==j1 && i1==j3)){ k1 = i2; k2 = i3; k3 = i1; k4 = j2; }

              a1x = cel->xH_w[k1]; a2x = cel->xH_w[k2]; a3x = cel->xH_w[k3]; a4x = cel->xH_w[k4];
              a1y = cel->yH_w[k1]; a2y = cel->yH_w[k2]; a3y = cel->yH_w[k3]; a4y = cel->yH_w[k4];
              a1z = cel->zH_w[k1]; a2z = cel->zH_w[k2]; a3z = cel->zH_w[k3]; a4z = cel->zH_w[k4];

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
              else                             sint = -sqrt(cost2);

              cel->c0H_w[e1*TRI + k] = cost;
              cel->s0H_w[e1*TRI + k] = sint;
              cel->c0H_w[e1*TRI + k] = 1.0;
              cel->s0H_w[e1*TRI + k] = 0.0;
          }
      }
  }

  free(sph_x); free(sph_y); free(sph_z); free(sph_ele);
  free(bic_x); free(bic_y); free(bic_z); free(bic_ele);
  free(shape); free(rp);    
  free(gxp);   free(gyp);   free(gzp);
  free(alpha); free(beta);  free(gamma);

// Numerical condition ---
  printf("====================================================================\n");
  printf("  Number of WBC             : %d\n",N_W);
  printf("  Radius of WBC             : %9.3e [um]\n",RAw*1.0e+06);
  printf("  Viscosity ratio           : %9.3e [-]\n",LAMBDA2);
  printf("  Surface shear elasticity  : %9.3e [N/m]\n",GSw);
  printf("  Average bending stiffness : %9.3e [J]\n",KCw);
  printf("====================================================================\n");

  return;
  #endif // N_W
#endif // CapsuleSHEARFLOW || CapsuleCHANNELFLOW
}
