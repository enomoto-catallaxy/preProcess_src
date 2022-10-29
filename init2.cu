#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "def.h"


void  add_nlink
//==========================================================
//
//  SET LINKING DATA
//
//
(
    int       na,            // first node
    int       nb,            // second node
    int       *nln,          // number of linking node
    int       *ln            // linking node id number
)
//----------------------------------------------------------
{
    int       flag = 0, i;


    for(i = 0; i < nln[na]; i++){
        if(nb == ln[7*na + i]){
            flag = 1;
            continue;
        }
    }

    if(flag == 0){
        ln[7*na + nln[na]] = nb;
        nln[na]++;
    }

    return;
}


void  add_elink
//==========================================================
//
//  SET LINKING DATA
//
//
(
    int       ni,            // id number of node
    int       ei,            // id number of element
    int       *nle,          // number of linking element
    int       *le            // linking element id number
)
//----------------------------------------------------------
{
    le[6*ni + nle[ni]] = ei;
    nle[ni]++;

    return;
}


int  checklink
//==========================================================
//
//  CHECK WHETHER LINKED OR NOT
//
//
(
    int       na,            // first node
    int       nb,            // second node
    int       *nln,          // number of linking node
    int       *ln            // linking node id number
)
//----------------------------------------------------------
{
    int       i;
    int       flag = 0;


    for(i = 0; i < nln[na]; i++){
        if(nb == ln[7*na + i]) flag = 1;
    }

    return flag;
}


void  sort_link_spring
//==========================================================
//
//  SORT LINKING DATA
//
//
(
    double    *x,            // position coordinate at n step
    int       *nln,          // number of linking node
    int       *ln,           // linking node id number
    int       vertex,
    int       ic
)
//----------------------------------------------------------
{
    int       j, k, l, n, na, nb, jid;
    int       lns[7];
    double    va[3], vb[3], c[3], inpro;
    double    gx = 0.0, gy = 0.0, gz = 0.0;


    for(j = 0;j < vertex;j++){
        jid = ic*vertex + j;
        gx += x[3*jid + 0];
        gy += x[3*jid + 1];
        gz += x[3*jid + 2];
    }
    gx /= vertex;
    gy /= vertex;
    gz /= vertex;

    for(j = 0;j < vertex;j++){
        jid = ic*vertex + j;
        n = 0;
        lns[0] = ln[7*jid + 0];

        for(k = 1;k < nln[jid];k++){
            na = lns[k - 1];

            va[0] = x[3*na + 0] - x[3*jid + 0];
            va[1] = x[3*na + 1] - x[3*jid + 1];
            va[2] = x[3*na + 2] - x[3*jid + 2];

            for(l = 0;l < nln[jid];l++){
                nb = ln[7*jid + l];
                if(na == nb) continue;

                if(checklink(na,nb,nln,ln) == 1){
                    vb[0] = x[3*nb + 0] - x[3*jid + 0];
                    vb[1] = x[3*nb + 1] - x[3*jid + 1];
                    vb[2] = x[3*nb + 2] - x[3*jid + 2];

                    c[0] = va[1]*vb[2] - va[2]*vb[1];
                    c[1] = va[2]*vb[0] - va[0]*vb[2];
                    c[2] = va[0]*vb[1] - va[1]*vb[0];

                    inpro = c[0]*(x[3*jid + 0] - gx)
                          + c[1]*(x[3*jid + 1] - gy)
                          + c[2]*(x[3*jid + 2] - gz);

                    if(inpro > 0){
                        lns[k] = nb;
                        n++;
                        continue;
                    }
                }
            }
        }

        if(n + 1 != nln[jid]) error(3);
        else                  for(k = 0;k < nln[jid];k++) ln[7*jid + k] = lns[k];
    }

    return;
}


void  sort_link_order
//==========================================================
//
//  SORT LINKING DATA
//
//
(
    int       *nln,          // number of linking node
    int       *ln,           // linking node id number
    int       vertex,
    int       ic
)
//----------------------------------------------------------
{
    int       j, k, l, jid;
    int       temp;


    for(j = 0; j < vertex; j++){
        jid = ic*vertex + j;
        ln[7*jid + nln[jid]] = jid;

        for(k = 0; k < nln[jid] + 1; k++){
            for(l = nln[jid]; l > k; l--){
                if(ln[7*jid + l - 1] > ln[7*jid + l]){
                    temp = ln[7*jid + l];
                    ln[7*jid + l] = ln[7*jid + l - 1];
                    ln[7*jid + l - 1] = temp;
                }
            }
        }
    }

    return;
}


void  set_contravariant
//==========================================================
//
//  SET REFERENCE CONTRAVARIANT METRIC TENSOR
//
//
(
    double    *xr,
    double    *n,
    int       *ele,
    int       cln,
    int       element
)
//----------------------------------------------------------
{
    int       ic, e, j, eid;
    int       ea, eb, ec;
    double    v1[3], v2[3], v[4];
    double    lv1, lv2, inv12, dv12;


    for(ic = 0;ic < cln;ic++){
        for(e = 0; e < element; e++){
            eid = ic*element + e;
            ea = ele[3*eid + 0];
            eb = ele[3*eid + 1];
            ec = ele[3*eid + 2];

            for(j = 0; j < 3; j++){
                v1[j] = xr[3*eb + j] - xr[3*ea + j];
                v2[j] = xr[3*ec + j] - xr[3*ea + j];
            }

            lv1 = v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2];
            lv2 = v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2];
            inv12 = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2];
            dv12 = lv1*lv2 - inv12*inv12;

            // covariant metric tensor
            v[0] = lv1;   v[1] = inv12;
            v[2] = inv12; v[3] = lv2;

            // contravariant metric tensor
            n[4*eid + 0] = v[3]/dv12;  n[4*eid + 1] = -v[1]/dv12;
            n[4*eid + 2] = -v[2]/dv12; n[4*eid + 3] = v[0]/dv12;
        }
    }

    return;
}


void  fem_init
//==========================================================
//
//  INITIAL SETTING OF FEM
//
//
(
    cell      *cel,
    fem       *fem
)
//----------------------------------------------------------
{
#if defined(KARMAN) || defined(SHEARFLOW) || defined(CHANNELFLOW)
  return;
#elif defined(CapsuleSHEARFLOW) || defined(CapsuleCHANNELFLOW)
  #if(N_R > 0)
  int       i, j, k, e, jid, eid, division,
              ea, eb, ec, checker;

// Initialize FEM ---
  printf("Initializing FEM ---------------------------------------------------\n");

// Parameters ---
  cel->n_edge = cel->vertex + cel->element - 2;
  cel->nnz    = (2*cel->n_edge + cel->vertex)*3;

  division  = (int)ceil((double)cel->vertex*(double)cel->n/(double)DIV);
  cel->n_th = division*DIV;
  cel->mn   = expand2pow2(3*cel->vertex);
  if(cel->mn < 256) error(3);

// Allocate variables ---
  fem->x   = (double *)malloc(sizeof(double)*cel->n_th*3          ); if(fem->x   == NULL) error(0);
  fem->xr  = (double *)malloc(sizeof(double)*cel->n_th*3          ); if(fem->xr  == NULL) error(0);
  fem->q   = (double *)malloc(sizeof(double)*cel->mn*cel->n       ); if(fem->q   == NULL) error(0);
  fem->qi  = (double *)malloc(sizeof(double)*cel->mn*cel->n       ); if(fem->qi  == NULL) error(0);
  fem->t   = (double *)malloc(sizeof(double)*cel->element*cel->n*4); if(fem->t   == NULL) error(0);
  fem->tp  = (double *)malloc(sizeof(double)*cel->element*cel->n*2); if(fem->tp  == NULL) error(0);
  fem->rg  = (double *)malloc(sizeof(double)*cel->element*cel->n  ); if(fem->rg  == NULL) error(0);
  fem->n   = (double *)malloc(sizeof(double)*cel->element*cel->n*4); if(fem->n   == NULL) error(0);
  fem->ele = (int    *)malloc(sizeof(int   )*cel->element*cel->n*6); if(fem->ele == NULL) error(0);
  fem->nln = (int    *)malloc(sizeof(int   )*cel->n_th            ); if(fem->nln == NULL) error(0);
  fem->ln  = (int    *)malloc(sizeof(int   )*cel->n_th*7          ); if(fem->ln  == NULL) error(0);
  fem->lo  = (int    *)malloc(sizeof(int   )*cel->n_th*7          ); if(fem->lo  == NULL) error(0);
  fem->nle = (int    *)malloc(sizeof(int   )*cel->n_th            ); if(fem->nle == NULL) error(0);
  fem->le  = (int    *)malloc(sizeof(int   )*cel->n_th*6          ); if(fem->le  == NULL) error(0);

  for(j = 0;j < cel->n_th*3;j++){
      fem->x[j]  = 0.0;
      fem->xr[j] = 0.0;
  }
  for(j = 0;j < cel->mn*cel->n;       j++) fem->q[j]   = 0.0;
  for(j = 0;j < cel->mn*cel->n;       j++) fem->qi[j]  = 0.0;
  for(j = 0;j < cel->element*cel->n*4;j++) fem->t[j]   = 0.0;
  for(j = 0;j < cel->element*cel->n*2;j++) fem->tp[j]  = 0.0;
  for(j = 0;j < cel->element*cel->n;  j++) fem->rg[j]  = 0.0;
  for(j = 0;j < cel->element*cel->n*4;j++) fem->n[j]   = 0.0;
  for(j = 0;j < cel->element*cel->n*6;j++) fem->ele[j] = 0;
  for(j = 0;j < cel->n_th;            j++) fem->nln[j] = 0;
  for(j = 0;j < cel->n_th*7;          j++) fem->ln[j]  = 0;
  for(j = 0;j < cel->n_th*7;          j++) fem->lo[j]  = 0;
  for(j = 0;j < cel->n_th;            j++) fem->nle[j] = 0;
  for(j = 0;j < cel->n_th*6;          j++) fem->le[j]  = 0;

  for(i = 0;i < cel->n;i++){
      for(j = 0;j < cel->vertex;j++){
          fem->xr[i*cel->vertex*3 + j*3 + 0]
          = fem->x[i*cel->vertex*3 + j*3 + 0]
          = cel->xH[i*cel->vertex + j];

          fem->xr[i*cel->vertex*3 + j*3 + 1] 
          = fem->x[i*cel->vertex*3 + j*3 + 1]
          = cel->yH[i*cel->vertex + j];

          fem->xr[i*cel->vertex*3 + j*3 + 2]
          = fem->x[i*cel->vertex*3 + j*3 + 2]
          = cel->zH[i*cel->vertex + j];
      }
  }

// Linking information ---
  for(i = 0;i < cel->n;i++){
      for(e = 0;e < cel->element;e++){
          eid = i*cel->element + e;
          fem->ele[6*eid  ] = cel->eleH[3*eid  ];
          fem->ele[6*eid+1] = cel->eleH[3*eid+1];
          fem->ele[6*eid+2] = cel->eleH[3*eid+2];
      }

      for(e = 0;e < cel->element;e++){
          eid = i*cel->element + e;
          ea = cel->eleH[3*eid + 0];
          eb = cel->eleH[3*eid + 1];
          ec = cel->eleH[3*eid + 2];

          add_nlink(ea,eb,fem->nln,fem->ln);
          add_nlink(ea,ec,fem->nln,fem->ln);
          add_nlink(eb,ec,fem->nln,fem->ln);
          add_nlink(eb,ea,fem->nln,fem->ln);
          add_nlink(ec,ea,fem->nln,fem->ln);
          add_nlink(ec,eb,fem->nln,fem->ln);

          add_elink(ea,eid,fem->nle,fem->le);
          add_elink(eb,eid,fem->nle,fem->le);
          add_elink(ec,eid,fem->nle,fem->le);
      }

      checker = 0;
      for(j = 0;j < cel->vertex;j++){
          jid = i*cel->vertex + j;
          checker += fem->nln[jid];
      }
      if(checker != 2*cel->n_edge) error(3);

      for(j = 0;j < cel->vertex;j++){
          jid = i*cel->vertex + j;
          for(k = 0;k < fem->nln[jid];k++){
              fem->lo[7*jid + k] = fem->ln[7*jid + k];
          }
      }

      sort_link_spring(fem->x,fem->nln,fem->ln,cel->vertex,i);
      sort_link_order(fem->nln,fem->lo,cel->vertex,i);
  }

// Set contravariant ---
  set_contravariant(fem->xr,fem->n,cel->eleH,cel->n,cel->element);

// Allocate matrix variables ---
  fem->ptr    = (int    *)malloc(sizeof(int   )*cel->mn  ); if(fem->ptr   == NULL) error(0);
  fem->index  = (int    *)malloc(sizeof(int   )*cel->mn*7); if(fem->index == NULL) error(0);
  fem->value  = (double *)malloc(sizeof(double)*cel->mn*7); if(fem->value == NULL) error(0);
  fem->b      = (double *)malloc(sizeof(double)*cel->mn  ); if(fem->b     == NULL) error(0);

  for(j = 0;j < cel->mn;  j++) fem->ptr[j]   = 0;
  for(j = 0;j < cel->mn*7;j++) fem->index[j] = 0;
  for(j = 0;j < cel->mn*7;j++) fem->value[j] = 0.0;
  for(j = 0;j < cel->mn;  j++) fem->b[j]     = 0.0;

  return;
  #endif // N_R
#endif // CapsuleSHEARFLOW || CapsuleCHANNELFLOW
}

void  fem_init_wbc
//==========================================================
//
//  INITIAL SETTING OF FEM FOR WBC
//
//
(
    cell      *cel,
    fem       *fem
)
//----------------------------------------------------------
{
#if defined(KARMAN) || defined(SHEARFLOW) || defined(CHANNELFLOW)
  return;
#elif defined(CapsuleSHEARFLOW) || defined(CapsuleCHANNELFLOW)
  #if(N_W > 0)
  int       i, j, k, e, jid, eid, division,
            ea, eb, ec, checker;

// Initialize FEM ---
  printf("Initializing FEM for WBC -------------------------------------------\n");

// Parameters ---
  cel->n_edge_w = cel->vertex_w + cel->element_w - 2;
  cel->nnz_w    = (2*cel->n_edge_w + cel->vertex_w)*3;

  division    = (int)ceil((double)cel->vertex_w*(double)cel->n_w/(double)DIV);
  cel->n_th_w = division*DIV;
  cel->mn_w   = expand2pow2(3*cel->vertex_w);
  if(cel->mn_w < 256) error(3);

// Allocate variables ---
  fem->x_w   = (double *)malloc(sizeof(double)*cel->n_th_w*3            ); if(fem->x_w   == NULL) error(0);
  fem->xr_w  = (double *)malloc(sizeof(double)*cel->n_th_w*3            ); if(fem->xr_w  == NULL) error(0);
  fem->q_w   = (double *)malloc(sizeof(double)*cel->mn_w*cel->n_w       ); if(fem->q_w   == NULL) error(0);
  fem->qi_w  = (double *)malloc(sizeof(double)*cel->mn_w*cel->n_w       ); if(fem->qi_w  == NULL) error(0);
  fem->t_w   = (double *)malloc(sizeof(double)*cel->element_w*cel->n_w*4); if(fem->t_w   == NULL) error(0);
  fem->tp_w  = (double *)malloc(sizeof(double)*cel->element_w*cel->n_w*2); if(fem->tp_w  == NULL) error(0);
  fem->rg_w  = (double *)malloc(sizeof(double)*cel->element_w*cel->n_w  ); if(fem->rg_w  == NULL) error(0);
  fem->n_w   = (double *)malloc(sizeof(double)*cel->element_w*cel->n_w*4); if(fem->n_w   == NULL) error(0);
  fem->ele_w = (int    *)malloc(sizeof(int   )*cel->element_w*cel->n_w*6); if(fem->ele_w == NULL) error(0);
  fem->nln_w = (int    *)malloc(sizeof(int   )*cel->n_th_w              ); if(fem->nln_w == NULL) error(0);
  fem->ln_w  = (int    *)malloc(sizeof(int   )*cel->n_th_w*7            ); if(fem->ln_w  == NULL) error(0);
  fem->lo_w  = (int    *)malloc(sizeof(int   )*cel->n_th_w*7            ); if(fem->lo_w  == NULL) error(0);
  fem->nle_w = (int    *)malloc(sizeof(int   )*cel->n_th_w              ); if(fem->nle_w == NULL) error(0);
  fem->le_w  = (int    *)malloc(sizeof(int   )*cel->n_th_w*6            ); if(fem->le_w  == NULL) error(0);

  for(j = 0;j < cel->n_th_w*3;j++){
      fem->x_w[j]  = 0.0;
      fem->xr_w[j] = 0.0;
  }
  for(j = 0;j < cel->mn_w*cel->n_w;       j++) fem->q_w[j]   = 0.0;
  for(j = 0;j < cel->mn_w*cel->n_w;       j++) fem->qi_w[j]  = 0.0;
  for(j = 0;j < cel->element_w*cel->n_w*4;j++) fem->t_w[j]   = 0.0;
  for(j = 0;j < cel->element_w*cel->n_w*2;j++) fem->tp_w[j]  = 0.0;
  for(j = 0;j < cel->element_w*cel->n_w;  j++) fem->rg_w[j]  = 0.0;
  for(j = 0;j < cel->element_w*cel->n_w*4;j++) fem->n_w[j]   = 0.0;
  for(j = 0;j < cel->element_w*cel->n_w*6;j++) fem->ele_w[j] = 0;
  for(j = 0;j < cel->n_th_w;              j++) fem->nln_w[j] = 0;
  for(j = 0;j < cel->n_th_w*7;            j++) fem->ln_w[j]  = 0;
  for(j = 0;j < cel->n_th_w*7;            j++) fem->lo_w[j]  = 0;
  for(j = 0;j < cel->n_th_w;              j++) fem->nle_w[j] = 0;
  for(j = 0;j < cel->n_th_w*6;            j++) fem->le_w[j]  = 0;

  for(i = 0;i < cel->n_w;i++){
      for(j = 0;j < cel->vertex_w;j++){
          fem->xr_w[i*cel->vertex_w*3 + j*3 + 0]
        = fem->x_w[ i*cel->vertex_w*3 + j*3 + 0]
        = cel->xH_w[i*cel->vertex_w   + j      ];

          fem->xr_w[i*cel->vertex_w*3 + j*3 + 1] 
        = fem->x_w[ i*cel->vertex_w*3 + j*3 + 1]
        = cel->yH_w[i*cel->vertex_w   + j      ];

          fem->xr_w[i*cel->vertex_w*3 + j*3 + 2]
        = fem->x_w[ i*cel->vertex_w*3 + j*3 + 2]
        = cel->zH_w[i*cel->vertex_w   + j      ];
      }
  }

// Linking information ---
  for(i = 0;i < cel->n_w;i++){
      for(e = 0;e < cel->element_w;e++){
          eid = i*cel->element_w + e;
          fem->ele_w[6*eid+0] = cel->eleH_w[3*eid+0];
          fem->ele_w[6*eid+1] = cel->eleH_w[3*eid+1];
          fem->ele_w[6*eid+2] = cel->eleH_w[3*eid+2];
      }

      for(e = 0;e < cel->element_w;e++){
          eid = i*cel->element_w + e;
          ea = cel->eleH_w[3*eid + 0];
          eb = cel->eleH_w[3*eid + 1];
          ec = cel->eleH_w[3*eid + 2];

          add_nlink(ea,eb,fem->nln_w,fem->ln_w);
          add_nlink(ea,ec,fem->nln_w,fem->ln_w);
          add_nlink(eb,ec,fem->nln_w,fem->ln_w);
          add_nlink(eb,ea,fem->nln_w,fem->ln_w);
          add_nlink(ec,ea,fem->nln_w,fem->ln_w);
          add_nlink(ec,eb,fem->nln_w,fem->ln_w);

          add_elink(ea,eid,fem->nle_w,fem->le_w);
          add_elink(eb,eid,fem->nle_w,fem->le_w);
          add_elink(ec,eid,fem->nle_w,fem->le_w);
      }

      checker = 0;
      for(j = 0;j < cel->vertex_w;j++){
          jid = i*cel->vertex_w + j;
          checker += fem->nln_w[jid];
      }
      if(checker != 2*cel->n_edge_w) error(3);

      for(j = 0;j < cel->vertex_w;j++){
          jid = i*cel->vertex_w + j;
          for(k = 0;k < fem->nln_w[jid];k++){
              fem->lo_w[7*jid + k] = fem->ln_w[7*jid + k];
          }
      }

      sort_link_spring(fem->x_w,fem->nln_w,fem->ln_w,cel->vertex_w,i);
      sort_link_order(fem->nln_w,fem->lo_w,cel->vertex_w,i);
  }

// Set contravariant ---
  set_contravariant(fem->xr_w,fem->n_w,cel->eleH_w,cel->n_w,cel->element_w);

// Allocate matrix variables ---
  fem->ptr_w    = (int    *)malloc(sizeof(int   )*cel->mn_w  ); if(fem->ptr_w   == NULL) error(0);
  fem->index_w  = (int    *)malloc(sizeof(int   )*cel->mn_w*7); if(fem->index_w == NULL) error(0);
  fem->value_w  = (double *)malloc(sizeof(double)*cel->mn_w*7); if(fem->value_w == NULL) error(0);
  fem->b_w      = (double *)malloc(sizeof(double)*cel->mn_w  ); if(fem->b_w     == NULL) error(0);

  for(j = 0;j < cel->mn_w;  j++) fem->ptr_w[j]   = 0;
  for(j = 0;j < cel->mn_w*7;j++) fem->index_w[j] = 0;
  for(j = 0;j < cel->mn_w*7;j++) fem->value_w[j] = 0.0;
  for(j = 0;j < cel->mn_w;  j++) fem->b_w[j]     = 0.0;

  return;
  #endif // N_W > 0
#endif  // CapsuleSHEARFLOW || CapsuleCHANNELFLOW
}
