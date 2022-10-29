#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "def.h"


__device__ double  atomicDoubleAdd2(double* address, double val)
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


__global__ void  calc_tension
//==========================================================
//
//  CALCULATE TENSION
//
//
(
    int       iter,
    double    *x,            // position coordinate at n step
    double    *t,            // tension of each element
    double    *tp,           // principal tension of each element
    double    *rg,           // root g alpha beta
    double    *n,            // reference contravariant metric tensor
    int       *ele,          // node component of element
    int       n_ele,
    int       n_cel,
    double    rgs
)
//----------------------------------------------------------
{
    int       j = blockDim.x*blockIdx.x + threadIdx.x;
    int       ea, eb, ec;
    int       i;
    double    v1d[3], v2d[3];
    double    vd[4], nd[4];
    double    lv1d, lv2d, inv12d, dv12d, dn12;
    double    i1, i2, js;
    double    judge, root;
    double    lam1_1, lam2_1, lam1_2, lam2_2;

    ea = ele[6*j + 0];
    eb = ele[6*j + 1];
    ec = ele[6*j + 2];

    for (i = 0; i < 3; i++) {
      v1d[i] = x[3*eb + i] - x[3*ea + i];
      v2d[i] = x[3*ec + i] - x[3*ea + i];
      if (i == 0) {
        if (v1d[i] > LX*0.5) v1d[i] -= LX; else if (v1d[i] < -LX*0.5) v1d[i] += LX;
        if (v2d[i] > LX*0.5) v2d[i] -= LX; else if (v2d[i] < -LX*0.5) v2d[i] += LX;
      } else if (i == 2) {
        if (v1d[i] > LZ*0.5) v1d[i] -= LZ; else if (v1d[i] < -LZ*0.5) v1d[i] += LZ;
        if (v2d[i] > LZ*0.5) v2d[i] -= LZ; else if (v2d[i] < -LZ*0.5) v2d[i] += LZ;
      }
    }

    lv1d   = sqrt(v1d[0]*v1d[0] + v1d[1]*v1d[1] + v1d[2]*v1d[2]);
    lv2d   = sqrt(v2d[0]*v2d[0] + v2d[1]*v2d[1] + v2d[2]*v2d[2]);
    inv12d = v1d[0]*v2d[0] + v1d[1]*v2d[1] + v1d[2]*v2d[2];
    dv12d  = (lv1d*lv1d)*(lv2d*lv2d) - inv12d*inv12d;
    dn12   = n[4*j + 0]*n[4*j + 3] - n[4*j + 1]*n[4*j + 2];
    rg[j]  = sqrt(dv12d);

    // covariant metric tensor: deformed state ---
    vd[0] = lv1d*lv1d; vd[1] = inv12d;
    vd[2] = inv12d;    vd[3] = lv2d*lv2d;

    // contravariant metric tensor: deformed state ---
    nd[0] = vd[3]/dv12d;  nd[1] = -vd[1]/dv12d;
    nd[2] = -vd[2]/dv12d; nd[3] = vd[0]/dv12d;

    i1 = n[4*j + 0]*vd[0] + n[4*j + 1]*vd[1]
       + n[4*j + 2]*vd[2] + n[4*j + 3]*vd[3] - 2.0;
    i2 = dv12d*dn12 - 1.0;
    js = sqrt(i2 + 1.0);

    #if defined (SKALAK)
    for (i = 0; i < 4; i++) t[4*j + i] = rgs*(i1 + 1.0)/js*n[4*j + i] + (C*i2 - 1.0)*js*nd[i]; // Normalized by Gs [N/m]
    #elif defined (NEOHOOKEAN)
    for (i = 0; i < 4; i++) t[4*j + i] = rgs*(n[4*j + i]/js - nd[i]*js/(i2 + 1.0)/(i2 + 1.0)); // Normalized by Gs [N/m]
    #endif

    // calculate principle stress
    if ((iter-1)%FILEOUTPUT == 0) {
      tp[j*2 + 0] = 0.0;
      tp[j*2 + 1] = 0.0;
    }

    judge = (i1 + 2.0)*(i1 + 2.0) - 4.0*(i2 + 1.0);
    judge = fabs(judge);

    root   = sqrt(judge);
    lam1_2 = 0.5*(i1 + 2.0 + root);
    lam2_2 = 0.5*(i1 + 2.0 - root);

    lam1_1 = sqrt(lam1_2);
    lam2_1 = sqrt(lam2_2);

    tp[j*2 + 0] += (double)M/(double)FILEOUTPUT*lam1_1/lam2_1*(lam1_2 - 1.0 + C*lam2_2*i2);
    tp[j*2 + 1] += (double)M/(double)FILEOUTPUT*lam2_1/lam1_1*(lam2_2 - 1.0 + C*lam1_2*i2);

    return;
}


__global__ void  initialize
//==========================================================
//
//  INITIALIZE VECTOR Q, B
//
//
(
    double    *qc,           // capsule load
    double    *bc            // capsule source term
)
//----------------------------------------------------------
{
    int       j = blockDim.x*blockIdx.x + threadIdx.x;


    qc[j] = 0.0;
    bc[j] = 0.0;

    return;
}


__global__ void  set_matrix_component
//==========================================================
//
//  SET INITIAL MATRIX
//
//
(
    int       *ptr,          // ptr of matrix A
    int       *index,        // index of matrix A
    int       *nln,          // number of linking node
    int       *lo,           // order of linking node
    int       n_node,        // number of nodes
    int       mn,            // number of total threads
    int       cid            // capsule id number
)
//----------------------------------------------------------
{
    int       j = blockDim.x*blockIdx.x + threadIdx.x;
    int       node = j/3, d = j%3, i;
    int       jj = node + cid*n_node;


    if(node < n_node){
        ptr[j] = nln[jj] + 1;
        for(i = 0;i < nln[jj] + 1;i++)
            index[i*mn + j] = 3*(lo[7*jj + i] - cid*n_node) + d;
    }
    else{
        ptr[j] = 1;
        index[0*mn + j] = j;
    }

    return;
}


__global__ void  set_matrix_coefficient
//==========================================================
//
//  SET FEM MATRIX
//
//
(
    double    *x,            // position coordinate at n step
    double    *t,            // tension of each element
    double    *rg,           // root g alpha beta
    double    *value,        // crs matrix value
    double    *bc,           // capsule source term
    int       *ele,          // node component of element
    int       *nln,          // number of linking node
    int       *lo,           // order of linking node
    int       *le,           // linking element id number
    int       n_node,        // number of nodes
    int       mn,            // number of total threads
    int       cid            // capsule id number
)
//----------------------------------------------------------
{
    int       j = blockDim.x*blockIdx.x + threadIdx.x;
    int       node = j/3, d = j%3;
    int       jj = node + cid*n_node;
    int       i, k, element, ea, eb, ec, en;
    int       oa, ob, oc, order;
    int       n = nln[jj];
    int       alpha, beta;
    double    gab, tb = 0.0, jacobi;
    double    vd[2], tvalue[7] = {0.0};
    double    f[3][2] = {{-1.0, -1.0}, {1.0, 0.0}, {0.0, 1.0}};


    if(node < n_node){
        for(i = 0; i < n; i++){
            element = le[6*jj + i];
            gab = rg[element];

//            ea = ele[3*element + 0];
//            eb = ele[3*element + 1];
//            ec = ele[3*element + 2];
            ea = ele[6*element + 0];
            eb = ele[6*element + 1];
            ec = ele[6*element + 2];

            // get order of ea, eb, ec
            for(k = 0; k < n + 1; k++){
                if(ea == lo[7*jj + k]) oa = k;
                if(eb == lo[7*jj + k]) ob = k;
                if(ec == lo[7*jj + k]) oc = k;
                if(jj == lo[7*jj + k]) order = k;
            }

            // set matrix A
            if(ea == jj){tvalue[oa] += gab/12.0; en = 0;}
            else         tvalue[oa] += gab/24.0;
            if(eb == jj){tvalue[ob] += gab/12.0; en = 1;}
            else         tvalue[ob] += gab/24.0;
            if(ec == jj){tvalue[oc] += gab/12.0; en = 2;}
            else         tvalue[oc] += gab/24.0;

            // set source term b
            vd[0] = x[3*eb + d] - x[3*ea + d];
            vd[1] = x[3*ec + d] - x[3*ea + d];
            if(d == 0){
                if(vd[0] > LX*0.5) vd[0] -= LX; else if(vd[0] < -LX*0.5) vd[0] += LX;
                if(vd[1] > LX*0.5) vd[1] -= LX; else if(vd[1] < -LX*0.5) vd[1] += LX;
            }
            else if(d == 2){
                if(vd[0] > LZ*0.5) vd[0] -= LZ; else if(vd[0] < -LZ*0.5) vd[0] += LZ;
                if(vd[1] > LZ*0.5) vd[1] -= LZ; else if(vd[1] < -LZ*0.5) vd[1] += LZ;
            }

            for(k = 0; k < 4; k++){
                alpha = k/2; beta = k%2;
                tb += gab*t[4*element + k]
                      *(vd[alpha]*f[en][beta] + vd[beta]*f[en][alpha])/4.0;
            }
        }

        jacobi = tvalue[order];

        for(k = 0; k < 7; k++)
            value[k*mn + j] = tvalue[k]/jacobi;

        bc[j] = tb/jacobi;
    }

    return;
}


__global__ void  substitute_load
//==========================================================
//
//  SUBSTITUTE LOAD (QX,QY,QZ) -> Q
//
//
(
    int       iter,
    double    *qd,           // applied load to each node
    double    *qid,          // applied load to each node
    double    *qc,           // capsule load
    int       n_node,        // number of nodes
    int       cid            // capsule id number
)
//----------------------------------------------------------
{
    int       j = blockDim.x*blockIdx.x + threadIdx.x;
    int       jj = j + 3*cid*n_node;


    // substitute load
    qd[jj] = qc[j];

    // calculate intergral load
    if((iter-1)%FILEOUTPUT == 0) qid[jj] = 0.0;
    qid[jj] += (double)M/(double)FILEOUTPUT*qc[j];

    return;
}


void  calc_load
//==========================================================
//
//  CALCULATE LOAD
//
//
(
    int       iter,
    double    *xd,           // position coordinate at n step
    double    *td,           // tension of each element
    double    *rgd,          // root g alpha beta
    double    *valued,       // crs matrix value
    double    *qd,           // applied load to each node
    double    *qid,          // applied load to each node
    int       *ptrd,         // ptr of matrix A
    int       *indexd,       // index of matrix A
    int       *eled,         // node component of element
    int       *nlnd,         // number of linking node
    int       *lod,          // order of linking node
    int       *led,          // linking element id number
    int       cln,
    int       vertex,
    int       mn,
    int       nnz
)
//----------------------------------------------------------
{
    static double  *qc, *bc;
    static double  *d_ra, *d_r, *d_p, *d_m, *d_n, *d_o;
    static int     n1st = 0;
    int            cid, flag;
    int            n_node = vertex;
    int            maxThreads = 256, threads, blocks;

    dim3 mgrid(mn/256,1,1), mblock(256,1,1);

    if(n1st == 0){
        getNumBlocksAndThreads
        (mn,maxThreads,blocks,threads);
        cudaMalloc((void**)&qc  ,sizeof(double)*mn);
        cudaMalloc((void**)&bc  ,sizeof(double)*mn);
        cudaMalloc((void**)&d_ra,sizeof(double)*mn);
        cudaMalloc((void**)&d_r ,sizeof(double)*mn);
        cudaMalloc((void**)&d_p ,sizeof(double)*mn);
        cudaMalloc((void**)&d_m ,sizeof(double)*mn);
        cudaMalloc((void**)&d_n ,sizeof(double)*mn);
        cudaMalloc((void**)&d_o ,sizeof(double)*blocks);
        n1st++;
    }

    for(cid = 0; cid < cln; cid++){
        initialize <<< mgrid,mblock >>> (qc,bc);

        set_matrix_component <<< mgrid,mblock >>>
        (ptrd,indexd,nlnd,lod,n_node,mn,cid);
        set_matrix_coefficient <<< mgrid,mblock >>>
        (xd,td,rgd,valued,bc,eled,nlnd,lod,led,n_node,mn,cid);

        flag = bicgstab
        (ptrd,indexd,valued,qc,bc,d_ra,d_r,d_p,d_m,d_n,d_o,mn,nnz);
        if(flag != 0) error(4);

        substitute_load <<< mgrid,mblock >>>
        (iter,qd,qid,qc,n_node,cid);
    }

    return;
}


__global__ void  q2f
//==========================================================
//
//  CONVERT Q TO F ON GPU
//
//
(
    double    *fx,
    double    *fy,
    double    *fz,
    int       n,
    int       vertex,	
    double    *x,
    double    *qd,
    int       *nln,
    int       *ln
)
//----------------------------------------------------------
{
    int       jid, k, j1, j2, j3;
    double    x1,  y1,  z1,
              x2,  y2,  z2,
              x3,  y3,  z3,
              xx1, yy1, zz1,
              xx2, yy2, zz2,
              nvx, nvy, nvz,
              nn,  ds = 0.0;


    jid = blockDim.x*blockIdx.x + threadIdx.x;
    if(jid >= n*vertex) return;

//  ic = jid/vertex;

    j1 = jid;
    x1 = x[j1*3 + 0];
    y1 = x[j1*3 + 1];
    z1 = x[j1*3 + 2];

    for(k = 0;k < nln[j1];k++){
        j2 = ln[j1*7 + k];
        j3 = ln[j1*7 + (k+1)%nln[j1]];

        x2 = x[j2*3 + 0]; x3 = x[j3*3 + 0];
        y2 = x[j2*3 + 1]; y3 = x[j3*3 + 1];
        z2 = x[j2*3 + 2]; z3 = x[j3*3 + 2];

        xx1 = x2 - x1;
        yy1 = y2 - y1;
        zz1 = z2 - z1;
        if(xx1 > LX*0.5) xx1 -= LX; else if(xx1 < -LX*0.5) xx1 += LX;
        if(zz1 > LZ*0.5) zz1 -= LZ; else if(zz1 < -LZ*0.5) zz1 += LZ;
        xx2 = x3 - x1;
        yy2 = y3 - y1;
        zz2 = z3 - z1;
        if(xx2 > LX*0.5) xx2 -= LX; else if(xx2 < -LX*0.5) xx2 += LX;
        if(zz2 > LZ*0.5) zz2 -= LZ; else if(zz2 < -LZ*0.5) zz2 += LZ;
        nvx = -(yy1*zz2 - zz1*yy2);
        nvy = -(zz1*xx2 - xx1*zz2);
        nvz = -(xx1*yy2 - yy1*xx2);
        nn  = sqrt(nvx*nvx + nvy*nvy + nvz*nvz);
        ds += nn/2.0/3.0;
    }

    fx[jid] = -1.0*qd[jid*3 + 0]*ds;
    fy[jid] = -1.0*qd[jid*3 + 1]*ds;
    fz[jid] = -1.0*qd[jid*3 + 2]*ds;

    return;
}


__global__ void  fourpoint_bend
//==========================================================
//
//  BENDING ENERGY BY FOUR-POINT INTERACTIONS ON GPU
//
//
(
    double    *x,
    double    *y,
    double    *z,
    double    *fx,
    double    *fy,
    double    *fz,
    double    *c0,
    double    *s0,
    int       n,
    int       vertex,
    int       element,
    int       *ele,
    int       *ec,
    double    br
)
//----------------------------------------------------------
{
    int       k, eid, e1, e2,
              i1, i2, i3, j1, j2, j3, k1, k2, k3, k4;
    double    cost0, sint0, cost, cost2, sint, b11, b12, b22, beta, eps = 1.0e-6,
              a1x, a2x, a3x, a4x, a21x, a31x, a34x, a24x, a23x, t1x, t2x, xix, zex, fxb,
              a1y, a2y, a3y, a4y, a21y, a31y, a34y, a24y, a23y, t1y, t2y, xiy, zey, fyb,
              a1z, a2z, a3z, a4z, a21z, a31z, a34z, a24z, a23z, t1z, t2z, xiz, zez, fzb;


    eid = blockDim.x*blockIdx.x + threadIdx.x;
    if(eid >= n*element) return;

//  ic = eid/element;
//  tkb = 2.0/sqrt(3.0)*kc[ic];

    e1 = eid;
    i1 = ele[e1*TRI  ];
    i2 = ele[e1*TRI+1];
    i3 = ele[e1*TRI+2];

    for(k = 0;k < TRI;k++){
        e2 = ec[e1*TRI+k];
        j1 = ele[e2*TRI  ];
        j2 = ele[e2*TRI+1];
        j3 = ele[e2*TRI+2];

        if     ((i1==j1 && i2==j2)||(i1==j2 && i2==j1)){ k1 = i3; k2 = i1; k3 = i2; k4 = j3; }
        else if((i1==j2 && i2==j3)||(i1==j3 && i2==j2)){ k1 = i3; k2 = i1; k3 = i2; k4 = j1; }
        else if((i1==j3 && i2==j1)||(i1==j1 && i2==j3)){ k1 = i3; k2 = i1; k3 = i2; k4 = j2; }
        else if((i2==j1 && i3==j2)||(i2==j2 && i3==j1)){ k1 = i1; k2 = i2; k3 = i3; k4 = j3; }
        else if((i2==j2 && i3==j3)||(i2==j3 && i3==j2)){ k1 = i1; k2 = i2; k3 = i3; k4 = j1; }
        else if((i2==j3 && i3==j1)||(i2==j1 && i3==j3)){ k1 = i1; k2 = i2; k3 = i3; k4 = j2; }
        else if((i3==j1 && i1==j2)||(i3==j2 && i1==j1)){ k1 = i2; k2 = i3; k3 = i1; k4 = j3; }
        else if((i3==j2 && i1==j3)||(i3==j3 && i1==j2)){ k1 = i2; k2 = i3; k3 = i1; k4 = j1; }
        else if((i3==j3 && i1==j1)||(i3==j1 && i1==j3)){ k1 = i2; k2 = i3; k3 = i1; k4 = j2; }

        a1x = x[k1]; a2x = x[k2]; a3x = x[k3]; a4x = x[k4];
        a1y = y[k1]; a2y = y[k2]; a3y = y[k3]; a4y = y[k4];
        a1z = z[k1]; a2z = z[k2]; a3z = z[k3]; a4z = z[k4];

        if(a2x - a1x > LX*0.5) a2x -= LX; else if(a2x - a1x < -LX*0.5) a2x += LX;
        if(a3x - a1x > LX*0.5) a3x -= LX; else if(a3x - a1x < -LX*0.5) a3x += LX;
        if(a4x - a1x > LX*0.5) a4x -= LX; else if(a4x - a1x < -LX*0.5) a4x += LX;
        if(a2z - a1z > LZ*0.5) a2z -= LZ; else if(a2z - a1z < -LZ*0.5) a2z += LZ;
        if(a3z - a1z > LZ*0.5) a3z -= LZ; else if(a3z - a1z < -LZ*0.5) a3z += LZ;
        if(a4z - a1z > LZ*0.5) a4z -= LZ; else if(a4z - a1z < -LZ*0.5) a4z += LZ;

        a21x = a2x - a1x; a31x = a3x - a1x; a34x = a3x - a4x; a24x = a2x - a4x; a23x = a2x - a3x;
        a21y = a2y - a1y; a31y = a3y - a1y; a34y = a3y - a4y; a24y = a2y - a4y; a23y = a2y - a3y;
        a21z = a2z - a1z; a31z = a3z - a1z; a34z = a3z - a4z; a24z = a2z - a4z; a23z = a2z - a3z;

        xix = a21y*a31z - a21z*a31y; zex = a34y*a24z - a34z*a24y;
        xiy = a21z*a31x - a21x*a31z; zey = a34z*a24x - a34x*a24z;
        xiz = a21x*a31y - a21y*a31x; zez = a34x*a24y - a34y*a24x;

        t1x = (a1x + a2x + a3x)/3.0; t2x = (a4x + a2x + a3x)/3.0;
        t1y = (a1y + a2y + a3y)/3.0; t2y = (a4y + a2y + a3y)/3.0;
        t1z = (a1z + a2z + a3z)/3.0; t2z = (a4z + a2z + a3z)/3.0;

        cost0 = c0[e1*TRI + k];
        sint0 = s0[e1*TRI + k];

        cost  = (xix*zex + xiy*zey + xiz*zez) 
               /sqrt(xix*xix + xiy*xiy + xiz*xiz)
               /sqrt(zex*zex + zey*zey + zez*zez);
        cost2 = 1.0 - cost*cost; if(cost2 < eps) continue;

        if(((xix-zex)*(t1x-t2x) + (xiy-zey)*(t1y-t2y) + (xiz-zez)*(t1z-t2z)) >= 0.0)
//      { sint =  sqrt(cost2); beta =  tkb*(sint*cost0 - cost*sint0)/sqrt(cost2); }
        { sint =  sqrt(cost2); beta =  2.0/sqrt(3.0)*br*(sint*cost0 - cost*sint0)/sqrt(cost2); }
        else
//      { sint = -sqrt(cost2); beta = -tkb*(sint*cost0 - cost*sint0)/sqrt(cost2); }
        { sint = -sqrt(cost2); beta = -2.0/sqrt(3.0)*br*(sint*cost0 - cost*sint0)/sqrt(cost2); }

        b11 = -beta*cost/(xix*xix + xiy*xiy + xiz*xiz);
        b12 =  beta/sqrt(xix*xix + xiy*xiy + xiz*xiz)/sqrt(zex*zex + zey*zey + zez*zez);
        b22 = -beta*cost/(zex*zex + zey*zey + zez*zez);

        fxb = b11*(xiy*-a23z - xiz*-a23y) + b12*(zey*-a23z - zez*-a23y);
        fyb = b11*(xiz*-a23x - xix*-a23z) + b12*(zez*-a23x - zex*-a23z);
        fzb = b11*(xix*-a23y - xiy*-a23x) + b12*(zex*-a23y - zey*-a23x);
        atomicDoubleAdd2(&fx[k1],fxb);
        atomicDoubleAdd2(&fy[k1],fyb);
        atomicDoubleAdd2(&fz[k1],fzb);

        fxb = b11*(xiy*-a31z - xiz*-a31y) + b22*(zey*a34z - zez*a34y)
            + b12*((xiy*a34z - xiz*a34y) + (zey*-a31z - zez*-a31y));
        fyb = b11*(xiz*-a31x - xix*-a31z) + b22*(zez*a34x - zex*a34z)
            + b12*((xiz*a34x - xix*a34z) + (zez*-a31x - zex*-a31z));
        fzb = b11*(xix*-a31y - xiy*-a31x) + b22*(zex*a34y - zey*a34x)
            + b12*((xix*a34y - xiy*a34x) + (zex*-a31y - zey*-a31x));
        atomicDoubleAdd2(&fx[k2],0.5*fxb);
        atomicDoubleAdd2(&fy[k2],0.5*fyb);
        atomicDoubleAdd2(&fz[k2],0.5*fzb);

        fxb = b11*(xiy*a21z - xiz*a21y) + b22*(zey*-a24z - zez*-a24y)
            + b12*((xiy*-a24z - xiz*-a24y) + (zey*a21z - zez*a21y));
        fyb = b11*(xiz*a21x - xix*a21z) + b22*(zez*-a24x - zex*-a24z)
            + b12*((xiz*-a24x - xix*-a24z) + (zez*a21x - zex*a21z));
        fzb = b11*(xix*a21y - xiy*a21x) + b22*(zex*-a24y - zey*-a24x)
            + b12*((xix*-a24y - xiy*-a24x) + (zex*a21y - zey*a21x));
        atomicDoubleAdd2(&fx[k3],0.5*fxb);
        atomicDoubleAdd2(&fy[k3],0.5*fyb);
        atomicDoubleAdd2(&fz[k3],0.5*fzb);
    }

    return;
}

void  stress
//==========================================================
//
//  CONSTITUTIVE LAW/SPRING MODEL
//
//
(
    process   *prc,
    domain    *cdo,
    cell      *cel,
    fem       *fem,
    lattice   *ltc
)
//----------------------------------------------------------
{
#if defined(KARMAN) || defined(SHEARFLOW) || defined(CHANNELFLOW)
  return;
#elif defined(CapsuleSHEARFLOW) || defined(CapsuleCHANNELFLOW)
  #if(N_R > 0)
  int       thread1 = 256, block1 = (cel->n*cel->vertex)/thread1+1,
            thread2 = 256, block2 = (cel->n*cel->element)/thread2+1;
  dim3      dimB1(thread1,1,1), dimG1(block1,1,1),
            dimB2(thread2,1,1), dimG2(block2,1,1);

  int       n_ele = cel->n*cel->element, ele_block = 256;
  dim3      g_ele(n_ele/ele_block,1), b_ele(ele_block,1);


  calc_tension <<< g_ele,b_ele >>>
  (prc->iter,fem->xd,fem->td,fem->tpd,fem->rgd,fem->nd,fem->eled,
   cel->element,cel->n,cel->rg);

  calc_load
  (prc->iter,fem->xd,fem->td,fem->rgd,fem->valued,fem->qd,fem->qid,
   fem->ptrd,fem->indexd,fem->eled,fem->nlnd,fem->lod,fem->led,
   cel->n,cel->vertex,cel->mn,cel->nnz);

  q2f <<< dimG1,dimB1 >>>
  (cel->fxD,cel->fyD,cel->fzD,cel->n,cel->vertex,
   fem->xd,fem->qd,fem->nlnd,fem->lnd);

  fourpoint_bend <<< dimG2,dimB2 >>>
  (cel->xD,cel->yD,cel->zD,cel->fxD,cel->fyD,cel->fzD,cel->c0D,cel->s0D,
   cel->n,cel->vertex,cel->element,cel->eleD,cel->ecD,cel->br);

  cudaThreadSynchronize();
  #endif // N_R

// Calculating load of WBC
  #if(N_W > 0)
  int       thread3 = 256, block3 = (cel->n_w*cel->vertex_w )/thread3+1,
            thread4 = 256, block4 = (cel->n_w*cel->element_w)/thread4+1;
  dim3      dimB3(thread3,1,1), dimG3(block3,1,1),
            dimB4(thread4,1,1), dimG4(block4,1,1);
  int       n_ele2 = cel->n_w*cel->element_w, ele_block2 = 256;
  dim3      g_ele2(n_ele2/ele_block2,1), b_ele2(ele_block2,1);

  calc_tension <<< g_ele2,b_ele2 >>>
  (prc->iter,fem->xd_w,fem->td_w,fem->tpd_w,fem->rgd_w,fem->nd_w,fem->eled_w,
   cel->element_w,cel->n_w,cel->rg_w);

  calc_load
  (prc->iter,fem->xd_w,fem->td_w,fem->rgd_w,fem->valued_w,fem->qd_w,fem->qid_w,
   fem->ptrd_w,fem->indexd_w,fem->eled_w,fem->nlnd_w,fem->lod_w,fem->led_w,
   cel->n_w,cel->vertex_w,cel->mn_w,cel->nnz_w);

  q2f <<< dimG3,dimB3 >>>
  (cel->fxD_w,cel->fyD_w,cel->fzD_w,cel->n_w,cel->vertex_w,
   fem->xd_w,fem->qd_w,fem->nlnd_w,fem->lnd_w);

  fourpoint_bend <<< dimG4,dimB4 >>>
  (cel->xD_w,cel->yD_w,cel->zD_w,cel->fxD_w,cel->fyD_w,cel->fzD_w,cel->c0D_w,cel->s0D_w,
   cel->n_w,cel->vertex_w,cel->element_w,cel->eleD_w,cel->ecD_w,cel->br_w);

  cudaThreadSynchronize();
  #endif // N_W > 0

  return;
#endif // CapsuleSHEARFLOW || CapsuleCHANNELFLOW
}

