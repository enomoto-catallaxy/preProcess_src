#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "def.h"

void commandline(int argc, char **argv, int *restart, int *device)
  //------------------------------------------------------------------------------------
{
  const char *argumentRestart = "-restart",
             *argumentDeviceNum = "-cuda_set_device";
  ///*
  while(argc > 0){
    if(strcmp(argv[argc-1], argumentRestart) == 0) {
      // restart file number
      *restart = 1;
    }
    if(strcmp(argv[argc-1], argumentDeviceNum) == 0) {
      // device number
      *device = atoi(&argv[argc][0]);
    }
    argc--;
  }
  //*/
  //  *restart=1;*device=0;
}

/*
void restart_flag
//==========================================================
//
//  RESTART FLAG
//
//
(
 int       restart,
 domain    *cdo
 )
//----------------------------------------------------------
{
  cdo->restart = restart;

  return;
}
*/
void  restart_flag
//==========================================================
//
//  RESTART FLAG
//
//
(
    int       argc,
    char      **argv,
    domain    *cdo
)
//----------------------------------------------------------
{

//    if (argc > 3) {  // rice
//      if (strncmp(argv[3],"-restart",8) == 0) cdo->restart = 1;
//      else                                   error(6);
//    }
    if (argc > 1) {  // Others
      if (strncmp(argv[1],"-restart",8) == 0) cdo->restart = 1;
      else                                    error(6);

    } else cdo->restart = 0;

}

/*
void OutputLatticeData(process *prc, domain *cdo, lattice *ltc)
//==========================================================
{
  FILE  *fp;
  char filename[256];  
  double err = 0.0, u;
  int    n = 0;

  sprintf(filename, "./result/lattice.dat");
  fp = fopen(filename, "w");
  if (fp == NULL) error(1);
  fprintf(fp,"Variables = \"X\",\"Y\",\"Z\", \"R\",\"U\",\"V\",\"W\" ,\"ERR\"\n");
  for (int ix = 0; ix < cdo->nx; ix++) {
    for (int iy = 0; iy < cdo->ny; iy++) {
      int iz = cdo->nz/2;
      int id = ix + iy*cdo->nx + iz*cdo->nx*cdo->ny;
      double x = -LX/2.0 - 1.5*cdo->dx + (double)ix*cdo->dx;
      double y = -LY/2.0 - 1.5*cdo->dx + (double)iy*cdo->dx;
      double z = -LZ/2.0 - 1.5*cdo->dx + (double)iz*cdo->dx;
      double r = x/fabs(x)*sqrt(x*x + y*y);
      u = 1.0*(1.0 - r*r/(R*R));
      if (ltc->bcH[id] == COMP) {
        err += (u - ltc->wmH[id])*(u - ltc->wmH[id]);
        n++;
      }
      fprintf(fp, "%e %e %e %e %e %e %e %e\n", x, y, z, r, ltc->umH[id], ltc->vmH[id], ltc->wmH[id], (ltc->bcH[id] == COMP ? fabs(u - ltc->wmH[id]) : 0.0)); 
    }
  }
  fclose(fp);
  printf("Resolution : %10.3e  n: %d  err : %e\n", KL, n, sqrt(err)/n);
}
//----------------------------------------------------------
*/

void OutputCapsuleCenter(process *prc, domain *cdo, cell *cel)
//==========================================================
{
  FILE  *fp;
  char filename[256];  
  sprintf(filename,"./result/center.dat");

  if (prc->fnum%50 == 0) sprintf(filename, "./result/center%04d.dat", prc->fnum);
  else                   sprintf(filename, "./result/center.dat");
  fp = fopen(filename, "w");
  if (fp == NULL) error(1);
  #if(N_R > 0)
  for (int idCaps = 0; idCaps < cel->n; idCaps++) {
    fprintf(fp, "%d %e %e %e \n", idCaps, cel->gx[3*idCaps + 0], cel->gx[3*idCaps + 1], cel->gx[3*idCaps + 2]); 
  }
  #endif
  #if(N_W > 0)
  for (int idCaps = 0; idCaps < cel->n_w; idCaps++) {
    fprintf(fp, "%d %e %e %e \n", idCaps, cel->gx_w[3*idCaps + 0], cel->gx_w[3*idCaps + 1], cel->gx_w[3*idCaps + 2]); 
  }
  #endif
  fclose(fp);
}
//----------------------------------------------------------

void InputCapsuleCenter(domain *cdo, cell *cel)
  //==========================================================
{
  FILE  *fp;
  char filename[256];  
  sprintf(filename,"./result/center.dat");
  fp = fopen(filename,"r");
  if(fp == NULL) error(1);
  #if(N_R > 0)
  for (int idCaps = 0; idCaps < cel->n; idCaps++) {
    fscanf(fp, "%*d %lf %lf %lf", 
        &cel->gx[3*idCaps + 0],
        &cel->gx[3*idCaps + 1],
        &cel->gx[3*idCaps + 2]);
  }
  #endif
  #if(N_W > 0)
  for (int idCaps = 0; idCaps < cel->n_w; idCaps++) {
    fscanf(fp, "%*d %lf %lf %lf", 
        &cel->gx_w[3*idCaps + 0],
        &cel->gx_w[3*idCaps + 1],
        &cel->gx_w[3*idCaps + 2]);
  }
  #endif
  fclose(fp);
}
//----------------------------------------------------------

void  restart_file
//==========================================================
//
//  RESTART FILE
//
//
(
 process   *prc,
 domain    *cdo,
 lattice   *ltc,
 cell      *cel,
 fem       *fem,
 Wall      *wall,
 Wall      *wall_host
 )
//----------------------------------------------------------
{
  FILE      *fp;
  char      filename[256];
  int       i, j;

  if(cdo->restart == 0) return;
  else printf("Initializing restart process ---------------------------------------\n");

  // Input restart file --- restart_adhe.dat
//  sprintf(filename,"./result/adhesion/restart_adhe.dat");
//  fp = fopen(filename,"r");
//  if(fp == NULL) error(1);

  // information of computation process
//  fscanf(fp,"%d\n",&prc->fnum_ad);
//  fclose(fp);

  // Input restart file ---
  sprintf(filename,"./result/restart.bin");
  fp = fopen(filename,"rb");
  if(fp == NULL) error(1);

  // information of computation process
  fread(&prc->iter, sizeof(int), 1, fp);
  fread(&prc->fnum, sizeof(int), 1, fp);
  if (prc->fnum > 100) { prc->iter = 0; prc->fnum = 0; }

  // information of lattice Boltzmann nodes
  fread(ltc->fnH,  sizeof(double), cdo->n*Q, fp);
  fread(ltc->dmH,  sizeof(double), cdo->n,   fp);
  fread(ltc->umH,  sizeof(double), cdo->n,   fp);
  fread(ltc->vmH,  sizeof(double), cdo->n,   fp);
  fread(ltc->wmH,  sizeof(double), cdo->n,   fp);
  fread(ltc->vfH,  sizeof(double), cdo->n,   fp);
  fread(ltc->vfH2, sizeof(double), cdo->n,   fp);

  // information of cell
  #if(N_R > 0)
  fread(cel->xH,  sizeof(double), cel->n*cel->vertex,    fp);
  fread(cel->yH,  sizeof(double), cel->n*cel->vertex,    fp);
  fread(cel->zH,  sizeof(double), cel->n*cel->vertex,    fp);
  fread(cel->rxH, sizeof(double), cel->n,                fp);
  fread(cel->ryH, sizeof(double), cel->n,                fp);
  fread(cel->rzH, sizeof(double), cel->n,                fp);
  fread(cel->uH,  sizeof(double), cel->n*cel->vertex,    fp);
  fread(cel->vH,  sizeof(double), cel->n*cel->vertex,    fp);
  fread(cel->wH,  sizeof(double), cel->n*cel->vertex,    fp);
  fread(fem->qi,  sizeof(double), cel->n*cel->mn,        fp);
  fread(fem->tp,  sizeof(double), cel->n*cel->element*2, fp);
  #endif

  // information of WBC
  #if(N_W > 0)
  fread(cel->xH_w,  sizeof(double),cel->n_w*cel->vertex_w,    fp);
  fread(cel->yH_w,  sizeof(double),cel->n_w*cel->vertex_w,    fp);
  fread(cel->zH_w,  sizeof(double),cel->n_w*cel->vertex_w,    fp);
  fread(cel->rxH_w, sizeof(double),cel->n_w,                  fp);
  fread(cel->ryH_w, sizeof(double),cel->n_w,                  fp);
  fread(cel->rzH_w, sizeof(double),cel->n_w,                  fp);
  fread(cel->uH_w,  sizeof(double),cel->n_w*cel->vertex_w,    fp);
  fread(cel->vH_w,  sizeof(double),cel->n_w*cel->vertex_w,    fp);
  fread(cel->wH_w,  sizeof(double),cel->n_w*cel->vertex_w,    fp);
  fread(fem->qi_w,  sizeof(double),cel->n_w*cel->mn_w,        fp);
  fread(fem->tp_w,  sizeof(double),cel->n_w*cel->element_w*2, fp);
  #endif
  #if(CYTOADHESION == true)
  fread(cel->f_adhH,            sizeof(double), cel->adnum*cel->advertex*3,              fp);
  fread(cel->is_knobH,          sizeof(bool  ), cel->adnum*cel->advertex,                fp);
  fread(cel->attachment_pointH, sizeof(int   ), cel->adnum*cel->advertex*NUMSPRING_KNOB, fp);
  fread(wall_host->is_attached, sizeof(int   ), wall->num_wall,                          fp);
  #endif
  fclose(fp);

  // Storage FEM data about RBC
  #if(N_R > 0)
  for (i = 0;i < cel->n;i++) {
    for (j = 0;j < cel->vertex;j++) {
      fem->x[i*cel->vertex*3 + j*3 + 0] = cel->xH[i*cel->vertex + j];
      fem->x[i*cel->vertex*3 + j*3 + 1] = cel->yH[i*cel->vertex + j];
      fem->x[i*cel->vertex*3 + j*3 + 2] = cel->zH[i*cel->vertex + j];
    }
  }
  #endif

  // Storage FEM data about WBC
  #if(N_W > 0)
  for (i = 0;i < cel->n_w;i++) {
    for (j = 0;j < cel->vertex_w;j++) {
      fem->x_w[i*cel->vertex_w*3 + j*3 + 0] = cel->xH_w[i*cel->vertex_w + j];
      fem->x_w[i*cel->vertex_w*3 + j*3 + 1] = cel->yH_w[i*cel->vertex_w + j];
      fem->x_w[i*cel->vertex_w*3 + j*3 + 2] = cel->zH_w[i*cel->vertex_w + j];
    }
  }
  #endif

  // Copy for GPU ---
  checkCudaErrors(cudaMemcpy(ltc->fnD  ,ltc->fnH,  sizeof(double)*cdo->n*Q, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->dmD  ,ltc->dmH,  sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->umD  ,ltc->umH,  sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->vmD  ,ltc->vmH,  sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->wmD  ,ltc->wmH,  sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->vfD  ,ltc->vfH,  sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(ltc->vfD2 ,ltc->vfH2, sizeof(double)*cdo->n,   cudaMemcpyHostToDevice));

  #if(N_R > 0)
  checkCudaErrors(cudaMemcpy(cel->xD,  cel->xH,  sizeof(double)*cel->n*cel->vertex,    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->yD,  cel->yH,  sizeof(double)*cel->n*cel->vertex,    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->zD,  cel->zH,  sizeof(double)*cel->n*cel->vertex,    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->rxD, cel->rxH, sizeof(double)*cel->n,                cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->ryD, cel->ryH, sizeof(double)*cel->n,                cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->rzD, cel->rzH, sizeof(double)*cel->n,                cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->uD,  cel->uH,  sizeof(double)*cel->n*cel->vertex,    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->vD,  cel->vH,  sizeof(double)*cel->n*cel->vertex,    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->wD,  cel->wH,  sizeof(double)*cel->n*cel->vertex,    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->qid, fem->qi,  sizeof(double)*cel->n*cel->mn,        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->tpd, fem->tp,  sizeof(double)*cel->n*cel->element*2, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->xd,  fem->x,   sizeof(double)*cel->n_th*3,           cudaMemcpyHostToDevice));
  #endif

  #if(N_W > 0)
  checkCudaErrors(cudaMemcpy(cel->xD_w,  cel->xH_w,  sizeof(double)*cel->n_w*cel->vertex_w,    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->yD_w,  cel->yH_w,  sizeof(double)*cel->n_w*cel->vertex_w,    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->zD_w,  cel->zH_w,  sizeof(double)*cel->n_w*cel->vertex_w,    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->rxD_w, cel->rxH_w, sizeof(double)*cel->n_w,                  cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->ryD_w, cel->ryH_w, sizeof(double)*cel->n_w,                  cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->rzD_w, cel->rzH_w, sizeof(double)*cel->n_w,                  cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->uD_w,  cel->uH_w,  sizeof(double)*cel->n_w*cel->vertex_w,    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->vD_w,  cel->vH_w,  sizeof(double)*cel->n_w*cel->vertex_w,    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->wD_w,  cel->wH_w,  sizeof(double)*cel->n_w*cel->vertex_w,    cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->qid_w, fem->qi_w,  sizeof(double)*cel->n_w*cel->mn_w,        cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->tpd_w, fem->tp_w,  sizeof(double)*cel->n_w*cel->element_w*2, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(fem->xd_w,  fem->x_w,   sizeof(double)*cel->n_th_w*3,             cudaMemcpyHostToDevice));
  #endif

  #if (CYTOADHESION == true)
  checkCudaErrors(cudaMemcpy(cel->f_adhD,            cel->f_adhH,            sizeof(double)*cel->adnum*cel->advertex*3,              cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->is_knobD,          cel->is_knobH,          sizeof(bool  )*cel->adnum*cel->advertex,                cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cel->attachment_pointD, cel->attachment_pointH, sizeof(int   )*cel->adnum*cel->advertex*NUMSPRING_KNOB, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(wall->is_attached,      wall_host->is_attached, sizeof(int   )*wall->num_wall,                          cudaMemcpyHostToDevice));
  #endif

  InputCapsuleCenter(cdo, cel);
  return;
}

void  error
//==========================================================
//
//  PRINT ERROR MESSAGE AND EXIT THE PROGRAM
//
//
(
 int       error_type
 )
//----------------------------------------------------------
{
  char      message[256];


  if(error_type == 0)
    sprintf(message,"MALLOC ERROR\n");

  if(error_type == 1)
    sprintf(message,"FILE OPEN ERROR\n");

  if(error_type == 2)
    sprintf(message,"MAPPING ERROR\n");

  if(error_type == 3)
    sprintf(message,"FEM PART ERROR\n");

  if(error_type == 4)
    sprintf(message,"MATRIX ERROR IN FEM\n");

  if(error_type == 5)
    sprintf(message,"MATRIX ERROR IN FRONT-TRACKING METHOD\n");

  if(error_type == 6)
    sprintf(message,"UNDEFINED OPTION\n");

  if(error_type == 7)
    sprintf(message,"NUMERICAL CONDITION ERROR IN DIFFUSION NUMBER\n");

  if(error_type == 8)
    sprintf(message,"CELL INITIAL POSITION ERROR\n");

  if(error_type == 9)
    sprintf(message,"VOLUME OVER\n");

  printf("%s\n",message);
  exit(1);

  return;
}


void  gpu_init
//==========================================================
//
//  INITIALIZE GPU
//
//
(
 int       argc,
 char      **argv
 )
//----------------------------------------------------------
{
  cudaDeviceProp prop;

  int       dev, num_dev;
  cudaSetDevice(DEV);

//  int       inum, dev, num_dev;
//  inum = atoi(argv[2]);
//  cudaSetDevice(inum);

  cudaGetDevice(&dev);
  cudaGetDeviceCount(&num_dev);
  cudaGetDeviceProperties(&prop,dev);

  printf("%s: device %d of %d\n",prop.name,dev,num_dev);

  return;
}


void  preprocess
//==========================================================
//
//  PREPROCESS OF COMPUTATION
//
//
(
 process   *prc
 )
//----------------------------------------------------------
{
  prc->fnum = 0;                     // file number
  prc->fnum_ad = 0;                  // file number for adhesion parameters
  prc->iter = prc->fnum*FILEOUTPUT;  // iteration
  return;
}


int  expand2pow2
//==========================================================
//
//  EXPAND VALUE TO POWER OF 2
//
//
(
 int       n
 )
//----------------------------------------------------------
{
  int       cnt = 0;

  while(n > 1){
    if(n%2 == 1) n++;
    n = n/2;
    cnt++;
  }

  n = (int)pow(2.0,cnt);
  return n;
}


void  output_data
//==========================================================
//
//  OUTPUT DATA
//
//
(
 process   *prc,
 domain    *cdo,
 lattice   *ltc,
 cell      *cel,
 fem       *fem,
 Wall      *wall,
 Wall      *wall_host
 )
//----------------------------------------------------------
{
  printf("Result file %04d writing...\n",prc->fnum);

  // Variables of fluid
  cudaMemcpy(ltc->fnH,  ltc->fnD,  sizeof(double)*cdo->n*Q, cudaMemcpyDeviceToHost);
  cudaMemcpy(ltc->dmH,  ltc->dmD,  sizeof(double)*cdo->n,   cudaMemcpyDeviceToHost);
  cudaMemcpy(ltc->umH,  ltc->umD,  sizeof(double)*cdo->n,   cudaMemcpyDeviceToHost);
  cudaMemcpy(ltc->vmH,  ltc->vmD,  sizeof(double)*cdo->n,   cudaMemcpyDeviceToHost);
  cudaMemcpy(ltc->wmH,  ltc->wmD,  sizeof(double)*cdo->n,   cudaMemcpyDeviceToHost);
  cudaMemcpy(ltc->vfH,  ltc->vfD,  sizeof(double)*cdo->n,   cudaMemcpyDeviceToHost);
  cudaMemcpy(ltc->vfH2, ltc->vfD2, sizeof(double)*cdo->n,   cudaMemcpyDeviceToHost);

  // Variables of RBC
  #if(N_R > 0)
  cudaMemcpy(cel->xH,  cel->xD,  sizeof(double)*cel->n*cel->vertex,    cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->yH,  cel->yD,  sizeof(double)*cel->n*cel->vertex,    cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->zH,  cel->zD,  sizeof(double)*cel->n*cel->vertex,    cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->rxH, cel->rxD, sizeof(double)*cel->n,                cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->ryH, cel->ryD, sizeof(double)*cel->n,                cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->rzH, cel->rzD, sizeof(double)*cel->n,                cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->uH,  cel->uD,  sizeof(double)*cel->n*cel->vertex,    cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->vH,  cel->vD,  sizeof(double)*cel->n*cel->vertex,    cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->wH,  cel->wD,  sizeof(double)*cel->n*cel->vertex,    cudaMemcpyDeviceToHost);
  cudaMemcpy(fem->qi,  fem->qid, sizeof(double)*cel->n*cel->mn,        cudaMemcpyDeviceToHost);
  cudaMemcpy(fem->tp,  fem->tpd, sizeof(double)*cel->n*cel->element*2, cudaMemcpyDeviceToHost);
  #endif

  // Variables of WBC
  #if (N_W > 0)
  cudaMemcpy(cel->xH_w,  cel->xD_w,  sizeof(double)*cel->n_w*cel->vertex_w,    cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->yH_w,  cel->yD_w,  sizeof(double)*cel->n_w*cel->vertex_w,    cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->zH_w,  cel->zD_w,  sizeof(double)*cel->n_w*cel->vertex_w,    cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->rxH_w, cel->rxD_w, sizeof(double)*cel->n_w,                  cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->ryH_w, cel->ryD_w, sizeof(double)*cel->n_w,                  cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->rzH_w, cel->rzD_w, sizeof(double)*cel->n_w,                  cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->uH_w,  cel->uD_w,  sizeof(double)*cel->n_w*cel->vertex_w,    cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->vH_w,  cel->vD_w,  sizeof(double)*cel->n_w*cel->vertex_w,    cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->wH_w,  cel->wD_w,  sizeof(double)*cel->n_w*cel->vertex_w,    cudaMemcpyDeviceToHost);
  cudaMemcpy(fem->qi_w,  fem->qid_w, sizeof(double)*cel->n_w*cel->mn_w,        cudaMemcpyDeviceToHost);
  cudaMemcpy(fem->tp_w,  fem->tpd_w, sizeof(double)*cel->n_w*cel->element_w*2, cudaMemcpyDeviceToHost);
  #endif

  // Variables of adhesion
  #if (CYTOADHESION == true)
  cudaMemcpy(cel->f_adhH,            cel->f_adhD,            sizeof(double)*cel->adnum*cel->advertex*3,              cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->attachment_pointH, cel->attachment_pointD, sizeof(int   )*cel->adnum*cel->advertex*NUMSPRING_KNOB, cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->is_knobH,          cel->is_knobD,          sizeof(bool  )*cel->adnum*cel->advertex,                cudaMemcpyDeviceToHost);
  cudaMemcpy(wall_host->is_attached, wall->is_attached,      sizeof(int   )*wall_host->num_wall,                     cudaMemcpyDeviceToHost);
//  cudaMemcpy(cel->konH,              cel->konD,              sizeof(double)*cel->adnum*cel->advertex*NUMSPRING_KNOB, cudaMemcpyDeviceToHost);
//  cudaMemcpy(cel->koffH,             cel->koffD,             sizeof(double)*cel->adnum*cel->advertex*NUMSPRING_KNOB, cudaMemcpyDeviceToHost);
  cudaMemcpy(cel->r_ijH,             cel->r_ijD,             sizeof(double)*cel->adnum*cel->advertex*NUMSPRING_KNOB, cudaMemcpyDeviceToHost);
  #endif

  // Output numerical condition file ---
  ///*
  FILE        *fp;
  char        filename[256];
  static bool st1 = false;

  if (st1 == false) {
    sprintf(filename,"./result/config.dat");
    fp = fopen(filename,"w"); if(fp == NULL) error(1);

    fprintf(fp,"%10.3e %10.3e %10.3e %10.3e %10.3e       \n",Ca,Re,KL,Df,Cr);
    fprintf(fp,"%10.3e %10.3e %10.3e                     \n",Dc,Uc,Lc);
    fprintf(fp,"%10.3e %10.3e %10.3e                     \n",LX,LY,LZ);
    fprintf(fp," %d     %d     %d     %d     %d          \n",N_R,N_I,N_W,N_C,N_P);
    fprintf(fp,"%10.3e %10.3e %10.3e %10.3e %10.3e       \n",CKL,CR,CW,CC,CP);
    fprintf(fp,"%10.3e %10.3e %10.3e %10.3e              \n",GS,GSw,GSc,GSp);
    fprintf(fp,"%10.3e %10.3e %10.3e                     \n",cdo->dt,cdo->dx,cdo->tau);
    fprintf(fp," %d     %d     %d     %d     %d          \n",cdo->nx,cdo->ny,cdo->nz,cdo->n,wall_host->num_wall);
    fprintf(fp," %d     %d     %d     %d     %d          \n",ITRLIMIT,FILEOUTPUT,STANDARDOUTPUT,FLOWSTART,NUM_FLAG);
    fprintf(fp,"%10.3e  %10.3e                           \n",LAMBDA1,LAMBDA2);
    fprintf(fp,"%10.3e  %d                               \n",FREQ,1); // (cos -> 0, sin -> 1)
    fclose(fp);

    st1 = true;
  }
  //*/

// Output fluid file data (.bin) ---
///*
    if (prc->fnum%50 == 0) {
      sprintf(filename,"./result/prof%04d.bin",prc->fnum);
      fp = fopen(filename,"wb"); if (fp == NULL) error(1);

      fwrite(&prc->iter, sizeof(int   ), 1,      fp);
      fwrite(ltc->dmH,   sizeof(double), cdo->n, fp);
      fwrite(ltc->umH,   sizeof(double), cdo->n, fp);
      fwrite(ltc->vmH,   sizeof(double), cdo->n, fp);
      fwrite(ltc->wmH,   sizeof(double), cdo->n, fp);
      fwrite(ltc->vfH,   sizeof(double), cdo->n, fp);
      fwrite(ltc->vfH2,  sizeof(double), cdo->n, fp);
      fwrite(ltc->bcH,   sizeof(int   ), cdo->n, fp);
      fclose(fp);
    }
//*/
#if(defined(CapsuleCHANNELFLOW) || defined(CapsuleSHEARFLOW))
// Output cell file date (.bin) ---

    sprintf(filename,"./result/cell/cell%04d.bin",prc->fnum);
    fp = fopen(filename,"wb");
    if (fp == NULL) error(1);

    // About RBC
    #if (N_R > 0)
    //fwrite(&cel->inflation_flag, sizeof(int   ),1,                       fp);
    fwrite(&cel->n,              sizeof(int   ),1,                       fp);
    fwrite(&cel->vertex,         sizeof(int   ),1,                       fp);
    fwrite(&cel->element,        sizeof(int   ),1,                       fp);
    fwrite(cel->xH,              sizeof(double),cel->n*cel->vertex,      fp);
    fwrite(cel->yH,              sizeof(double),cel->n*cel->vertex,      fp);
    fwrite(cel->zH,              sizeof(double),cel->n*cel->vertex,      fp);
    fwrite(cel->uH,              sizeof(double),cel->n*cel->vertex,      fp);
    fwrite(cel->vH,              sizeof(double),cel->n*cel->vertex,      fp);
    fwrite(cel->wH,              sizeof(double),cel->n*cel->vertex,      fp);
    fwrite(fem->qi,              sizeof(double),cel->n*cel->vertex*3,    fp);
    fwrite(fem->tp,              sizeof(double),cel->n*cel->element*2,   fp);
    fwrite(cel->eleH,            sizeof(int   ),cel->n*cel->element*TRI, fp);
    #endif

    // About WBC
    #if (N_W > 0)
    fwrite(&cel->n_w,       sizeof(int   ), 1,                           fp);
    fwrite(&cel->vertex_w,  sizeof(int   ), 1,                           fp);
    fwrite(&cel->element_w, sizeof(int   ), 1,                           fp);
    fwrite(cel->xH_w,       sizeof(double), cel->n_w*cel->vertex_w,      fp);
    fwrite(cel->yH_w,       sizeof(double), cel->n_w*cel->vertex_w,      fp);
    fwrite(cel->zH_w,       sizeof(double), cel->n_w*cel->vertex_w,      fp);
    fwrite(cel->uH_w,       sizeof(double), cel->n_w*cel->vertex_w,      fp);
    fwrite(cel->vH_w,       sizeof(double), cel->n_w*cel->vertex_w,      fp);
    fwrite(cel->wH_w,       sizeof(double), cel->n_w*cel->vertex_w,      fp);
    fwrite(fem->qi_w,       sizeof(double), cel->n_w*cel->vertex_w*3,    fp);
    fwrite(fem->tp_w,       sizeof(double), cel->n_w*cel->element_w*2,   fp);
    fwrite(cel->eleH_w,     sizeof(int   ), cel->n_w*cel->element_w*TRI, fp);
    #endif

    #if (CYTOADHESION == true)
    fwrite(cel->f_adhH,            sizeof(double), cel->adnum*cel->advertex*3,              fp);
    fwrite(cel->attachment_pointH, sizeof(int   ), cel->adnum*cel->advertex*NUMSPRING_KNOB, fp);
    fwrite(cel->is_knobH,          sizeof(bool  ), cel->adnum*cel->advertex,                fp);
    fwrite(wall_host->is_attached, sizeof(int   ), wall_host->num_wall,                     fp);
    //fwrite(cel->konH,              sizeof(double), cel->adnum*cel->advertex*NUMSPRING_KNOB, fp);
    //fwrite(cel->koffH,             sizeof(double), cel->adnum*cel->advertex*NUMSPRING_KNOB, fp);
    fwrite(cel->r_ijH,             sizeof(double), cel->adnum*cel->advertex*NUMSPRING_KNOB, fp);
    #endif
    fclose(fp);

  // Output cell file date (.dat : for all condition)---
/*
  int       ic, k, e, j1, j2, flag;
  double    xx, zz;

  sprintf(filename,"./result/cell/cell%04d.dat",prc->fnum);
  fp = fopen(filename,"w"); if(fp == NULL) error(1);

  fprintf(fp,"Variables = \"X\",\"Y\",\"Z\",\"U\",\"V\",\"W\",\"QX\",\"QY\",\"QZ\",\"KNOB\" \n");
  fprintf(fp,"ZONE NODES=%d, DATAPACKING=POINT, ZONETYPE=FETRIANGLE\n\n",
      cel->vertex*cel->n);

  for(ic = 0;ic < cel->n;ic++){
    for(j = 0;j < cel->vertex;j++){
      fprintf(fp,"%10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %5d\n",
          cel->xH[ic*cel->vertex + j],
          cel->yH[ic*cel->vertex + j],
          cel->zH[ic*cel->vertex + j],
          cel->uH[ic*cel->vertex + j],
          cel->vH[ic*cel->vertex + j],
          cel->wH[ic*cel->vertex + j],
          fem->qi[ic*cel->vertex*3 + j*3 + 0],
          fem->qi[ic*cel->vertex*3 + j*3 + 1],
          fem->qi[ic*cel->vertex*3 + j*3 + 2],
          (ic < cel->num_irbc ? cel->is_knobH[j] : -1));
    }
  }
  for(ic = 0;ic < cel->n;ic++){
    for(e = 0;e < cel->element;e++){
      for(k = 0;k < TRI;k++){
        j1 = cel->eleH[(ic*cel->element + e)*TRI+k];
        j2 = cel->eleH[(ic*cel->element + e)*TRI+(k+1)%TRI];
        xx = fabs(cel->xH[j2] - cel->xH[j1]);
        zz = fabs(cel->zH[j2] - cel->zH[j1]);
        if(xx > LX*0.5 || zz > LZ*0.5){ flag = -1; break; }
        else                                      flag = 0;
      }
      if(flag == -1) continue;
      fprintf(fp,"%10d %10d %10d\n",
          cel->eleH[(ic*cel->element + e)*TRI  ] + 1,
          cel->eleH[(ic*cel->element + e)*TRI+1] + 1,
          cel->eleH[(ic*cel->element + e)*TRI+2] + 1);
    }
  }
  fclose(fp);
*/

  // Output Center of Capsules for volume correction
  OutputCapsuleCenter(prc, cdo, cel);
  /*
#if defined(CapsuleSHEARFLOW)
  //Output wall file of shear flow---

  sprintf(filename,"./result/wall%04d.dat",prc->fnum);
  fp = fopen(filename,"w"); if(fp == NULL) error(1);

  fprintf(fp,"I=%d,J=%d,K=%d\n",cdo->nx,2,cdo->nz);
  for(j = 0;j < cdo->nw;j++){
  fprintf(fp5,"%10.3e %10.3e %10.3e %4d\n",
  ltc->xwH[j],
  ltc->ywH[j],
  ltc->zwH[j],
  ltc->reonH[j]);
  }
  fclose(fp);
#elif defined(CapsuleCHANNELFLOW)
  //Output wall file of channel flow---

  sprintf(filename,"./result/wall%04d.dat",prc->fnum);
  fp = fopen(filename,"w"); if(fp == NULL) error(1);

  //    fprintf(fp,"I=%d,J=%d,K=%d\n",cdo->nx,2,cdo->nz);
  for(j = 0;j < cdo->nw;j++){
  fprintf(fp,"%10.3e %10.3e %10.3e\n",
  ltc->xwH[j],
  ltc->ywH[j],
  ltc->zwH[j]);
  }
  fclose(fp);
#endif
   */
  // Output cytoadhesion file ---
//  int flagc = cytoadhe,flagr = rosette;

//  if(flagc == 1){
/*
    #if (CYTOADHESION == true)
    int       spring, idNode, id_wall;
    double    xxx, yyy, zzz, d;

    sprintf(filename,"./result/adhesion/adhe%04d.dat",prc->fnum);
    fp = fopen(filename,"w"); if(fp == NULL) error(1);
    fprintf(fp,"Variables = \"X\",\"Y\",\"Z\",\"d\",\"WallID\",\"Xw\",\"Yw\",\"Zw\",\"NodeID\"\n");

    for(ic = 0;ic < cel->adnum;ic++){
      for(j = 0;j < cel->advertex;j++){
        idNode = ic*cel->advertex + j;
        for(spring = 0;spring < NUMSPRING_KNOB;spring++){
          id_wall = cel->attachment_pointH[idNode*NUMSPRING_KNOB + spring];
          if (id_wall == -1) continue;

          xxx = cel->xH[idNode] - wall_host->x[id_wall*3 + 0];
          yyy = cel->yH[idNode] - wall_host->x[id_wall*3 + 1];
          zzz = cel->zH[idNode] - wall_host->x[id_wall*3 + 2];
          d = sqrt(xxx*xxx + yyy*yyy + zzz*zzz);
          fprintf(fp,"%10.3e %10.3e %10.3e %10.3e %10d %10.3e %10.3e %10.3e %d\n",
              cel->xH[idNode],
              cel->yH[idNode],
              cel->zH[idNode],
              d,
              id_wall,
              wall_host->x[id_wall*3 + 0],
              wall_host->x[id_wall*3 + 1],
              wall_host->x[id_wall*3 + 2],
              j);
        }    
      }
    }
    fclose(fp);
    #endif
*/

//  validation   
//  OutputLatticeData(prc, cdo, ltc);

  if (prc->fnum%100 == 0) sprintf(filename,"./result/restart%04d.bin",prc->fnum);
  else                    sprintf(filename,"./result/restart.bin");
  fp = fopen(filename,"wb"); if(fp == NULL) error(1);

  // information of computation process
  fwrite(&prc->iter, sizeof(int),1, fp);
  fwrite(&prc->fnum, sizeof(int),1, fp);

  // information of lattice Boltzmann nodes
  fwrite(ltc->fnH,  sizeof(double),cdo->n*Q, fp);
  fwrite(ltc->dmH,  sizeof(double),cdo->n,   fp);
  fwrite(ltc->umH,  sizeof(double),cdo->n,   fp);
  fwrite(ltc->vmH,  sizeof(double),cdo->n,   fp);
  fwrite(ltc->wmH,  sizeof(double),cdo->n,   fp);
  fwrite(ltc->vfH,  sizeof(double),cdo->n,   fp);
  fwrite(ltc->vfH2, sizeof(double),cdo->n,   fp);

  // information of cell
  #if(N_R > 0)
  fwrite(cel->xH,  sizeof(double),cel->n*cel->vertex,    fp);
  fwrite(cel->yH,  sizeof(double),cel->n*cel->vertex,    fp);
  fwrite(cel->zH,  sizeof(double),cel->n*cel->vertex,    fp);
  fwrite(cel->rxH, sizeof(double),cel->n,                fp);
  fwrite(cel->ryH, sizeof(double),cel->n,                fp);
  fwrite(cel->rzH, sizeof(double),cel->n,                fp);
  fwrite(cel->uH,  sizeof(double),cel->n*cel->vertex,    fp);
  fwrite(cel->vH,  sizeof(double),cel->n*cel->vertex,    fp);
  fwrite(cel->wH,  sizeof(double),cel->n*cel->vertex,    fp);
  fwrite(fem->qi,  sizeof(double),cel->n*cel->mn,        fp);
  fwrite(fem->tp,  sizeof(double),cel->n*cel->element*2, fp);
  #endif
  #if(N_W > 0)
  fwrite(cel->xH_w,  sizeof(double),cel->n_w*cel->vertex_w,    fp);
  fwrite(cel->yH_w,  sizeof(double),cel->n_w*cel->vertex_w,    fp);
  fwrite(cel->zH_w,  sizeof(double),cel->n_w*cel->vertex_w,    fp);
  fwrite(cel->rxH_w, sizeof(double),cel->n_w,                  fp);
  fwrite(cel->ryH_w, sizeof(double),cel->n_w,                  fp);
  fwrite(cel->rzH_w, sizeof(double),cel->n_w,                  fp);
  fwrite(cel->uH_w,  sizeof(double),cel->n_w*cel->vertex_w,    fp);
  fwrite(cel->vH_w,  sizeof(double),cel->n_w*cel->vertex_w,    fp);
  fwrite(cel->wH_w,  sizeof(double),cel->n_w*cel->vertex_w,    fp);
  fwrite(fem->qi_w,  sizeof(double),cel->n_w*cel->mn_w,        fp);
  fwrite(fem->tp_w,  sizeof(double),cel->n_w*cel->element_w*2, fp);
  #endif
  #if(CYTOADHESION == true)
  fwrite(cel->f_adhH,            sizeof(double), cel->adnum*cel->advertex*3,              fp);
  fwrite(cel->is_knobH,          sizeof(bool  ), cel->adnum*cel->advertex,                fp);
  fwrite(cel->attachment_pointH, sizeof(int   ), cel->adnum*cel->advertex*NUMSPRING_KNOB, fp);
  fwrite(wall_host->is_attached, sizeof(int   ), wall_host->num_wall,                     fp);
  #endif

  fclose(fp);

// ==================================================
// Area & volume calculation of Capsule
// ==================================================
//  valiable of area caluculation

    int           i, j,  ea, eb, ec, tv = 0;
    double        x0, y0, z0, x1, y1, z1, x2, y2, z2,
                  x01, y01, z01, x02, y02, z02,
                  vec_x, vec_y, vec_z, norm, ds, r[3], uvec[3],
	          S = 0.0, V = 0.0, Vt = 0.0, Hct = 0.0;
    static double dt = 0.0; 

    // calculating nondimensional time
    if (prc->iter != 0) dt += cdo->dt; 

    #if(N_R > 0)
    // calculating the area of a RBC
    for (i = 0;i < cel->n;i++) {
      for (j = 0; j < cel->element; j++) {

        ea = cel->eleH[j*TRI+0] + i*cel->vertex;
        eb = cel->eleH[j*TRI+1] + i*cel->vertex;
        ec = cel->eleH[j*TRI+2] + i*cel->vertex;

        x0 = cel->xH[ea];  x1 = cel->xH[eb];  x2 = cel->xH[ec];
        y0 = cel->yH[ea];  y1 = cel->yH[eb];  y2 = cel->yH[ec];
        z0 = cel->zH[ea];  z1 = cel->zH[eb];  z2 = cel->zH[ec];

        x01 = x1 - x0;     x02 = x2 - x0;   
        y01 = y1 - y0;     y02 = y2 - y0;
        z01 = z1 - z0;     z02 = z2 - z0;

        vec_x = y01*z02 - z01*y02;
        vec_y = z01*x02 - x01*z02;
        vec_z = x01*y02 - y01*x02;
        norm  = sqrt(vec_x*vec_x + vec_y*vec_y + vec_z*vec_z);
    
        ds  = 0.5*norm;
        S  += ds;

        uvec[0] = vec_x/norm;
        uvec[1] = vec_y/norm;
        uvec[2] = vec_z/norm;

        r[0] = (x0 + x1 + x2)/3.0;
        r[1] = (y0 + y1 + y2)/3.0;
        r[2] = (z0 + z1 + z2)/3.0;

        Vt  += (r[0]*uvec[0] + r[1]*uvec[1] + r[2]*uvec[2])*ds;
      }
    }
    Vt /= 3.0;

    // Caluculating the volume of capsules using VOF
    for (i = 0; i < cdo->n; i++) {
      V += ltc->vfH[i]; if (ltc->bcH[i] == COMP) tv++;
    }
    Hct = V/(double)tv*1.0e+02;

    // Output surface and volume of capsules
    sprintf(filename,"./surface_volume.dat");
    if (prc->fnum == 0) {
      fp = fopen(filename,"w"); if (fp == NULL) error(1);
      fprintf(fp,"%10.3e %10.3e %10.3e %10.3e %10.3e\n",dt,S,V,Vt,Hct);
    } else {
      fp = fopen(filename,"a"); if (fp == NULL) error(1);
      fprintf(fp,"%10.3e %10.3e %10.3e %10.3e %10.3e\n",dt,S,V,Vt,Hct); 
    }
    fclose(fp);
    #endif 

    #if(N_W > 0 || N_C > 0 || N_P > 0)
    int           e, nc, nv, ne, tv = 0, tv2 = 0;
    static int    *ele;
    double        S2 = 0.0, V2 = 0.0, Vt2 = 0.0, Hct2 = 0.0;
    static double *x, *y, *z, 

    // calculating the area of other cell(WBC or Platelet or Cancer)
         if (cel->n_w > 0) { nc = cel->n_w; nv = cel->vertex_w; ne = cel->element_w; }
    else if (cel->n_p > 0) { nc = cel->n_p; nv = cel->vertex_p; ne = cel->element_p; }
    else if (cel->n_c > 0) { nc = cel->n_c; nv = cel->vertex_c; ne = cel->element_c; }

    if (prc->iter == 0) {
      x   = (double *)malloc(sizeof(double)*nc*nv    );
      y   = (double *)malloc(sizeof(double)*nc*nv    );
      z   = (double *)malloc(sizeof(double)*nc*nv    );
      ele = (int    *)malloc(sizeof(int   )*nc*ne*TRI);
    }

    // Storage the data about another capsule (WBC)
    for (i = 0; i < cel->n_w; i++) {
      for (j = 0;j < cel->vertex_w;j++) {
        x[i*nv+j] = cel->xH_w[i*nv+j];
        y[i*nv+j] = cel->yH_w[i*nv+j];
        z[i*nv+j] = cel->zH_w[i*nv+j];
      }
    }
    for (i = 0; i < cel->n_w; i++) {
      for (e = 0; e < cel->element_w; e++) {
        ele[i*TRI*ne + TRI*e+0] = cel->eleH_w[i*TRI*ne + TRI*e+0];
        ele[i*TRI*ne + TRI*e+1] = cel->eleH_w[i*TRI*ne + TRI*e+1];
        ele[i*TRI*ne + TRI*e+2] = cel->eleH_w[i*TRI*ne + TRI*e+2];
      }
    }

    for (i = 0;i < nc;i++) {
      for (j = 0; j < ne; j++) {
        ea = ele[j*TRI+0] + i*nv;
        eb = ele[j*TRI+1] + i*nv;
        ec = ele[j*TRI+2] + i*nv;

        x0 = x[ea];     x1 = x[eb];     x2 = x[ec];
        y0 = y[ea];     y1 = y[eb];     y2 = y[ec];
        z0 = z[ea];     z1 = z[eb];     z2 = z[ec];

        x01 = x1 - x0;  x02 = x2 - x0; 
        y01 = y1 - y0;  y02 = y2 - y0;
        z01 = z1 - z0;  z02 = z2 - z0;

        vec_x = y01*z02 - z01*y02;
        vec_y = z01*x02 - x01*z02;
        vec_z = x01*y02 - y01*x02;
        norm  = sqrt(vec_x*vec_x + vec_y*vec_y + vec_z*vec_z);
    
        ds  = 0.5*norm;
        S2 += ds;

        uvec[0] = vec_x/norm;
        uvec[1] = vec_y/norm;
        uvec[2] = vec_z/norm;

        r[0] = (x0 + x1 + x2)/3.0;
        r[1] = (y0 + y1 + y2)/3.0;
        r[2] = (z0 + z1 + z2)/3.0;

        Vt2 += (r[0]*uvec[0] + r[1]*uvec[1] + r[2]*uvec[2])*ds;
      }
    }
    Vt2 /= 3.0;

    // Caluculating the volume of capsules using VOF
    for (i = 0; i < cdo->n; i++) {
      V2 += ltc->vfH2[i]; if(ltc->bcH[i] == COMP) tv2++;
    }
    Hct2 = V2/(double)tv2*1.0e+02;

    // Output surface and volume of capsules
    sprintf(filename,"./surface_volume.dat");
    if (prc->fnum == 0) {
      fp = fopen(filename,"w"); if (fp == NULL) error(1);
      fprintf(fp,"%10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e\n",dt,S,S2,V,Vt,V2,Vt2,Hct,Hct2);
    } else {
      fp = fopen(filename,"a"); if (fp == NULL) error(1);
      fprintf(fp,"%10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e\n",dt,S,S2,V,Vt,V2,Vt2,Hct,Hct2); 
    }
    fclose(fp);
    #endif 

//    printf("\n");
//    printf("Volume of a Capsule = %10.3e  Volume of Domain = %d  Hct = %10.3e\n",V,tv,Hct);
//    printf("Hct = 40% -> n = %d\n",(int)floor(40.0/Hct));
//    printf("Hct = 30% -> n = %d\n",(int)floor(30.0/Hct));
//    printf("Hct = 20% -> n = %d\n",(int)floor(20.0/Hct));
//    printf("Hct = 10% -> n = %d\n",(int)floor(10.0/Hct));
//    exit(0);
#endif

  prc->fnum++;

  return;
}

