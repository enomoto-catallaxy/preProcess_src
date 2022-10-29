//----------------------------------------------------------
//
//  CUDA LATTICE BOLTZMANN CODE
//
//----------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "def.h"
//void  commandline(int argc,char **argv,int *restart,int *device);
//void  restart_flag (int restart, domain *cdo);
//void  gpu_init     (int device);

void  restart_flag (int argc, char **argv, domain *cdo);
void  gpu_init     (int argc, char **argv);


int  main(int argc, char *argv[])
{
    process        prc;
    domain         cdo;
    lattice        ltc;
    cell           cel;
    fem            fem;
    Wall           wall, wall_host;
//    double         elapsed_time;
//    unsigned int   timer;
//    int restart = 0, device = 0;
    int vol_iter = 0;


    // PREPROCESS ----------------------------------
//    commandline(argc,argv,&restart,&device);
//    gpu_init(device);
//    restart_flag(restart,&cdo);
    gpu_init(argc,argv);
    preprocess(&prc);
    restart_flag(argc,argv,&cdo);   
    // INITIALIZE ----------------------------------
    cdo_init(&cdo,&ltc);
    flu_init(&cdo,&ltc);
    // INITIALIZE CELL INFORMATION ABOUT RBC -------
    cel_init(&cdo,&cel);
    fem_init(&cel,&fem);
    // INITIALIZE CELL INFORMATION ABOUT WBC -------
    cel_init_wbc(&cdo,&cel);
    fem_init_wbc(&cel,&fem);
    // INITIALIZE MEMORY & VOF ---------------------
    mem_init(&cdo,&ltc,&cel,&fem);
    vof_init(&cdo,&ltc,&cel);
    // INITIALIZE ADHESION PROCEDURE ---------------
    wall_init(&cdo, &wall, &wall_host);
    adhesion_init(&cdo, &cel, &wall);
    volume_init(&cdo, &cel);
//    inflation_init(&cdo,&cel,&fem);
    // RESTART PROCESS -----------------------------
    restart_file(&prc,&cdo,&ltc,&cel,&fem,&wall,&wall_host);
    // DATA OUTPUT ---------------------
    output_data(&prc,&cdo,&ltc,&cel,&fem,&wall,&wall_host);

    // START TIMER --------------------------------
//    cutCreateTimer(&timer);
//    cutResetTimer(timer);
//    cutStartTimer(timer);

    // MAIN LOOP ----------------------------------
    while(prc.iter < ITRLIMIT){

        prc.iter++;
        if(prc.iter%STANDARDOUTPUT == 0)
        printf("iteration %10d\n",prc.iter);

        // FINITE ELEMENT METHOD
        if ((prc.iter-1)%M == 0) 
        stress(&prc,&cdo,&cel,&fem,&ltc);
        // ADHESION
        if ((prc.iter-1)%M == 0)
        Adhesion(&prc, &cdo, &cel, &wall, &wall_host);
        // IMMERSED BOUNDARY METHOD
        if ((prc.iter-1)%M == 0) 
        ibm_st1(&cdo,&ltc,&cel);
        // LATTICE BOLTZMANN METHOD
        lbm_d3q19(&prc,&cdo,&ltc);
        // IMMERSED BOUNDARY METHOD
        if ((prc.iter)%M == 0) 
        ibm_st2(&cdo,&ltc,&cel,&fem);
        // THINC/WLIC METHOD
        if ((prc.iter)%M == 0)
        thinc_wlic(&cdo,&ltc);
        // FRONT-TRACKING METHOD
        if ((prc.iter)%K == 0)
        front_tracking(&cdo,&ltc,&cel);

        // VOLUME CORRECTION
        if ((prc.iter)%M == 0) {
          vol_iter = volume_correction(&cdo, &cel, &fem);
          if (vol_iter != 0 && prc.iter%STANDARDOUTPUT == 0) {
            printf(" %d : volume correction %d\n", prc.iter, vol_iter);
          }
        }
        // DATA OUTPUT
        if (prc.iter%FILEOUTPUT == 0)
        output_data(&prc,&cdo,&ltc,&cel,&fem,&wall, &wall_host);
    }

    // STOP TIMER ---------------------------------
//    cutStopTimer(timer);
    // POSTPROCESS --------------------------------
//    elapsed_time = cutGetTimerValue(timer)*1.0e-03;
//    printf("Elapsed Time = %9.3e [sec]\n",elapsed_time);

    return 0;
}

