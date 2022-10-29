#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "def.h"


template <unsigned int blockSize>
__global__ void
reduce5(double *g_idata, double *g_odata)
{
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i+blockSize];
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; __syncthreads(); }
        if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; __syncthreads(); }
        if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; __syncthreads(); }
        if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; __syncthreads(); }
        if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; __syncthreads(); }
        if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; }
    }
    
    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


template <unsigned int blockSize>
__global__ void
dot5(double *d_i1, double *d_i2, double *d_o)
{
    extern __shared__ double sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    sdata[tid] = d_i1[i]*d_i2[i] + d_i1[i+blockSize]*d_i2[i+blockSize];
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; __syncthreads(); }
        if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; __syncthreads(); }
        if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; __syncthreads(); }
        if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; __syncthreads(); }
        if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; __syncthreads(); }
        if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; }
    }
    
    // write result for this block to global mem 
    if (tid == 0) d_o[blockIdx.x] = sdata[0];
}


void  dot
//==========================================================
//
//
//
//
(
    int       size,
    int       threads, 
    int       blocks,
    double    *d_i1,
    double    *d_i2, 
    double    *d_o
)
//----------------------------------------------------------
{
    dim3      dimBlock(threads, 1, 1);
    dim3      dimGrid(blocks, 1, 1);
    unsigned int  smemSize = threads * sizeof(double);
    if( threads <= 32) smemSize = smemSize*2;
//    int       smemSize = threads * sizeof(double);


    switch (threads)
    {
    case 512:
        dot5<512><<< dimGrid, dimBlock, smemSize >>>(d_i1, d_i2, d_o); break;
    case 256:
        dot5<256><<< dimGrid, dimBlock, smemSize >>>(d_i1, d_i2, d_o); break;
    case 128:
        dot5<128><<< dimGrid, dimBlock, smemSize >>>(d_i1, d_i2, d_o); break;
    case 64:
        dot5< 64><<< dimGrid, dimBlock, smemSize >>>(d_i1, d_i2, d_o); break;
    case 32:
        dot5< 32><<< dimGrid, dimBlock, smemSize >>>(d_i1, d_i2, d_o); break;
    case 16:
        dot5< 16><<< dimGrid, dimBlock, smemSize >>>(d_i1, d_i2, d_o); break;
    case  8:
        dot5<  8><<< dimGrid, dimBlock, smemSize >>>(d_i1, d_i2, d_o); break;
    case  4:
        dot5<  4><<< dimGrid, dimBlock, smemSize >>>(d_i1, d_i2, d_o); break;
    case  2:
        dot5<  2><<< dimGrid, dimBlock, smemSize >>>(d_i1, d_i2, d_o); break;
    case  1:
        dot5<  1><<< dimGrid, dimBlock, smemSize >>>(d_i1, d_i2, d_o); break;
    }
}


void  reduce
//==========================================================
//
//
//
//
(
    int       size,
    int       threads, 
    int       blocks,
    double    *d_idata, 
    double    *d_odata
)
//----------------------------------------------------------
{
    dim3      dimBlock(threads, 1, 1);
    dim3      dimGrid(blocks, 1, 1);
    unsigned int smemSize = threads * sizeof(double);
    if( threads <= 32) smemSize = smemSize*2;
//    int       smemSize = threads * sizeof(double);


    switch (threads)
    {
    case 512:
        reduce5<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
    case 256:
        reduce5<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
    case 128:
        reduce5<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
    case 64:
        reduce5< 64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
    case 32:
        reduce5< 32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
    case 16:
        reduce5< 16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
    case  8:
        reduce5<  8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
    case  4:
        reduce5<  4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
    case  2:
        reduce5<  2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
    case  1:
        reduce5<  1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata); break;
    }
}


void  getNumBlocksAndThreads
//==========================================================
//
//
//
//
(
    int       n, 
    int       maxThreads, 
    int       &blocks, 
    int       &threads
)
//----------------------------------------------------------
{
    if (n == 1) threads = 1;
    else        threads = (n < maxThreads*2) ? n / 2 : maxThreads;
    blocks = n / (threads * 2);
    if(blocks > 65535){ printf("Reduciton Block Overflow!\n"); exit(1); }
}


double  dot_product
//==========================================================
//
//
//
//
(
    double    *d_i1,
    double    *d_i2, 
    double    *d_o, 
    int       n
)
//----------------------------------------------------------
{
    int       maxThreads = 256, threads, blocks;
    double    gpu_result = 0.0;


    getNumBlocksAndThreads(n, maxThreads, blocks, threads);
    dot(n, threads, blocks, d_i1, d_i2, d_o);

    n = blocks;
    while(n > 1){
        threads = 0; blocks = 0;
        getNumBlocksAndThreads(n, maxThreads, blocks, threads);
        reduce(n, threads, blocks, d_o, d_o);
        n = n / (threads*2);
    }
    cudaMemcpy(&gpu_result,d_o,sizeof(double),cudaMemcpyDeviceToHost);

    return gpu_result;
}

