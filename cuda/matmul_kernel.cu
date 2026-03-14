#include "matmul.h"
#include <cuda_runtime.h>

#define TILE 32

__global__ void matmul_kernel(
    const double* A,
    const double* B,
    double* C,
    int M,
    int K,
    int N)
{
    __shared__ double As[TILE][TILE+1];
    __shared__ double Bs[TILE][TILE+1];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    if (row >= M || col >= N) return;

    double val = 0;

    for(int t=0; t<(K+TILE-1)/TILE; t++)
    {

        if(row < M && t*TILE+threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] =
                A[row*K + t*TILE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0;

        if(col < N && t*TILE+threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] =
                B[(t*TILE+threadIdx.y)*N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for(int i=0;i<TILE;i++)
            val += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }
    C[row * N + col] = 0.0f; 
    if(row<M && col<N)
        C[row*N + col] = val;
}


void matmul_cuda(
    const double* A,
    const double* B,
    double* C,
    int M,
    int K,
    int N)
{
    dim3 block(TILE,TILE);

    dim3 grid(
        (N+TILE-1)/TILE,
        (M+TILE-1)/TILE
    );

    matmul_kernel<<<grid,block>>>(A,B,C,M,K,N);
}