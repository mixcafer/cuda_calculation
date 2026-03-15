#include "matmul.h"
#include <cuda_runtime.h>


#define TILE 32
#define VEC 2

__global__ void matmul_kernel_vec(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    int M,
    int K,
    int N)
{
    __shared__ double As[TILE][TILE+1];
    __shared__ double Bs[TILE][TILE+1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx * VEC;

    double val[VEC] = {0.0,0.0};

    for(int t=0; t<(K+TILE-1)/TILE; t++)
    {
        int tiled_k = t*TILE;

        // vector load A
        if(row < M && tiled_k + tx*VEC < K)
        {
            double2 a = *reinterpret_cast<const double2*>(
                &A[row*K + tiled_k + tx*VEC]
            );

            As[ty][tx*VEC]   = a.x;
            As[ty][tx*VEC+1] = a.y;
        }
        else
        {
            As[ty][tx*VEC] = 0.0;
            As[ty][tx*VEC+1] = 0.0;
        }

        // vector load B
        if(col < N && tiled_k + ty < K)
        {
            double2 b = *reinterpret_cast<const double2*>(
                &B[(tiled_k + ty)*N + col]
            );

            Bs[ty][tx*VEC]   = b.x;
            Bs[ty][tx*VEC+1] = b.y;
        }
        else
        {
            Bs[ty][tx*VEC] = 0.0;
            Bs[ty][tx*VEC+1] = 0.0;
        }

        __syncthreads();

        for(int i=0;i<TILE;i++)
        {
            double a = As[ty][i];

            val[0] += a * Bs[i][tx*VEC];
            val[1] += a * Bs[i][tx*VEC+1];
        }

        __syncthreads();
    }

    if(row < M && col < N)
    {
        C[row*N + col] = val[0];
        if(col+1 < N)
            C[row*N + col + 1] = val[1];
    }
}


void matmul_cuda(
    const double* A,
    const double* B,
    double* C,
    int M,
    int K,
    int N)
{
    dim3 block(TILE/2,TILE);

    dim3 grid(
        (N+TILE-1)/TILE,
        (M+TILE-1)/TILE
    );

    matmul_kernel_vec<<<grid,block>>>(A,B,C,M,K,N);
}