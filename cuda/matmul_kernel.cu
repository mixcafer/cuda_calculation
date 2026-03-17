#include "matmul.h"
#include <cuda_runtime.h>


#define TILE 32

template<typename scalar_t, typename VecType, int VEC>
__global__ void matmul_kernel_vec(
    const scalar_t*  __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int M,
    int K,
    int N
){
    __shared__ scalar_t As[TILE][TILE+1];
    __shared__ scalar_t Bs[TILE][TILE+1];

    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;

    int row = by*TILE + ty;
    int col = bx*TILE + tx*VEC;
    scalar_t val[VEC] = {0};

    for(int t = 0; t < (K + TILE - 1) / TILE; t++){
        int tile_k = t*TILE;
        // load matrix A
        if(row < M && tile_k+tx*VEC + VEC - 1 < K){
            VecType a = *reinterpret_cast<const VecType*>(
                &A[row*K+tile_k+tx*VEC]
            );

            scalar_t* a_ptr = (scalar_t*)&a; //vectorizaiton load
            for(int i=0;i<VEC;i++){
                As[ty][tx*VEC+i] = a_ptr[i];
            }
        }
        else{
            int base = tile_k+tx*VEC;
            if(row < M){
                for(int i=0;i<K-base;i++){
                    As[ty][tx] = A[row*K+base+i];
                }
            }
            else{
                As[ty][tx] = 0;
            }
        }
        // load matrix B
        if(col + VEC - 1 < N && tile_k + ty < K){
            VecType b = *reinterpret_cast<const VecType*> (
                &B[(tile_k+ty)*N+col]
            );

            scalar_t* b_ptr = (scalar_t*)&b; //vectorization load

            for(int i=0;i<VEC;i++){
                Bs[ty][tx*VEC+i] = b_ptr[i];
            }
        }
        else{
            if(tile_k + ty<K){
                for(int i=0;i<N-col;i++){
                    int idx = col + i;
                    Bs[ty][tx] = B[(tile_k+ty)*N+idx];
                }
            }
            else {
                Bs[ty][tx] = 0;
            }
        }
        __syncthreads();//synchronize until all threads accomplish loading
        
        for(int i = 0;i < TILE; i++){
            scalar_t a = As[ty][i];
            for (int v = 0; v < VEC; v++)
            {
                val[v] += a*Bs[i][tx*VEC+v];   
            }
        }

        __syncthreads();
    }
    if(row < M)
    {
        for(int v=0;v<VEC;v++)
        {
            if(col+v < N)
                C[row*N + col+v] = val[v];
        }
    }
}


template<typename scalar_t>
void matmul_cuda_dispatch(
    const scalar_t* A,
    const scalar_t* B,
    scalar_t* C,
    int M, int K, int N)
{
    dim3 grid(
        (N+TILE-1)/TILE,
        (M+TILE-1)/TILE
    );

    if constexpr(std::is_same_v<scalar_t,float>)
    {
        dim3 block(TILE/4, TILE);
        matmul_kernel_vec<
            float,
            float4,
            4
        ><<<grid,block>>>(A,B,C,M,K,N);
    }

    else if constexpr(std::is_same_v<scalar_t,double>)
    {
        dim3 block(TILE/2, TILE);
        matmul_kernel_vec<
            double,
            double2,
            2
        ><<<grid,block>>>(A,B,C,M,K,N);
    }
}