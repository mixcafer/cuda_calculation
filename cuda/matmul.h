#pragma once

#include <torch/extension.h>

void matmul_cuda(
    const float* A,
    const float* B,
    float* C,
    int M,
    int K,
    int N
);

torch::Tensor matmul_forward(
    torch::Tensor A,
    torch::Tensor B
);