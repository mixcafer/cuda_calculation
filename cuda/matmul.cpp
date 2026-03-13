#include "matmul.h"

torch::Tensor matmul_forward(
    torch::Tensor A,
    torch::Tensor B)
{

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M,N},A.options());

    matmul_cuda(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M,K,N
    );

    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul", &matmul_forward);
}