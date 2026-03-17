#include "matmul.h"

torch::Tensor matmul_forward(
    torch::Tensor A,
    torch::Tensor B)
{

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M,N},A.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        A.scalar_type(),
        "matmul_cuda",
        ([&]
        {
            matmul_cuda_dispatch<scalar_t>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                M,K,N
            );
        })
    );

    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("matmul", &matmul_forward);
}