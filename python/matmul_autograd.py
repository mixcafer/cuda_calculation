import torch
import __init__
import cuda_matmul

class MatmulFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, B):

        ctx.save_for_backward(A,B)

        return cuda_matmul.matmul(A,B)

    @staticmethod 
    def backward(ctx, grad):

        A,B = ctx.saved_tensors

        gradA = cuda_matmul.matmul(grad, B.t())
        gradB = cuda_matmul.matmul(A.t(), grad)

        return gradA,gradB


def matmul(A,B):
    return MatmulFunction.apply(A,B)