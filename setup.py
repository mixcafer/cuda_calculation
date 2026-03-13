from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cuda_matmul',
    ext_modules=[
        CUDAExtension(
            name='cuda_matmul',
            sources=[
                'cuda/matmul.cpp',
                'cuda/matmul_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)