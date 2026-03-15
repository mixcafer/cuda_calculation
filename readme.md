# CUDA Matrix Multiplication

本项目实现了基于 CUDA 的矩阵乘法。
1. 实现共享内存，向量化加载，bank conflict避免优化

## 编译方法

1. 确保已安装 [CUDA Toolkit]和c++编译器。
2. 在项目根目录下运行以下命令：

```bash
mkdir build
python setup.py build
```
