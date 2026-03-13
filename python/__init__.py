import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent.parent

# build目录
BUILD = ROOT / "build"

# 查找build下的lib目录
for p in BUILD.glob("lib*"):
    sys.path.append(str(p))

# 导入cuda模块
import cuda_matmul