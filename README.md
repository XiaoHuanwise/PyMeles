# PyMeles

PyMeles 是一个基于Python的DG（Discontinuous Galerkin）可压缩流求解器。

## 目录
- [PyMeles](#pymeles)
  - [目录](#目录)
  - [安装](#安装)
    - [环境配置](#环境配置)
      - [使用conda配置环境](#使用conda配置环境)

## 安装

### 环境配置

本项目依赖于特定的 Python 包和科学计算库。推荐使用 conda 创建独立的虚拟环境来管理这些依赖。

#### 使用conda配置环境

1. 确保已安装 [Anaconda](https://www.anaconda.com/products/distribution) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)（也许需要更改清华或其他镜像源以获得更好体验）

2. 创建 conda 虚拟环境：
    ```bash
    # bash in linux, powershell in windows
    conda create -n pymeles python=3.14 -c conda-forge
    conda activate pymeles
    # 安装 pytorch （安装指令可参考 pythrch 官网 https://pytorch.org/get-started/locally/ ）
    # 下面关于 torch 的安装命令根据需要选择其中一个即可
    # 纯 cpu 版本大小较小（100M） gpu 版本大小较大（2G）
    # 纯 cpu 版本（不做深度学习视觉处理可不装 torchvision）
    pip3 install torch
    #带 gpu 版本（请根据情况自行选择合适的cuda版本）
    pip3 install torch --index-url https://download.pytorch.org/whl/cu130
    # 下载慢的话可以换源
    pip3 install torch --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu130
    # 安装其他依赖（下载慢的话可以换源）
    pip3 install numpy scipy matplotlib sympy numba intel-cmplr-lib-rt psutil
    # 若希望使用 conda 安装，请注意 pytorch 和其他依赖均使用 conda 安装
    # 否则可能会导致 import torch 时出现 libiomp5md.dll 冲突。
    # 若使用下面两个指令之一安装，则上面关于 pip3 install 的指令不需要运行
    # 使用 conda 安装带 gpu 版本可参考下面的命令
    conda install numpy scipy matplotlib sympy numba intel-cmplr-lib-rt psutil pytorch-gpu -c conda-forge
    # 使用 conda 安装纯 cpu 版本可参考下面的命令
    conda install numpy scipy matplotlib sympy numba intel-cmplr-lib-rt psutil pytorch-cpu -c conda-forge
