#!/usr/bin/env python

from setuptools import setup, find_packages
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, (
    "WCSAC is designed to work with Python 3.6 and greater. "
    + "Please install it before proceeding."
)

setup(
    name="wc_sac",
    version="0.0.0",
    packages=find_packages(include=["wc_sac", "wc_sac.*"]),
    python_requires=">=3.6",
    # 说明：
    # - 为了让 `pip install -e .` 在常见环境中顺利完成，这里只保留最小依赖（数据预处理/大多数工具仅需 numpy）。
    # - 强依赖（tensorflow1 / mujoco_py / mpi4py 等）会在很多 Windows/新 Python 环境里无法安装，
    #   因此挪到 extras 里，按需安装。
    install_requires=[
        "numpy",
    ],
    extras_require={
        # 环境依赖（仅当你需要运行 Gym 环境时安装）
        "env": ["gym==0.15.3"],
        # 训练相关（TF1 + MPI）。注意：通常要求 Python 3.7 左右的旧环境。
        "train-tf1": ["scipy", "mpi4py", "tensorflow==1.15.5"],
    },
)
