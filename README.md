# WCSAC

## Installation & Operation

### 推荐：可编辑安装（用于迁移/开发）

在 `CVaR_SAC/` 目录下执行：

```bash
python -m pip install -e .
```

这样安装后你可以在任意目录 `import wc_sac`，并且修改源码无需重复安装。

### 依赖说明（推荐按用途安装）

本项目统一使用 `requirements.txt`（数据处理/环境/训练都在同一份里）。

```bash
python -m pip install -r requirements.txt
```

注意：训练依赖 TensorFlow 1.x，通常建议用 Conda 创建旧 Python（例如 3.7）环境；在新 Python 上 `requirements.txt` 会自动跳过 TF1（不影响数据处理/画图/环境）。

pip install tensorflow

- 下载并安装 Microsoft MPI (这是关键) 这是 Windows 上运行 MPI 程序的必要系统组件。

- 下载链接 : Microsoft MPI v10.1.3
- 操作 : 下载页面里的两个文件都要安装：
  - msmpisetup.exe (运行时)
  - msmpisdk.msi (SDK，编译依赖)
- 双击安装，一路 Next 即可。
- 重新安装 mpi4py 安装完 MS-MPI 后，使用 pip 安装 mpi4py，它会自动链接到刚安装的 MS-MPI：

```
pip install mpi4py
```
- 重启终端 安装完 MS-MPI 后，环境变量需要更新。 请务必关闭当前终端，然后重新打开一个终端 。