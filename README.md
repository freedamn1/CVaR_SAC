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


