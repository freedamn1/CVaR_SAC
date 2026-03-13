# WCSAC 抢占式云服务动态定价（PyTorch）

本项目已精简为仅支持抢占式云服务动态定价任务，不再包含机器人导航/Safety-Gym/dm-control 相关代码。

## 环境安装

```bash
conda env create -f conda_env.yml
conda activate pytorch_wcsac
```

## 数据准备

默认读取旧项目中的容量序列文件：

`../../wc_sac/dataset/excessive_capacity_cpu_300sec_azure.npz`

要求 NPZ 内包含键：`excessive_capacity_cpu`。

## 训练

```bash
python train.py
```

可以通过 Hydra 覆盖参数，例如：

```bash
python train.py seed=2 risk_level=0.1 num_train_steps=100000
python train.py excessive_capacity_npz=D:/project/Python_project/wcsac/WCSAC/wc_sac/dataset/excessive_capacity_cpu_300sec_azure.npz
```

## 核心配置

- 训练配置：`config/train_pricing.yaml`
- Agent 配置：`config/agent/wcsac_pricing.yaml`
- 定价环境：`envs/preemptive_pricing_env.py`
