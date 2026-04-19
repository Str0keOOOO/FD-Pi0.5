# FD-Pi0.5

## 模型介绍



## 安装

```bash
git clone --recurse-submodules https://github.com/Str0keOOOO/FD-Pi0.5.git

# Or if you already cloned the repo:
cd FD-Pi0.5
git submodule update --init --recursive
```

## 制作数据集

先去下载[LIBERO Datasets – LIBERO](https://libero-project.github.io/datasets)中的libero100，选取其中的libero10，或者去[yifengzhu-hf/LIBERO-datasets · Datasets at Hugging Face](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets)下载更好

```bash
uv venv --python 3.8 examples/libero/.venv

source examples/libero/.venv/bin/activate

uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match

uv pip install -e packages/openpi-client

uv pip install -e third_party/libero

uv pip install h5py==3.11.0

python examples/libero/convert_libero_force_data_to_libero_data.py ^
    --input_dir /path/to/libero_10 ^
    --output-dir /path/to/libero_10_with_force ^
    --bddl-dir third_party/libero/libero/libero/bddl_files/libero_10
```

这样通过原始的HDF5重新模拟仿真加入了末端力数据一共六维

再通过另一个环境

```
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

```bash
source .venv/bin/activate

python examples/libero/convert_libero_data_to_lerobot.py \
  --input-dir /mnt/c/Users/Administrator/Desktop/libero_100/libero_10_with_force \
  --output-dir /mnt/c/Users/Administrator/Desktop/libero_100/libero_10_with_force_lerobot
```

即可以制作出带有力的lerobot数据集用于微调模型

数据集格式如下

![image-20260416010701675](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260416010701675.png)

官方数据集如下https://huggingface.co/datasets/HuggingFaceVLA/libero

## 模型微调

| **模块名称**            | **参数路径正则 (Regex)** | **状态**   | **备注**                        |
| ----------------------- | ------------------------ | ---------- | ------------------------------- |
| **Tokenizer**           | N/A                      | **冻结**   | 无参数，纯逻辑处理              |
| **SigLIP**              | `.*img.*`                | **冻结**   | 保持强大的通用视觉感知力        |
| **VLM Backbone**        | `.*llm.*_0.*`            | **冻结**   | 锁死预训练的语言与语义知识      |
| **State Proj**          | `.*state_proj.*`         | **可学习** | 学习本体感觉的特征对齐          |
| **Force Proj**          | `.*force_proj.*`         | **可学习** | 学习 6 维物理力的特征拉伸       |
| **FDM (p / Attention)** | `.*fdm.*`                | **可学习** | **核心**：训练虚拟探针预测力    |
| **Action Expert**       | `.*llm.*_1.*`            | **可学习** | 学习如何将多模态 Token 转为动作 |
| **Action IO Proj**      | `.*action_.*_proj`       | **可学习** | 动作流匹配的线性投影层          |

不用 LoRA（直接进行模块化全量微调）

```bash
uv run scripts/compute_norm_stats.py pi05_libero_fdm
```

```bash
export WANDB_API_KEY=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
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_fdm_wo --exp-name=pi05_libero_fdm_wo --overwrite
```

# 实验验证

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```

## pi05_libero_fdm_wo 

```bash
MUJOCO_GL=glx python examples/libero/main.py python examples/libero/main.py --force_mode real --task_suite_name libero_10
```

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero_fdm_wo --policy.dir /你的/绝对/路径/checkpoints/pi05_libero_fdm_wo/xxx
```

## pi05_libero_fdm_sensor_free

```bash
python examples/libero/main.py --force_mode dummy --task_suite_name libero_10
```

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero_fdm_sensor_free --policy.dir /你的/绝对/路径/checkpoints/pi05_libero_fdm_sensor_free/xxx
```

## pi05_libero_fdm_wo 

```bash
python examples/libero/main.py --force_mode real --task_suite_name libero_10
```

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config pi05_libero_fdm_upper_bound --policy.dir /你的/绝对/路径/checkpoints/pi05_libero_fdm_upper_bound/xxx
```

