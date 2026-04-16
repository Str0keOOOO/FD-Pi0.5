# FD-Pi0.5

## 模型介绍



## 安装

```bash
git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git

# Or if you already cloned the repo:
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

Usage (tyro):
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

python examples/libero/convert_libero_data_to_lerobot.py ^
    --input_dir /path/to/libero_10_with_force ^
    --output-dir /path/to/libero_10_with_force_lerobot ^
    --bddl-dir third_party/libero/libero/libero/bddl_files/libero_10
```

即可以制作出带有力的lerobot数据集用于微调模型

数据集格式如下

![image-20260416010701675](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260416010701675.png)

官方数据集如下https://huggingface.co/datasets/HuggingFaceVLA/libero

## 模型微调