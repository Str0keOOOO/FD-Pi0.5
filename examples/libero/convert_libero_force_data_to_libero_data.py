"""
Extract end-effector wrench (force + torque) from LIBERO HDF5 demos and write it back into the dataset.

This script loads each LIBERO `.hdf5` file, replays the recorded MuJoCo states in robosuite, and
computes the robot end-effector wrench at every timestep:
    wrench[t] = [Fx, Fy, Fz, Tx, Ty, Tz]

The output is still **LIBERO HDF5 format**: the script copies each input file into `output_dir`
and then adds a dataset at:
    data/<demo>/obs/robot0_eef_force   shape=(T, 6)

Usage (tyro):
python examples/libero/convert_libero_force_data_to_libero_data.py ^
    --input_dir /path/to/libero_10 ^
    --output-dir /path/to/libero_10_with_force ^
    --bddl-dir third_party/libero/libero/libero/bddl_files/libero_10

Notes:
- Requires: robosuite, libero, mujoco, h5py, numpy, tqdm, tyro
- `bddl_dir` should contain the matching `.bddl` files for the tasks in the dataset.
"""

import os
import json
import shutil
import h5py
import numpy as np
import robosuite as suite
import libero.libero.envs
from tqdm import tqdm
import tyro


def main(
    input_dir: str,
    output_dir: str,
    bddl_dir: str = "third_party/libero/libero/libero/bddl_files/libero_10",
):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    hdf5_files = [f for f in os.listdir(input_dir) if f.endswith(".hdf5")]
    print(f"找到 {len(hdf5_files)} 个 HDF5 文件待处理...")

    for file_name in hdf5_files:
        file_path = os.path.join(input_dir, file_name)
        out_file_path = os.path.join(output_dir, file_name)

        print("\n=====================================")
        print(f"正在处理: {file_name}")

        if not os.path.exists(out_file_path):
            shutil.copy(file_path, out_file_path)

        file_name_no_ext, _ = os.path.splitext(file_name)
        bddl_name = file_name_no_ext[:-5]

        bddl_file_path = os.path.join(bddl_dir, f"{bddl_name}.bddl")

        with h5py.File(out_file_path, "r+") as f:
            env_args = json.loads(f["data"].attrs["env_args"])
            env_name = env_args["env_name"]
            env_kwargs = env_args.get("env_kwargs", {})
            env_kwargs.pop("bddl_file_name", None)

            env = suite.make(env_name, **env_kwargs, bddl_file_name=bddl_file_path)

            demos = list(f["data"].keys())

            for demo in tqdm(demos, desc=f"提取 {file_name}"):
                states = f[f"data/{demo}/states"][:]
                env.reset()

                extracted_forces = []
                for t in range(states.shape[0]):
                    env.sim.set_state_from_flattened(states[t])
                    env.sim.forward()

                    force = env.robots[0].ee_force  # (Fx, Fy, Fz)
                    torque = env.robots[0].ee_torque  # (Tx, Ty, Tz)
                    eef_wrench = np.concatenate([force, torque], axis=-1)
                    extracted_forces.append(eef_wrench)

                forces_array = np.array(extracted_forces)  # (T, 6)

                # 将提取出的力写入新的 HDF5 的 obs 组中
                obs_group = f[f"data/{demo}/obs"]

                if "robot0_eef_force" in obs_group:
                    del obs_group["robot0_eef_force"]

                obs_group.create_dataset("robot0_eef_force", data=forces_array)

        print(f"✅ 文件 {file_name} 提取完成,力数据已注入。")


if __name__ == "__main__":
    tyro.cli(main)
