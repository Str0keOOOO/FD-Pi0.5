import json
import os
from pathlib import Path

import h5py
import numpy as np
import tyro
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def main(input_dir: str, output_dir: str):
    """
    将带力矩的 LIBERO HDF5 数据集转换为纯本地的 LeRobot 格式。

    :param data_dir: 包含预处理后 HDF5 文件的输入目录。
    :param output_dir: 生成的 LeRobot 数据集要保存的本地完整路径。
    """
    print(f"初始化本地 LeRobot 数据集，保存路径: {output_dir}")
    dataset = LeRobotDataset.create(
        repo_id="fd-pi0.5/libero_10_force_lerobot",
        root=output_dir,
        robot_type="panda",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["x", "y", "z", "rx", "ry", "rz", "gripper_0", "gripper_1"],
            },
            "force": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["fx", "fy", "fz", "tx", "ty", "tz"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["x", "y", "z", "rx", "ry", "rz", "gripper"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    hdf5_files = [f for f in os.listdir(input_dir) if f.endswith(".hdf5")]
    print(f"找到 {len(hdf5_files)} 个 HDF5 文件待处理...")

    for file_name in hdf5_files:
        file_path = os.path.join(input_dir, file_name)

        with h5py.File(file_path, "r") as f:
            demos = list(f["data"].keys())
            for demo in demos:
                images = f[f"data/{demo}/obs/agentview_rgb"][:]
                wrist_images = f[f"data/{demo}/obs/eye_in_hand_rgb"][:]
                images = np.stack(
                    [
                        np.asarray(
                            Image.fromarray(frame.astype(np.uint8)[::-1, ::-1]).resize((256, 256), resample=Image.BILINEAR),
                            dtype=np.uint8,
                        )
                        for frame in images
                    ],
                    axis=0,
                )
                wrist_images = np.stack(
                    [
                        np.asarray(
                            Image.fromarray(frame.astype(np.uint8)[::-1, ::-1]).resize((256, 256), resample=Image.BILINEAR),
                            dtype=np.uint8,
                        )
                        for frame in wrist_images
                    ],
                    axis=0,
                )

                ee_states = f[f"data/{demo}/obs/ee_states"][:]
                gripper_states = f[f"data/{demo}/obs/gripper_states"][:]
                states = np.concatenate([ee_states, gripper_states], axis=-1).astype(np.float32)

                actions = f[f"data/{demo}/actions"][:].astype(np.float32)
                forces = f[f"data/{demo}/obs/robot0_eef_force"][:].astype(np.float32)

                env_args = f["data"].attrs.get("env_args", "{}")
                env_args = json.loads(env_args)["bddl_file"]
                no_suffix = Path(env_args).stem
                task_msg = str(no_suffix)
                print(f"Processing demo: {demo}, task info: {task_msg}")

                num_steps = actions.shape[0]
                for t in range(num_steps):
                    dataset.add_frame(
                        {
                            "image": images[t],
                            "wrist_image": wrist_images[t],
                            "state": states[t],
                            "force": forces[t],
                            "actions": actions[t],
                            "task": task_msg,
                        }
                    )
                dataset.save_episode()

    print(f"\n✅ 转换完成！本地数据集已成功保存至: {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
