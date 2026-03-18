"""LSDD-Net数据集: 加载CSV, 滑窗切割, 构建PyTorch Dataset。

CSV格式 (23列):
  timestamp, fw_x, fw_y, fw_z, fb_x, fb_y, fb_z,
  vw_x, vw_y, vw_z, vb_x, vb_y, vb_z,
  qw, qx, qy, qz,
  wind_gt_x, wind_gt_y, wind_gt_z, bf_gt_x, bf_gt_y, bf_gt_z
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional

from .normalize import Normalizer


def parse_episode_name(filename: str) -> Dict[str, str]:
    """从文件名解析数据条件。

    示例: episode_000_circle_wind_s0.7_bf0.8.csv
    返回: {"trajectory": "circle", "wind": "wind_s0.7", "bf": "bf0.8"}
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("_")

    # 跳过 "episode" 和 编号
    # 典型模式: episode_NNN_<trajectory>_<wind>_<bf>
    info = {"trajectory": "", "wind": "", "bf": "", "raw": name}

    # 提取轨迹类型
    traj_types = ["hover", "circle", "yaw", "figure8", "ellipse"]
    for i, p in enumerate(parts):
        if p in traj_types:
            info["trajectory"] = p
            # yaw_spin 是两个词
            if p == "yaw" and i + 1 < len(parts) and parts[i + 1] == "spin":
                info["trajectory"] = "yaw_spin"
            break

    # 提取风场条件
    if "nowind" in name:
        info["wind"] = "nowind"
    elif "wind" in name:
        # 匹配 wind_s0.7 或 wind_s1.4 等
        match = re.search(r"wind_s([\d.]+)", name)
        if match:
            info["wind"] = f"wind_s{match.group(1)}"

    # 提取机体力条件
    if "nobf" in name:
        info["bf"] = "nobf"
    else:
        match = re.search(r"bf([\d.]+)", name)
        if match:
            info["bf"] = f"bf{match.group(1)}"

    return info


def load_csv(path: str) -> np.ndarray:
    """加载CSV文件, 跳过header行。

    Returns:
        (N, 23) float32数组。
    """
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float32)
    assert data.shape[1] == 23, f"期望23列, 实际{data.shape[1]}列: {path}"
    return data


def split_train_val(
    csv_files: List[str],
    val_conditions: List[str],
) -> Tuple[List[str], List[str]]:
    """按条件组合划分训练/验证集。

    Args:
        csv_files: 所有CSV文件路径列表。
        val_conditions: 验证集条件关键词列表, 文件名中包含任一则为验证集。
            例: ["wind_s1.4_bf2.2", "nowind_bf1.5"]

    Returns:
        (train_files, val_files) 路径列表。
    """
    train, val = [], []
    for f in csv_files:
        name = os.path.basename(f)
        is_val = any(cond in name for cond in val_conditions)
        (val if is_val else train).append(f)
    return train, val


class LSDDDataset(Dataset):
    """LSDD-Net训练数据集。

    从CSV文件加载数据, 滑窗切割为固定长度序列。
    """

    def __init__(
        self,
        csv_files: List[str],
        window_length: int = 500,
        stride: int = 50,
        normalizer: Optional[Normalizer] = None,
    ):
        """
        Args:
            csv_files: CSV文件路径列表。
            window_length: 滑窗长度 (步数, 默认500=5秒@100Hz)。
            stride: 滑窗步长 (默认50=0.5秒)。
            normalizer: 标准化器 (如果为None则不做标准化)。
        """
        self.window_length = window_length
        self.normalizer = normalizer

        # 加载所有集, 建立窗口索引
        self.episodes: List[np.ndarray] = []
        self.windows: List[Tuple[int, int]] = []  # (episode_idx, start_frame)

        for path in csv_files:
            data = load_csv(path)
            episode_idx = len(self.episodes)
            self.episodes.append(data)

            # 滑窗切割
            n_frames = data.shape[0]
            if n_frames < window_length:
                continue
            for start in range(0, n_frames - window_length + 1, stride):
                self.windows.append((episode_idx, start))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个训练样本。

        Returns:
            字典: {
                "fw": (L, 3),      # 世界系MDOB力
                "fb": (L, 3),      # 机体系MDOB力
                "vw": (L, 3),      # 世界系速度
                "vb": (L, 3),      # 机体系速度
                "q":  (L, 4),      # 四元数 (w,x,y,z)
                "wind_gt": (L, 3), # 世界系风力真值
                "bf_gt":   (L, 3), # 机体系body force真值
            }
        """
        ep_idx, start = self.windows[idx]
        data = self.episodes[ep_idx]
        seg = data[start:start + self.window_length]  # (L, 23)

        # 按列索引切分
        sample = {
            "fw": torch.from_numpy(seg[:, 1:4].copy()),
            "fb": torch.from_numpy(seg[:, 4:7].copy()),
            "vw": torch.from_numpy(seg[:, 7:10].copy()),
            "vb": torch.from_numpy(seg[:, 10:13].copy()),
            "q":  torch.from_numpy(seg[:, 13:17].copy()),
            "wind_gt": torch.from_numpy(seg[:, 17:20].copy()),
            "bf_gt":   torch.from_numpy(seg[:, 20:23].copy()),
        }

        # 标准化 (四元数不处理)
        if self.normalizer is not None:
            for key in ["fw", "fb", "vw", "vb", "wind_gt", "bf_gt"]:
                sample[key] = self.normalizer.transform(key, sample[key])

        return sample


class EpisodeDataset(Dataset):
    """单集完整序列数据集, 用于评估/可视化 (不做滑窗切割)。"""

    def __init__(
        self,
        csv_files: List[str],
        normalizer: Optional[Normalizer] = None,
    ):
        self.normalizer = normalizer
        self.episodes = [load_csv(f) for f in csv_files]
        self.filenames = [os.path.basename(f) for f in csv_files]

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seg = self.episodes[idx]
        sample = {
            "fw": torch.from_numpy(seg[:, 1:4].copy()),
            "fb": torch.from_numpy(seg[:, 4:7].copy()),
            "vw": torch.from_numpy(seg[:, 7:10].copy()),
            "vb": torch.from_numpy(seg[:, 10:13].copy()),
            "q":  torch.from_numpy(seg[:, 13:17].copy()),
            "wind_gt": torch.from_numpy(seg[:, 17:20].copy()),
            "bf_gt":   torch.from_numpy(seg[:, 20:23].copy()),
            "timestamp": torch.from_numpy(seg[:, 0].copy()),
        }
        if self.normalizer is not None:
            for key in ["fw", "fb", "vw", "vb", "wind_gt", "bf_gt"]:
                sample[key] = self.normalizer.transform(key, sample[key])
        return sample
