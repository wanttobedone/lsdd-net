"""通道级z-score标准化器。四元数不做标准化。"""

import json
import torch
from torch import Tensor
from typing import Dict


# 各通道分组定义：名称 → (CSV起始列索引, 维度)
CHANNEL_GROUPS = {
    "fw": (1, 3),       # 世界系MDOB力
    "fb": (4, 3),       # 机体系MDOB力
    "vw": (7, 3),       # 世界系速度
    "vb": (10, 3),      # 机体系速度
    "wind_gt": (17, 3), # 风力真值
    "bf_gt": (20, 3),   # 机体力真值
}

# 四元数 (13-16列) 不做标准化，已是单位四元数


class Normalizer:
    """通道级z-score标准化器。

    对力、速度等通道做 (x - mean) / std 标准化，
    四元数保持原值不处理。
    """

    def __init__(self):
        self.stats: Dict[str, Dict[str, Tensor]] = {}

    def fit_from_arrays(self, data_list: list):
        """从numpy数组列表中计算各通道的均值和标准差。

        Args:
            data_list: 元素为 (N, 23) numpy数组的列表，每个对应一个CSV文件。
        """
        import numpy as np

        # 拼接所有数据
        all_data = np.concatenate(data_list, axis=0)  # (总帧数, 23)

        for name, (start, dim) in CHANNEL_GROUPS.items():
            channel_data = all_data[:, start:start + dim]
            mean = channel_data.mean(axis=0).astype(np.float32)
            std = channel_data.std(axis=0).astype(np.float32)
            # 防止除零
            std = np.where(std < 1e-8, 1.0, std)
            self.stats[name] = {
                "mean": torch.from_numpy(mean),
                "std": torch.from_numpy(std),
            }

    def transform(self, name: str, x: Tensor) -> Tensor:
        """标准化: (x - mean) / std。

        Args:
            name: 通道组名称 (如 "fw", "vw")。
            x: (..., dim) 张量。

        Returns:
            标准化后的张量，形状不变。
        """
        s = self.stats[name]
        mean = s["mean"].to(x.device)
        std = s["std"].to(x.device)
        return (x - mean) / std

    def inverse_transform(self, name: str, x: Tensor) -> Tensor:
        """反标准化: x * std + mean。"""
        s = self.stats[name]
        mean = s["mean"].to(x.device)
        std = s["std"].to(x.device)
        return x * std + mean

    def save(self, path: str):
        """保存标准化参数到 .pt 文件。"""
        save_dict = {}
        for name, s in self.stats.items():
            save_dict[f"{name}_mean"] = s["mean"]
            save_dict[f"{name}_std"] = s["std"]
        torch.save(save_dict, path)

    def load(self, path: str):
        """从 .pt 文件加载标准化参数。"""
        save_dict = torch.load(path, map_location="cpu", weights_only=True)
        self.stats = {}
        for key in save_dict:
            if key.endswith("_mean"):
                name = key[:-5]
                self.stats[name] = {
                    "mean": save_dict[f"{name}_mean"],
                    "std": save_dict[f"{name}_std"],
                }
