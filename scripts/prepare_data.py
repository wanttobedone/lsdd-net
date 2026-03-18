#!/usr/bin/env python3
"""数据准备脚本: 扫描CSV, 划分训练/验证集, 计算标准化参数。

用法:
    python scripts/prepare_data.py --csv_dir data/raw/ --config configs/train_phase1.yaml
"""

import os
import sys
import json
import argparse
import glob

import yaml
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lsdd_net.dataset import load_csv, split_train_val, parse_episode_name
from lsdd_net.normalize import Normalizer


def main():
    parser = argparse.ArgumentParser(description="LSDD-Net 数据准备")
    parser.add_argument("--csv_dir", type=str, required=True, help="CSV文件目录")
    parser.add_argument("--config", type=str, required=True, help="训练配置YAML")
    args = parser.parse_args()

    # 加载配置
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 扫描CSV文件
    csv_pattern = os.path.join(args.csv_dir, "episode_*.csv")
    csv_files = sorted(glob.glob(csv_pattern))

    if not csv_files:
        print(f"错误: 在 {args.csv_dir} 中未找到 episode_*.csv 文件")
        sys.exit(1)

    print(f"找到 {len(csv_files)} 个CSV文件")

    # 解析各集条件
    for f in csv_files:
        info = parse_episode_name(f)
        print(f"  {os.path.basename(f)}: {info['trajectory']}/{info['wind']}/{info['bf']}")

    # 划分训练/验证集
    val_conditions = cfg["data"].get("val_conditions", [])
    train_files, val_files = split_train_val(csv_files, val_conditions)
    print(f"\n训练集: {len(train_files)} 集")
    print(f"验证集: {len(val_files)} 集")

    if val_files:
        print("验证集文件:")
        for f in val_files:
            print(f"  {os.path.basename(f)}")

    # 加载训练集数据, 计算标准化参数
    print("\n加载训练数据并计算标准化参数...")
    train_arrays = []
    total_frames = 0
    for f in train_files:
        data = load_csv(f)
        train_arrays.append(data)
        total_frames += data.shape[0]

    normalizer = Normalizer()
    normalizer.fit_from_arrays(train_arrays)

    # 保存标准化参数
    norm_path = cfg["data"]["norm_stats_path"]
    os.makedirs(os.path.dirname(norm_path), exist_ok=True)
    normalizer.save(norm_path)
    print(f"标准化参数已保存: {norm_path}")

    # 打印标准化统计
    print("\n各通道统计量:")
    for name, stats in normalizer.stats.items():
        mean = stats["mean"].numpy()
        std = stats["std"].numpy()
        print(f"  {name:>8s}: mean={mean}, std={std}")

    # 计算窗口数量
    window_length = cfg["data"]["window_length"]
    stride = cfg["data"]["stride"]
    n_train_windows = sum(
        max(0, (arr.shape[0] - window_length) // stride + 1) for arr in train_arrays
    )

    val_arrays = [load_csv(f) for f in val_files]
    n_val_windows = sum(
        max(0, (arr.shape[0] - window_length) // stride + 1) for arr in val_arrays
    )

    # 保存数据清单
    manifest = {
        "csv_dir": os.path.abspath(args.csv_dir),
        "train_files": [os.path.abspath(f) for f in train_files],
        "val_files": [os.path.abspath(f) for f in val_files],
        "norm_stats_path": os.path.abspath(norm_path),
        "window_length": window_length,
        "stride": stride,
        "total_train_frames": total_frames,
        "n_train_windows": n_train_windows,
        "n_val_windows": n_val_windows,
    }

    manifest_path = cfg["data"]["manifest_path"]
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n数据清单已保存: {manifest_path}")

    # 摘要
    print(f"\n========= 数据准备完成 =========")
    print(f"总帧数: {total_frames} ({total_frames / 100:.1f} 秒)")
    print(f"训练窗口: {n_train_windows}")
    print(f"验证窗口: {n_val_windows}")
    print(f"窗口长度: {window_length} 步 ({window_length / 100:.1f} 秒)")
    print(f"窗口步长: {stride} 步 ({stride / 100:.1f} 秒)")


if __name__ == "__main__":
    main()
