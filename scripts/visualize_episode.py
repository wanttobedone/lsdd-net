#!/usr/bin/env python3
"""单集可视化: 加载一个CSV + checkpoint, 逐帧推理并绘制时序对比图。

用法:
    python scripts/visualize_episode.py \
        --csv data/raw/episode_000_circle_wind_s1.4_bf1.5.csv \
        --checkpoint checkpoints/phase1_best.pt \
        --norm_stats checkpoints/norm_stats.pt
"""

import os
import sys
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lsdd_net.model import LSDDNet
from lsdd_net.dataset import load_csv
from lsdd_net.normalize import Normalizer
from lsdd_net.rotation_utils import quat_to_rotmat, rotate_vector


def main():
    parser = argparse.ArgumentParser(description="单集可视化")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--norm_stats", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="输出图片路径")
    args = parser.parse_args()

    device = torch.device("cpu")  # 可视化不需要GPU

    # 加载模型
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = ckpt.get("model_config", ckpt.get("config", {}).get("model", {}))
    model = LSDDNet(**model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 标准化器
    normalizer = Normalizer()
    normalizer.load(args.norm_stats)

    # 加载数据
    data = load_csv(args.csv)
    N = data.shape[0]
    print(f"加载: {args.csv}, {N}帧 ({N/100:.1f}秒)")

    # 逐帧step推理
    preds_w, preds_b = [], []
    states = None

    with torch.no_grad():
        for t in range(N):
            fw = torch.from_numpy(data[t, 1:4]).unsqueeze(0)
            fb = torch.from_numpy(data[t, 4:7]).unsqueeze(0)
            vw = torch.from_numpy(data[t, 7:10]).unsqueeze(0)
            vb = torch.from_numpy(data[t, 10:13]).unsqueeze(0)
            q  = torch.from_numpy(data[t, 13:17]).unsqueeze(0)

            # 标准化
            fw_n = normalizer.transform("fw", fw)
            fb_n = normalizer.transform("fb", fb)
            vw_n = normalizer.transform("vw", vw)
            vb_n = normalizer.transform("vb", vb)

            F_hat_w, F_hat_b, states = model.step(fw_n, fb_n, vw_n, vb_n, q, states)

            # 反标准化
            F_hat_w = normalizer.inverse_transform("wind_gt", F_hat_w)
            F_hat_b = normalizer.inverse_transform("bf_gt", F_hat_b)

            preds_w.append(F_hat_w.numpy()[0])
            preds_b.append(F_hat_b.numpy()[0])

    pred_w = np.array(preds_w)
    pred_b = np.array(preds_b)

    # 真值与观测
    obs_w = data[:, 1:4]
    gt_w = data[:, 17:20]
    gt_b = data[:, 20:23]
    t = data[:, 0] - data[0, 0]

    # 计算瞬变力
    q_all = torch.from_numpy(data[:, 13:17])
    R = quat_to_rotmat(q_all)
    F_hat_b_world = rotate_vector(R, torch.from_numpy(pred_b)).numpy()
    transient = obs_w - pred_w - F_hat_b_world

    # 绘制 4行3列 图
    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True)
    labels_xyz = ["X", "Y", "Z"]
    row_titles = ["MDOB观测 (世界系)", "世界系低频力", "机体系低频力", "瞬变力残差"]

    for j in range(3):
        # MDOB观测
        axes[0, j].plot(t, obs_w[:, j], "gray", alpha=0.5)
        axes[0, j].set_ylabel(f"{labels_xyz[j]} (N)")

        # 世界系力: 预测 vs 真值
        axes[1, j].plot(t, gt_w[:, j], "b-", linewidth=1, label="真值")
        axes[1, j].plot(t, pred_w[:, j], "r--", linewidth=1, label="估计")
        if j == 0:
            axes[1, j].legend(fontsize=8)

        # 机体系力: 预测 vs 真值
        axes[2, j].plot(t, gt_b[:, j], "b-", linewidth=1, label="真值")
        axes[2, j].plot(t, pred_b[:, j], "r--", linewidth=1, label="估计")
        if j == 0:
            axes[2, j].legend(fontsize=8)

        # 瞬变力
        axes[3, j].plot(t, transient[:, j], "green", alpha=0.5)
        axes[3, j].set_xlabel("时间 (s)")

    for i, title in enumerate(row_titles):
        axes[i, 0].set_title(title, fontsize=11, loc="left")

    fig.suptitle(os.path.basename(args.csv), fontsize=14)
    fig.tight_layout()

    # 保存或显示
    if args.output:
        fig.savefig(args.output, dpi=150)
        print(f"图表已保存: {args.output}")
    else:
        output = os.path.splitext(args.csv)[0] + "_lsdd.png"
        fig.savefig(output, dpi=150)
        print(f"图表已保存: {output}")

    plt.close(fig)


if __name__ == "__main__":
    main()
