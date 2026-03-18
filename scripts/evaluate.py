#!/usr/bin/env python3
"""LSDD-Net 评估脚本: 逐集step模式推理, 计算指标, 生成可视化。

用法:
    python scripts/evaluate.py --config configs/train_phase1.yaml \
        --checkpoint checkpoints/phase1_best.pt
"""

import os
import sys
import json
import argparse

import yaml
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lsdd_net.model import LSDDNet
from lsdd_net.dataset import EpisodeDataset
from lsdd_net.normalize import Normalizer


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """计算预测与真值之间的指标。

    Args:
        pred: (N, 3) 预测值。
        gt:   (N, 3) 真值。

    Returns:
        指标字典。
    """
    err = pred - gt
    mae = np.abs(err).mean(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))
    # 每个轴的相关系数
    corr = []
    for i in range(3):
        if gt[:, i].std() < 1e-8:
            corr.append(0.0)
        else:
            corr.append(np.corrcoef(pred[:, i], gt[:, i])[0, 1])
    return {
        "mae_xyz": mae.tolist(),
        "rmse_xyz": rmse.tolist(),
        "corr_xyz": corr,
        "mae_mean": float(mae.mean()),
        "rmse_mean": float(rmse.mean()),
    }


def run_episode_step(model, episode, normalizer, device):
    """对单集用step模式逐帧推理 (模拟部署)。

    Returns:
        pred_w: (N, 3) 世界系力估计 (原始尺度)。
        pred_b: (N, 3) 机体系力估计 (原始尺度)。
    """
    model.eval()
    states = None
    preds_w, preds_b = [], []

    N = episode["fw"].shape[0]
    with torch.no_grad():
        for t in range(N):
            # 提取单步, 加batch维度
            fw = episode["fw"][t:t+1].to(device)   # (1, 3)
            fb = episode["fb"][t:t+1].to(device)
            vw = episode["vw"][t:t+1].to(device)
            vb = episode["vb"][t:t+1].to(device)
            q  = episode["q"][t:t+1].to(device)

            # 标准化
            if normalizer is not None:
                fw = normalizer.transform("fw", fw)
                fb = normalizer.transform("fb", fb)
                vw = normalizer.transform("vw", vw)
                vb = normalizer.transform("vb", vb)

            F_hat_w, F_hat_b, states = model.step(fw, fb, vw, vb, q, states)

            # 反标准化 (如果训练时标准化了输出的GT, 这里需要反标准化)
            # 注意: 训练时wind_gt和bf_gt也被标准化了, 所以F_hat也在标准化空间
            if normalizer is not None:
                F_hat_w = normalizer.inverse_transform("wind_gt", F_hat_w)
                F_hat_b = normalizer.inverse_transform("bf_gt", F_hat_b)

            preds_w.append(F_hat_w.cpu().numpy())
            preds_b.append(F_hat_b.cpu().numpy())

    return np.concatenate(preds_w, axis=0), np.concatenate(preds_b, axis=0)


def plot_episode(
    timestamp, pred_w, pred_b, gt_w, gt_b, obs_w, title, save_path,
):
    """绘制单集时序对比图。"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    labels = ["X", "Y", "Z"]
    t = timestamp - timestamp[0]  # 归零

    for i in range(3):
        # 世界系力
        ax = axes[i, 0]
        ax.plot(t, obs_w[:, i], "gray", alpha=0.3, label="MDOB观测")
        ax.plot(t, gt_w[:, i], "b-", linewidth=1.5, label="真值(风力)")
        ax.plot(t, pred_w[:, i], "r--", linewidth=1.5, label="LSDD估计")
        ax.set_ylabel(f"世界系 {labels[i]} (N)")
        if i == 0:
            ax.legend(fontsize=8)
            ax.set_title("世界系低频力")

        # 机体系力
        ax = axes[i, 1]
        ax.plot(t, gt_b[:, i], "b-", linewidth=1.5, label="真值(body force)")
        ax.plot(t, pred_b[:, i], "r--", linewidth=1.5, label="LSDD估计")
        ax.set_ylabel(f"机体系 {labels[i]} (N)")
        if i == 0:
            ax.legend(fontsize=8)
            ax.set_title("机体系低频力")

    axes[2, 0].set_xlabel("时间 (s)")
    axes[2, 1].set_xlabel("时间 (s)")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="LSDD-Net 评估")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model_cfg = cfg["model"]
    model = LSDDNet(
        d_model=model_cfg["d_model"],
        d_state=model_cfg["d_state"],
        n_layers=model_cfg["n_layers"],
        d_conv=model_cfg["d_conv"],
        mlp_hidden=model_cfg["mlp_hidden"],
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"已加载模型: {args.checkpoint}")

    # 标准化器
    normalizer = Normalizer()
    normalizer.load(cfg["data"]["norm_stats_path"])

    # 加载验证集 (逐集完整序列)
    manifest_path = cfg["data"]["manifest_path"]
    with open(manifest_path) as f:
        manifest = json.load(f)

    val_files = manifest["val_files"]
    if not val_files:
        print("警告: 验证集为空, 使用训练集前3集")
        val_files = manifest["train_files"][:3]

    val_ds = EpisodeDataset(val_files)  # 不做标准化, evaluate函数内手动处理
    print(f"评估集数: {len(val_ds)}")

    os.makedirs(args.output_dir, exist_ok=True)
    all_metrics = {}

    for idx in range(len(val_ds)):
        episode = val_ds[idx]
        filename = val_ds.filenames[idx]
        print(f"\n评估: {filename}")

        # step模式推理
        pred_w, pred_b = run_episode_step(model, episode, normalizer, device)

        # 真值 (原始尺度, 不需要反标准化)
        gt_w = episode["wind_gt"].numpy()
        gt_b = episode["bf_gt"].numpy()
        obs_w = episode["fw"].numpy()
        timestamp = episode["timestamp"].numpy()

        # 计算指标
        metrics_w = compute_metrics(pred_w, gt_w)
        metrics_b = compute_metrics(pred_b, gt_b)

        print(f"  世界系力 MAE: {metrics_w['mae_xyz']}, RMSE: {metrics_w['rmse_xyz']}")
        print(f"  机体系力 MAE: {metrics_b['mae_xyz']}, RMSE: {metrics_b['rmse_xyz']}")

        all_metrics[filename] = {"world": metrics_w, "body": metrics_b}

        # 绘图
        plot_path = os.path.join(args.output_dir, f"{os.path.splitext(filename)[0]}.png")
        plot_episode(timestamp, pred_w, pred_b, gt_w, gt_b, obs_w, filename, plot_path)
        print(f"  图表已保存: {plot_path}")

    # 保存汇总指标
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n指标汇总: {metrics_path}")

    # 打印总体平均
    all_mae_w = np.mean([m["world"]["mae_mean"] for m in all_metrics.values()])
    all_mae_b = np.mean([m["body"]["mae_mean"] for m in all_metrics.values()])
    print(f"\n=== 总体平均 ===")
    print(f"世界系力 MAE: {all_mae_w:.4f} N")
    print(f"机体系力 MAE: {all_mae_b:.4f} N")


if __name__ == "__main__":
    main()
