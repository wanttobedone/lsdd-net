#!/usr/bin/env python3
"""LSDD-Net Phase 1 训练脚本。

用法:
    python scripts/train.py --config configs/train_phase1.yaml
"""

import os
import sys
import json
import argparse
import time

import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lsdd_net.model import LSDDNet
from lsdd_net.dataset import LSDDDataset
from lsdd_net.losses import combined_loss_phase1
from lsdd_net.normalize import Normalizer


def main():
    parser = argparse.ArgumentParser(description="LSDD-Net 训练")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="恢复训练的checkpoint路径")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # 设置随机种子
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载数据清单
    manifest_path = cfg["data"]["manifest_path"]
    with open(manifest_path) as f:
        manifest = json.load(f)

    # 标准化器
    normalizer = Normalizer()
    normalizer.load(manifest["norm_stats_path"])

    # 数据集
    train_ds = LSDDDataset(
        manifest["train_files"],
        window_length=cfg["data"]["window_length"],
        stride=cfg["data"]["stride"],
        normalizer=normalizer,
    )
    val_ds = LSDDDataset(
        manifest["val_files"],
        window_length=cfg["data"]["window_length"],
        stride=cfg["data"]["stride"],
        normalizer=normalizer,
    )

    print(f"训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=cfg["training"]["num_workers"],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=False, num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
    )

    # 模型
    model_cfg = cfg["model"]
    model = LSDDNet(
        d_model=model_cfg["d_model"],
        d_state=model_cfg["d_state"],
        n_layers=model_cfg["n_layers"],
        d_conv=model_cfg["d_conv"],
        mlp_hidden=model_cfg["mlp_hidden"],
    ).to(device)

    print(f"模型参数量: {model.count_parameters()}")

    # 优化器
    opt_cfg = cfg["training"]["optimizer"]
    if opt_cfg["type"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=opt_cfg["lr"],
            weight_decay=opt_cfg["weight_decay"],
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=opt_cfg["lr"],
        )

    # 学习率调度器
    sched_cfg = cfg["training"]["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=sched_cfg["T_max"],
    )

    # 恢复训练
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"从 epoch {start_epoch} 恢复训练")

    # TensorBoard
    log_dir = cfg["training"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Checkpoint目录
    ckpt_dir = cfg["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    # 损失权重
    loss_cfg = cfg["training"]["loss_weights"]
    lambda_recon = loss_cfg["recon"]
    lambda_smooth = loss_cfg["smooth"]

    # 训练循环
    epochs = cfg["training"]["epochs"]
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # === 训练 ===
        model.train()
        train_metrics = {"total": 0, "sup": 0, "recon": 0, "smooth": 0}
        n_batches = 0

        for batch in train_loader:
            # 移到设备
            fw = batch["fw"].to(device)
            fb = batch["fb"].to(device)
            vw = batch["vw"].to(device)
            vb = batch["vb"].to(device)
            q = batch["q"].to(device)
            wind_gt = batch["wind_gt"].to(device)
            bf_gt = batch["bf_gt"].to(device)

            # 前向
            F_hat_w, F_hat_b, R = model(fw, fb, vw, vb, q)

            # 计算损失 (注意: L_recon需要未标准化的F_obs_w, 但训练时fw已标准化)
            # 这里fw已标准化, F_hat_w也在标准化空间, wind_gt也标准化了
            # 所以各损失在标准化空间中计算, 是一致的
            losses = combined_loss_phase1(
                F_hat_w, F_hat_b, wind_gt, bf_gt,
                fw, R,  # fw已标准化, 但各项在同一空间下计算
                lambda_recon=lambda_recon,
                lambda_smooth=lambda_smooth,
            )

            # 反向传播
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in train_metrics:
                train_metrics[k] += losses[k].item()
            n_batches += 1

        for k in train_metrics:
            train_metrics[k] /= max(n_batches, 1)

        # === 验证 ===
        model.eval()
        val_metrics = {"total": 0, "sup": 0, "recon": 0, "smooth": 0}
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                fw = batch["fw"].to(device)
                fb = batch["fb"].to(device)
                vw = batch["vw"].to(device)
                vb = batch["vb"].to(device)
                q = batch["q"].to(device)
                wind_gt = batch["wind_gt"].to(device)
                bf_gt = batch["bf_gt"].to(device)

                F_hat_w, F_hat_b, R = model(fw, fb, vw, vb, q)
                losses = combined_loss_phase1(
                    F_hat_w, F_hat_b, wind_gt, bf_gt,
                    fw, R,
                    lambda_recon=lambda_recon,
                    lambda_smooth=lambda_smooth,
                )

                for k in val_metrics:
                    val_metrics[k] += losses[k].item()
                n_val += 1

        for k in val_metrics:
            val_metrics[k] /= max(n_val, 1)

        # 学习率更新
        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        # 记录到TensorBoard
        for k in train_metrics:
            writer.add_scalar(f"train/{k}", train_metrics[k], epoch)
        for k in val_metrics:
            writer.add_scalar(f"val/{k}", val_metrics[k], epoch)
        writer.add_scalar("lr", lr, epoch)

        # 打印进度
        dt = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_metrics['total']:.4f} "
            f"(sup={train_metrics['sup']:.4f} recon={train_metrics['recon']:.4f} "
            f"smooth={train_metrics['smooth']:.4f}) | "
            f"val_loss={val_metrics['total']:.4f} | "
            f"lr={lr:.2e} | {dt:.1f}s"
        )

        # 保存checkpoint
        is_best = val_metrics["total"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["total"]

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": cfg,
        }

        if (epoch + 1) % cfg["training"]["save_every"] == 0:
            path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
            torch.save(ckpt, path)

        if is_best:
            path = os.path.join(ckpt_dir, "phase1_best.pt")
            torch.save(ckpt, path)
            print(f"  ★ 新最优验证损失: {best_val_loss:.6f}")

    # 保存最终模型
    final_path = os.path.join(ckpt_dir, "phase1_final.pt")
    torch.save(ckpt, final_path)
    print(f"\n训练完成。最终模型: {final_path}")
    print(f"最优模型: {os.path.join(ckpt_dir, 'phase1_best.pt')} (val_loss={best_val_loss:.6f})")

    writer.close()


if __name__ == "__main__":
    main()
