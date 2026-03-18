#!/usr/bin/env python3
"""LSDD-Net Phase 2 微调脚本 (真机无GT, 冻结骨干仅调MLP头)。

用法:
    python scripts/finetune.py --config configs/train_phase2.yaml
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
from lsdd_net.losses import combined_loss_phase2
from lsdd_net.normalize import Normalizer


def main():
    parser = argparse.ArgumentParser(description="LSDD-Net Phase 2 微调")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载标准化参数 (复用Phase 1)
    normalizer = Normalizer()
    normalizer.load(cfg["data"]["norm_stats_path"])

    # 扫描真机数据
    import glob
    csv_files = sorted(glob.glob(os.path.join(cfg["data"]["csv_dir"], "*.csv")))
    if not csv_files:
        print(f"错误: 在 {cfg['data']['csv_dir']} 中未找到CSV文件")
        sys.exit(1)
    print(f"找到 {len(csv_files)} 个CSV文件")

    # 数据集 (不做train/val划分, 全部用于微调)
    dataset = LSDDDataset(
        csv_files,
        window_length=cfg["data"]["window_length"],
        stride=cfg["data"]["stride"],
        normalizer=normalizer,
    )
    loader = DataLoader(
        dataset, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=cfg["training"]["num_workers"],
        pin_memory=True, drop_last=True,
    )
    print(f"微调样本: {len(dataset)}")

    # 加载Phase 1预训练模型
    model_cfg = cfg["model"]
    model = LSDDNet(
        d_model=model_cfg["d_model"],
        d_state=model_cfg["d_state"],
        n_layers=model_cfg["n_layers"],
        d_conv=model_cfg["d_conv"],
        mlp_hidden=model_cfg["mlp_hidden"],
    ).to(device)

    pretrained_path = cfg["training"]["pretrained_checkpoint"]
    ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"已加载预训练权重: {pretrained_path}")

    # 冻结骨干
    if cfg["training"].get("freeze_backbone", True):
        model.freeze_backbone()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"冻结骨干: 可训练参数 {trainable}/{total}")

    # 优化器 (仅可训练参数)
    opt_cfg = cfg["training"]["optimizer"]
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt_cfg["lr"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["training"]["scheduler"]["T_max"],
    )

    # TensorBoard
    log_dir = cfg["training"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    ckpt_dir = cfg["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    lambda_smooth = cfg["training"]["loss_weights"]["smooth"]

    # 训练循环
    epochs = cfg["training"]["epochs"]
    best_loss = float("inf")

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        metrics = {"total": 0, "recon": 0, "smooth": 0}
        n_batches = 0

        for batch in loader:
            fw = batch["fw"].to(device)
            fb = batch["fb"].to(device)
            vw = batch["vw"].to(device)
            vb = batch["vb"].to(device)
            q = batch["q"].to(device)

            F_hat_w, F_hat_b, R = model(fw, fb, vw, vb, q)

            # Phase 2: 无GT, 仅用L_recon + L_smooth
            losses = combined_loss_phase2(
                F_hat_w, F_hat_b, fw, R,
                lambda_smooth=lambda_smooth,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for k in metrics:
                metrics[k] += losses[k].item()
            n_batches += 1

        for k in metrics:
            metrics[k] /= max(n_batches, 1)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        for k in metrics:
            writer.add_scalar(f"finetune/{k}", metrics[k], epoch)
        writer.add_scalar("finetune/lr", lr, epoch)

        dt = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss={metrics['total']:.4f} "
            f"(recon={metrics['recon']:.4f} smooth={metrics['smooth']:.4f}) | "
            f"lr={lr:.2e} | {dt:.1f}s"
        )

        # 保存
        is_best = metrics["total"] < best_loss
        if is_best:
            best_loss = metrics["total"]

        ckpt_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
        }

        if (epoch + 1) % cfg["training"]["save_every"] == 0:
            torch.save(ckpt_data, os.path.join(ckpt_dir, f"phase2_epoch_{epoch:03d}.pt"))

        if is_best:
            torch.save(ckpt_data, os.path.join(ckpt_dir, "phase2_best.pt"))
            print(f"  ★ 新最优: {best_loss:.6f}")

    print(f"\nPhase 2 微调完成。最优模型: {os.path.join(ckpt_dir, 'phase2_best.pt')}")
    writer.close()


if __name__ == "__main__":
    main()
