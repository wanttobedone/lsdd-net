#!/usr/bin/env python3
"""导出模型为TorchScript格式 (用于NUC部署)。

用法:
    python scripts/export_model.py --checkpoint checkpoints/phase1_best.pt \
        --output checkpoints/lsdd_scripted.pt
"""

import os
import sys
import argparse

import yaml
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lsdd_net.model import LSDDNet


def main():
    parser = argparse.ArgumentParser(description="导出LSDD-Net模型")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="checkpoints/lsdd_exported.pt")
    args = parser.parse_args()

    # 加载checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    # 重建模型
    model_cfg = cfg["model"]
    model = LSDDNet(
        d_model=model_cfg["d_model"],
        d_state=model_cfg["d_state"],
        n_layers=model_cfg["n_layers"],
        d_conv=model_cfg["d_conv"],
        mlp_hidden=model_cfg["mlp_hidden"],
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 保存为标准PyTorch格式 (含模型配置, 部署时直接加载)
    # 不用TorchScript (Mamba的dict state不易script化), 直接保存state_dict + config
    export = {
        "model": model.state_dict(),
        "model_config": model_cfg,
        "epoch": ckpt.get("epoch", -1),
        "best_val_loss": ckpt.get("best_val_loss", -1),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(export, args.output)
    print(f"模型已导出: {args.output}")

    # 验证: 加载并测试forward
    model2 = LSDDNet(**model_cfg)
    model2.load_state_dict(export["model"])
    model2.eval()

    # 测试step模式
    with torch.no_grad():
        fw = torch.randn(1, 3)
        fb = torch.randn(1, 3)
        vw = torch.randn(1, 3)
        vb = torch.randn(1, 3)
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        F_w, F_b, states = model2.step(fw, fb, vw, vb, q, None)
        print(f"step模式测试通过: F_w={F_w.shape}, F_b={F_b.shape}")

        # 连续10步 (模拟部署)
        for _ in range(10):
            F_w, F_b, states = model2.step(fw, fb, vw, vb, q, states)
        print(f"连续step测试通过: 10步无报错")

    # 参数量统计
    params = model.count_parameters()
    print(f"\n参数量统计:")
    for k, v in params.items():
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
