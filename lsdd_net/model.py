"""LSDD-Net: 双分支低频结构化扰动解耦网络。

架构：
  世界系分支: [R^T·F_obs_w, q, v_w](10D) → Mamba → MLP → F̂_w(3D)
  机体系分支: [F_obs_b, q, v_b](10D) → Mamba → MLP → F̂_b(3D)
  瞬变力: F̂_t = F_obs_w - F̂_w - R·F̂_b (输出空间计算，不参与学习)
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict, Tuple

from .mamba import MambaEncoder
from .rotation_utils import quat_to_rotmat, rotate_vector, rotate_vector_inv


class LSDDNet(nn.Module):
    """低频结构化扰动解耦网络。

    双独立分支架构，每个分支包含：
      - 输入投影层
      - Mamba时序编码器
      - MLP输出头
    两分支完全独立，通过坐标系预变换实现归纳偏置。
    """

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 4,
        d_conv: int = 4,
        mlp_hidden: int = 32,
    ):
        """
        Args:
            d_model: Mamba模型维度。
            d_state: SSM隐状态维度。
            n_layers: Mamba层数。
            d_conv: 因果卷积核宽度。
            mlp_hidden: 输出MLP的隐藏层维度。
        """
        super().__init__()
        self.d_model = d_model

        # 输入投影: 10D → d_model
        # 世界系分支输入: [R^T·F_obs_w(3), q(4), v_w(3)] = 10D
        # 机体系分支输入: [F_obs_b(3), q(4), v_b(3)] = 10D
        self.world_input_proj = nn.Linear(10, d_model)
        self.body_input_proj = nn.Linear(10, d_model)

        # Mamba编码器 (各自独立)
        self.world_encoder = MambaEncoder(d_model, d_state, n_layers, d_conv)
        self.body_encoder = MambaEncoder(d_model, d_state, n_layers, d_conv)

        # MLP输出头: d_model → 3D力向量
        self.world_head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 3),
        )
        self.body_head = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 3),
        )

    def forward(
        self,
        fw: Tensor, fb: Tensor,
        vw: Tensor, vb: Tensor,
        q: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """训练模式前向传播。

        Args:
            fw: (B, L, 3) 世界系MDOB残差力。
            fb: (B, L, 3) 机体系MDOB残差力。
            vw: (B, L, 3) 世界系速度。
            vb: (B, L, 3) 机体系速度。
            q:  (B, L, 4) 姿态四元数 (w,x,y,z)。

        Returns:
            F_hat_w: (B, L, 3) 世界系低频力估计。
            F_hat_b: (B, L, 3) 机体系低频力估计。
            R: (B, L, 3, 3) 旋转矩阵 (供损失函数使用)。
        """
        # 计算旋转矩阵
        R = quat_to_rotmat(q)  # (B, L, 3, 3)

        # === 世界系分支 ===
        # 预变换: 将世界系力旋转到机体系观察
        fw_rotated = rotate_vector_inv(R, fw)  # R^T @ F_obs_w → (B, L, 3)
        x_world = torch.cat([fw_rotated, q, vw], dim=-1)  # (B, L, 10)
        x_world = self.world_input_proj(x_world)           # (B, L, d_model)
        h_world, _ = self.world_encoder(x_world)           # (B, L, d_model)
        F_hat_w = self.world_head(h_world)                 # (B, L, 3)

        # === 机体系分支 ===
        x_body = torch.cat([fb, q, vb], dim=-1)            # (B, L, 10)
        x_body = self.body_input_proj(x_body)               # (B, L, d_model)
        h_body, _ = self.body_encoder(x_body)               # (B, L, d_model)
        F_hat_b = self.body_head(h_body)                    # (B, L, 3)

        return F_hat_w, F_hat_b, R

    def step(
        self,
        fw: Tensor, fb: Tensor,
        vw: Tensor, vb: Tensor,
        q: Tensor,
        states: Optional[Dict[str, list]] = None,
    ) -> Tuple[Tensor, Tensor, Dict[str, list]]:
        """部署模式：单步推理。

        Args:
            fw: (B, 3) 世界系MDOB残差力 (单步)。
            fb: (B, 3) 机体系MDOB残差力。
            vw: (B, 3) 世界系速度。
            vb: (B, 3) 机体系速度。
            q:  (B, 4) 姿态四元数。
            states: 包含 "world_states" 和 "body_states" 的字典。

        Returns:
            F_hat_w: (B, 3) 世界系低频力估计。
            F_hat_b: (B, 3) 机体系低频力估计。
            new_states: 更新后的状态字典。
        """
        if states is None:
            states = {
                "world_states": [None] * len(self.world_encoder.layers),
                "body_states": [None] * len(self.body_encoder.layers),
            }

        # 计算旋转矩阵 (单步)
        R = quat_to_rotmat(q)  # (B, 3, 3)

        # 世界系分支
        fw_rotated = rotate_vector_inv(R, fw)
        x_world = torch.cat([fw_rotated, q, vw], dim=-1)
        x_world = self.world_input_proj(x_world)
        h_world, world_states = self.world_encoder.step(x_world, states["world_states"])
        F_hat_w = self.world_head(h_world)

        # 机体系分支
        x_body = torch.cat([fb, q, vb], dim=-1)
        x_body = self.body_input_proj(x_body)
        h_body, body_states = self.body_encoder.step(x_body, states["body_states"])
        F_hat_b = self.body_head(h_body)

        new_states = {
            "world_states": world_states,
            "body_states": body_states,
        }
        return F_hat_w, F_hat_b, new_states

    def freeze_backbone(self):
        """冻结编码器和输入投影层，仅保留MLP头可训练。用于Phase 2微调。"""
        for param in self.world_input_proj.parameters():
            param.requires_grad = False
        for param in self.body_input_proj.parameters():
            param.requires_grad = False
        for param in self.world_encoder.parameters():
            param.requires_grad = False
        for param in self.body_encoder.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """解冻所有参数。"""
        for param in self.parameters():
            param.requires_grad = True

    def count_parameters(self) -> Dict[str, int]:
        """统计各部分参数量。"""
        def _count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            "world_input_proj": _count(self.world_input_proj),
            "body_input_proj": _count(self.body_input_proj),
            "world_encoder": _count(self.world_encoder),
            "body_encoder": _count(self.body_encoder),
            "world_head": _count(self.world_head),
            "body_head": _count(self.body_head),
            "total": _count(self),
        }
