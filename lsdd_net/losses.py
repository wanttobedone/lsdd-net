"""LSDD-Net损失函数。

Phase 1 (仿真有GT):  L = L_sup + λ₁·L_recon + λ₂·L_smooth
Phase 2 (真机无GT):  L = L_recon + λ₂·L_smooth

其中 L_recon 为瞬变力幅度惩罚 (非双坐标系重构，因后者代数恒为零)。
"""

import torch
from torch import Tensor

from .rotation_utils import rotate_vector


def supervised_loss(
    F_hat_w: Tensor, F_hat_b: Tensor,
    wind_gt: Tensor, bf_gt: Tensor,
) -> Tensor:
    """监督预测损失: 世界系力和机体系力分别与GT做MSE。

    目标值为无MDOB滤波滞后的真值，迫使网络隐式学习逆滤波。

    Args:
        F_hat_w: (B, L, 3) 世界系力估计。
        F_hat_b: (B, L, 3) 机体系力估计。
        wind_gt: (B, L, 3) 世界系风力真值。
        bf_gt:   (B, L, 3) 机体系body force真值。

    Returns:
        标量损失值。
    """
    loss_w = torch.nn.functional.mse_loss(F_hat_w, wind_gt)
    loss_b = torch.nn.functional.mse_loss(F_hat_b, bf_gt)
    return loss_w + loss_b


def reconstruction_loss(
    F_hat_w: Tensor, F_hat_b: Tensor,
    F_obs_w: Tensor, R: Tensor,
) -> Tensor:
    """瞬变力幅度惩罚 (修正版重构损失)。

    L_recon = (1/L) Σ ||F_obs_w - F̂_w - R·F̂_b||²

    物理含义: 鼓励两个定常分支尽可能解释观测信号，
    使瞬变残差最小。配合L_smooth，网络被迫用最平滑的方式
    解释最多的信号，自然得到低频结构化分量。

    注: 原方案的双坐标系重构损失因 F_obs_b ≡ R^T·F_obs_w
    (mdob_estimator.cpp:66) 代数上恒为零，故替换为此形式。

    Args:
        F_hat_w: (B, L, 3) 世界系力估计。
        F_hat_b: (B, L, 3) 机体系力估计。
        F_obs_w: (B, L, 3) 世界系MDOB观测力。
        R:       (B, L, 3, 3) 旋转矩阵 (机体→世界)。

    Returns:
        标量损失值。
    """
    # 瞬变力 = 观测 - 世界系定常 - 旋转后的机体系定常
    F_transient = F_obs_w - F_hat_w - rotate_vector(R, F_hat_b)
    return (F_transient ** 2).mean()


def smoothness_loss(F_hat: Tensor) -> Tensor:
    """时间平滑损失: 惩罚相邻时间步的力变化。

    仅对世界系分支施加，机体系力可能存在阶跃突变 (如喷枪开关)。

    L_smooth = (1/(L-1)) Σ ||F̂(t) - F̂(t-1)||²

    Args:
        F_hat: (B, L, 3) 力估计序列。

    Returns:
        标量损失值。
    """
    diff = F_hat[:, 1:, :] - F_hat[:, :-1, :]
    return (diff ** 2).mean()


def combined_loss_phase1(
    F_hat_w: Tensor, F_hat_b: Tensor,
    wind_gt: Tensor, bf_gt: Tensor,
    F_obs_w: Tensor, R: Tensor,
    lambda_recon: float = 0.1,
    lambda_smooth: float = 0.01,
) -> dict:
    """Phase 1 总损失: L_sup + λ₁·L_recon + λ₂·L_smooth。

    Args:
        F_hat_w: (B, L, 3) 世界系力估计。
        F_hat_b: (B, L, 3) 机体系力估计。
        wind_gt: (B, L, 3) 风力真值。
        bf_gt:   (B, L, 3) 机体力真值。
        F_obs_w: (B, L, 3) 世界系MDOB观测力。
        R:       (B, L, 3, 3) 旋转矩阵。
        lambda_recon: 重构损失权重。
        lambda_smooth: 平滑损失权重。

    Returns:
        字典: {"total": 总损失, "sup": ..., "recon": ..., "smooth": ...}
    """
    l_sup = supervised_loss(F_hat_w, F_hat_b, wind_gt, bf_gt)
    l_recon = reconstruction_loss(F_hat_w, F_hat_b, F_obs_w, R)
    l_smooth = smoothness_loss(F_hat_w)

    total = l_sup + lambda_recon * l_recon + lambda_smooth * l_smooth

    return {
        "total": total,
        "sup": l_sup.detach(),
        "recon": l_recon.detach(),
        "smooth": l_smooth.detach(),
    }


def combined_loss_phase2(
    F_hat_w: Tensor, F_hat_b: Tensor,
    F_obs_w: Tensor, R: Tensor,
    lambda_smooth: float = 0.01,
) -> dict:
    """Phase 2 总损失: L_recon + λ₂·L_smooth (无GT监督)。

    Args:
        F_hat_w: (B, L, 3) 世界系力估计。
        F_hat_b: (B, L, 3) 机体系力估计。
        F_obs_w: (B, L, 3) 世界系MDOB观测力。
        R:       (B, L, 3, 3) 旋转矩阵。
        lambda_smooth: 平滑损失权重。

    Returns:
        字典: {"total": 总损失, "recon": ..., "smooth": ...}
    """
    l_recon = reconstruction_loss(F_hat_w, F_hat_b, F_obs_w, R)
    l_smooth = smoothness_loss(F_hat_w)

    total = l_recon + lambda_smooth * l_smooth

    return {
        "total": total,
        "recon": l_recon.detach(),
        "smooth": l_smooth.detach(),
    }
