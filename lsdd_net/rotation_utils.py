"""四元数与旋转矩阵工具函数，全部支持batch操作。"""

import torch
from torch import Tensor


def quat_to_rotmat(q: Tensor) -> Tensor:
    """四元数 (w,x,y,z) 转旋转矩阵 R_wb (机体系→世界系)。

    Args:
        q: 四元数张量，形状 (..., 4)，顺序 (w, x, y, z)，匹配CSV和MDOB格式。

    Returns:
        旋转矩阵，形状 (..., 3, 3)。
    """
    w, x, y, z = q.unbind(-1)

    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),      1 - 2*(x*x + y*y),
    ], dim=-1)

    return R.reshape(q.shape[:-1] + (3, 3))


def rotate_vector(R: Tensor, v: Tensor) -> Tensor:
    """旋转向量: R @ v。

    Args:
        R: (..., 3, 3) 旋转矩阵。
        v: (..., 3) 向量。

    Returns:
        旋转后的向量 (..., 3)。
    """
    return torch.einsum("...ij,...j->...i", R, v)


def rotate_vector_inv(R: Tensor, v: Tensor) -> Tensor:
    """逆旋转向量: R^T @ v (世界系→机体系)。

    Args:
        R: (..., 3, 3) 旋转矩阵。
        v: (..., 3) 向量。

    Returns:
        逆旋转后的向量 (..., 3)。
    """
    return torch.einsum("...ji,...j->...i", R, v)
