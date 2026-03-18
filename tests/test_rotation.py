"""四元数旋转工具函数的正确性测试。"""

import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lsdd_net.rotation_utils import quat_to_rotmat, rotate_vector, rotate_vector_inv


def test_identity_quaternion():
    """单位四元数应产生单位旋转矩阵。"""
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # (w,x,y,z)
    R = quat_to_rotmat(q)
    assert torch.allclose(R[0], torch.eye(3), atol=1e-6)


def test_90deg_z_rotation():
    """绕Z轴旋转90度: x轴应变为y轴。"""
    angle = np.pi / 2
    q = torch.tensor([[np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)]], dtype=torch.float32)
    R = quat_to_rotmat(q)
    v = torch.tensor([[1.0, 0.0, 0.0]])
    result = rotate_vector(R, v)
    expected = torch.tensor([[0.0, 1.0, 0.0]])
    assert torch.allclose(result, expected, atol=1e-5)


def test_inverse_rotation():
    """R^T @ (R @ v) 应等于 v。"""
    # 随机四元数
    q = torch.randn(5, 4)
    q = q / q.norm(dim=-1, keepdim=True)  # 归一化
    R = quat_to_rotmat(q)
    v = torch.randn(5, 3)

    rotated = rotate_vector(R, v)
    recovered = rotate_vector_inv(R, rotated)
    assert torch.allclose(recovered, v, atol=1e-5)


def test_rotation_determinant():
    """旋转矩阵行列式应为1。"""
    q = torch.randn(10, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    R = quat_to_rotmat(q)
    dets = torch.det(R)
    assert torch.allclose(dets, torch.ones(10), atol=1e-5)


def test_batch_shape():
    """验证batch维度正确传播。"""
    q = torch.randn(2, 50, 4)
    q = q / q.norm(dim=-1, keepdim=True)
    R = quat_to_rotmat(q)
    assert R.shape == (2, 50, 3, 3)

    v = torch.randn(2, 50, 3)
    out = rotate_vector(R, v)
    assert out.shape == (2, 50, 3)


if __name__ == "__main__":
    test_identity_quaternion()
    test_90deg_z_rotation()
    test_inverse_rotation()
    test_rotation_determinant()
    test_batch_shape()
    print("全部旋转测试通过!")
