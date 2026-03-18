"""LSDDNet整体模型测试: forward/step形状, 梯度, freeze。"""

import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lsdd_net.model import LSDDNet


def test_forward_shape():
    """训练模式forward输出形状正确。"""
    model = LSDDNet(d_model=32, d_state=8, n_layers=2, d_conv=4, mlp_hidden=16)

    B, L = 4, 100
    fw = torch.randn(B, L, 3)
    fb = torch.randn(B, L, 3)
    vw = torch.randn(B, L, 3)
    vb = torch.randn(B, L, 3)
    q = torch.randn(B, L, 4)
    q = q / q.norm(dim=-1, keepdim=True)

    F_hat_w, F_hat_b, R = model(fw, fb, vw, vb, q)

    assert F_hat_w.shape == (B, L, 3), f"F_hat_w形状错误: {F_hat_w.shape}"
    assert F_hat_b.shape == (B, L, 3), f"F_hat_b形状错误: {F_hat_b.shape}"
    assert R.shape == (B, L, 3, 3), f"R形状错误: {R.shape}"


def test_step_shape():
    """部署模式step输出形状正确。"""
    model = LSDDNet(d_model=32, d_state=8, n_layers=2, d_conv=4, mlp_hidden=16)
    model.eval()

    B = 1
    fw = torch.randn(B, 3)
    fb = torch.randn(B, 3)
    vw = torch.randn(B, 3)
    vb = torch.randn(B, 3)
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    with torch.no_grad():
        F_hat_w, F_hat_b, states = model.step(fw, fb, vw, vb, q, None)

    assert F_hat_w.shape == (B, 3)
    assert F_hat_b.shape == (B, 3)
    assert "world_states" in states
    assert "body_states" in states

    # 连续step
    for _ in range(5):
        F_hat_w, F_hat_b, states = model.step(fw, fb, vw, vb, q, states)
    assert F_hat_w.shape == (B, 3)


def test_gradient_flow():
    """训练模式下梯度正确流过整个网络。"""
    model = LSDDNet(d_model=32, d_state=8, n_layers=2, d_conv=4, mlp_hidden=16)

    B, L = 2, 50
    fw = torch.randn(B, L, 3, requires_grad=True)
    fb = torch.randn(B, L, 3)
    vw = torch.randn(B, L, 3)
    vb = torch.randn(B, L, 3)
    q = torch.randn(B, L, 4)
    q = q / q.norm(dim=-1, keepdim=True)

    F_hat_w, F_hat_b, R = model(fw, fb, vw, vb, q)
    loss = F_hat_w.sum() + F_hat_b.sum()
    loss.backward()

    assert fw.grad is not None
    # 检查模型参数梯度
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} 梯度为None"


def test_freeze_backbone():
    """freeze_backbone应只保留MLP头可训练。"""
    model = LSDDNet(d_model=32, d_state=8, n_layers=2, d_conv=4, mlp_hidden=16)

    total_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.freeze_backbone()
    total_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert total_after < total_before, "freeze后可训练参数应减少"

    # MLP头应仍可训练
    for name, param in model.world_head.named_parameters():
        assert param.requires_grad, f"world_head.{name} 应可训练"
    for name, param in model.body_head.named_parameters():
        assert param.requires_grad, f"body_head.{name} 应可训练"

    # 编码器应冻结
    for name, param in model.world_encoder.named_parameters():
        assert not param.requires_grad, f"world_encoder.{name} 应冻结"

    # unfreeze恢复
    model.unfreeze_all()
    total_unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total_unfrozen == total_before


def test_parameter_count():
    """参数量统计。"""
    model = LSDDNet(d_model=64, d_state=16, n_layers=4, d_conv=4, mlp_hidden=32)
    counts = model.count_parameters()
    print(f"参数量: {counts}")
    # 总量应在100K-200K范围
    assert 50_000 < counts["total"] < 500_000, f"参数量异常: {counts['total']}"


if __name__ == "__main__":
    test_forward_shape()
    print("forward形状测试通过")

    test_step_shape()
    print("step形状测试通过")

    test_gradient_flow()
    print("梯度流测试通过")

    test_freeze_backbone()
    print("freeze/unfreeze测试通过")

    test_parameter_count()
    print("参数量测试通过")

    print("\n全部模型测试通过!")
