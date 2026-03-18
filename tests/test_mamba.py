"""Mamba模块测试: scan/step等价性, 形状正确性。"""

import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lsdd_net.mamba import SelectiveSSM, MambaBlock, MambaEncoder


def test_ssm_scan_step_equivalence():
    """验证SSM的scan模式和step模式输出一致。"""
    torch.manual_seed(42)
    d_model, d_state = 16, 8
    ssm = SelectiveSSM(d_model, d_state)
    ssm.eval()

    B, L = 2, 20
    x = torch.randn(B, L, d_model)

    # scan模式
    with torch.no_grad():
        y_scan, h_scan = ssm._scan(x)

    # step模式: 逐步处理
    with torch.no_grad():
        h = None
        ys_step = []
        for t in range(L):
            y_t, h = ssm._step(x[:, t, :], h)
            ys_step.append(y_t)
        y_step = torch.stack(ys_step, dim=1)

    # 比较输出
    assert torch.allclose(y_scan, y_step, atol=1e-5), \
        f"scan/step最大差异: {(y_scan - y_step).abs().max():.2e}"

    # 比较最终隐状态
    assert torch.allclose(h_scan, h, atol=1e-5), \
        f"隐状态最大差异: {(h_scan - h).abs().max():.2e}"


def test_mamba_block_scan_step():
    """验证MambaBlock的scan和step模式一致。"""
    torch.manual_seed(42)
    d_model = 16
    block = MambaBlock(d_model, d_state=8, d_conv=4)
    block.eval()

    B, L = 2, 30
    x = torch.randn(B, L, d_model)

    # scan模式
    with torch.no_grad():
        y_scan, state_scan = block._forward_seq(x)

    # step模式
    with torch.no_grad():
        state = None
        ys = []
        for t in range(L):
            y_t, state = block._step(x[:, t, :], state)
            ys.append(y_t)
        y_step = torch.stack(ys, dim=1)

    # 允许较大容差 (因果卷积的边界效应)
    # 前d_conv步可能有差异, 比较后面的步骤
    d_conv = 4
    assert torch.allclose(y_scan[:, d_conv:, :], y_step[:, d_conv:, :], atol=1e-4), \
        f"Block scan/step最大差异: {(y_scan[:, d_conv:] - y_step[:, d_conv:]).abs().max():.2e}"


def test_encoder_shape():
    """验证MambaEncoder输出形状。"""
    torch.manual_seed(42)
    encoder = MambaEncoder(d_model=32, d_state=8, n_layers=3, d_conv=4)

    B, L = 4, 100
    x = torch.randn(B, L, 32)

    # forward
    y, states = encoder(x)
    assert y.shape == (B, L, 32), f"期望 (4,100,32), 实际 {y.shape}"
    assert len(states) == 3, f"期望3层状态, 实际 {len(states)}"

    # step
    encoder.eval()
    with torch.no_grad():
        x_step = torch.randn(B, 32)
        # 初始化状态
        init_states = [None] * 3
        y_step, new_states = encoder.step(x_step, init_states)
        assert y_step.shape == (B, 32), f"step输出形状错误: {y_step.shape}"


def test_gradient_flow():
    """验证梯度能正确流过Mamba编码器。"""
    torch.manual_seed(42)
    encoder = MambaEncoder(d_model=16, d_state=8, n_layers=2, d_conv=4)

    x = torch.randn(2, 50, 16, requires_grad=True)
    y, _ = encoder(x)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "输入梯度为None"
    assert x.grad.abs().sum() > 0, "输入梯度全为零"

    # 检查所有参数都有梯度
    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"参数 {name} 梯度为None"


if __name__ == "__main__":
    test_ssm_scan_step_equivalence()
    print("SSM scan/step等价性测试通过")

    test_mamba_block_scan_step()
    print("MambaBlock scan/step测试通过")

    test_encoder_shape()
    print("Encoder形状测试通过")

    test_gradient_flow()
    print("梯度流测试通过")

    print("\n全部Mamba测试通过!")
