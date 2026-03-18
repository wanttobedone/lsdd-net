"""纯PyTorch实现的Mamba (选择性状态空间模型)。

无需mamba-ssm依赖，支持两种运行模式：
  - 训练模式：并行处理整个序列 forward(x[B,L,D])
  - 部署模式：逐步递归更新 step(x[B,D], state)

参考论文：Gu & Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", ICLR 2024.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Dict, Tuple


class SelectiveSSM(nn.Module):
    """选择性状态空间模型 (S6)。

    核心操作：输入依赖的离散化 + 状态递归。
    训练时顺序扫描 (L=500足够快)，部署时单步递归 O(1)。
    """

    def __init__(self, d_model: int, d_state: int = 16):
        """
        Args:
            d_model: 模型维度 (每个通道独立处理)。
            d_state: SSM隐状态维度 (N)。
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A矩阵：对数参数化保证负实部 (稳定性)
        # 初始化为 -log(1, 2, ..., N)，模仿HiPPO
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).unsqueeze(0).expand(d_model, -1).clone())

        # D：残差跳跃连接
        self.D = nn.Parameter(torch.ones(d_model))

        # 输入依赖的投影：B, C, delta
        self.linear_B = nn.Linear(d_model, d_state, bias=False)
        self.linear_C = nn.Linear(d_model, d_state, bias=False)
        self.linear_delta = nn.Linear(d_model, d_model, bias=True)

        # delta的偏置初始化：使softplus输出在合理范围 (0.001~0.1)
        with torch.no_grad():
            self.linear_delta.bias.uniform_(math.log(0.001), math.log(0.1))

    def forward(self, x: Tensor, h: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """前向传播。

        Args:
            x: 输入张量。
               训练模式: (B, L, D)
               部署模式: (B, D) 单步
            h: 隐状态 (B, D, N)。训练模式可为None (自动初始化为零)。

        Returns:
            y: 输出，形状与x相同。
            h_final: 最终隐状态 (B, D, N)。
        """
        if x.dim() == 2:
            return self._step(x, h)
        return self._scan(x, h)

    def _scan(self, x: Tensor, h: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """训练模式：顺序扫描整个序列。

        Args:
            x: (B, L, D)
            h: (B, D, N) 或 None

        Returns:
            y: (B, L, D)
            h_final: (B, D, N)
        """
        B, L, D = x.shape
        N = self.d_state

        # 初始化隐状态
        if h is None:
            h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)

        # 计算输入依赖参数
        A = -torch.exp(self.A_log)                     # (D, N) 负值保证稳定性
        B_input = self.linear_B(x)                     # (B, L, N)
        C_input = self.linear_C(x)                     # (B, L, N)
        delta = F.softplus(self.linear_delta(x))       # (B, L, D) 正值

        # 顺序扫描
        ys = []
        for t in range(L):
            x_t = x[:, t, :]           # (B, D)
            B_t = B_input[:, t, :]     # (B, N)
            C_t = C_input[:, t, :]     # (B, N)
            delta_t = delta[:, t, :]   # (B, D)

            # ZOH离散化
            A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, D, N)
            B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)           # (B, D, N)

            # 状态更新: h = A_bar * h + B_bar * x
            h = A_bar * h + B_bar * x_t.unsqueeze(-1)  # (B, D, N)

            # 输出: y = (C * h).sum(-1) + D * x
            y_t = (C_t.unsqueeze(1) * h).sum(-1) + self.D * x_t  # (B, D)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, D)
        return y, h

    def _step(self, x: Tensor, h: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """部署模式：单步递归更新。

        Args:
            x: (B, D) 单个时间步输入。
            h: (B, D, N) 上一步的隐状态。

        Returns:
            y: (B, D) 输出。
            h_new: (B, D, N) 更新后的隐状态。
        """
        B, D = x.shape
        N = self.d_state

        if h is None:
            h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)

        A = -torch.exp(self.A_log)                            # (D, N)
        B_t = self.linear_B(x)                                # (B, N)
        C_t = self.linear_C(x)                                # (B, N)
        delta_t = F.softplus(self.linear_delta(x))            # (B, D)

        # ZOH离散化
        A_bar = torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, D, N)
        B_bar = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)           # (B, D, N)

        # 状态更新
        h_new = A_bar * h + B_bar * x.unsqueeze(-1)

        # 输出
        y = (C_t.unsqueeze(1) * h_new).sum(-1) + self.D * x

        return y, h_new


class MambaBlock(nn.Module):
    """完整的Mamba块：因果卷积 + SiLU门控 + 选择性SSM。

    结构: input → expand 2x → split → (conv1d → SiLU → SSM) * SiLU(gate) → project
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        """
        Args:
            d_model: 模型维度。
            d_state: SSM隐状态维度。
            d_conv: 因果卷积核宽度。
            expand: 内部扩展倍数。
        """
        super().__init__()
        self.d_model = d_model
        self.d_conv = d_conv
        d_inner = d_model * expand

        # 输入投影：扩展到2x (一半给SSM路径，一半给门控)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # 因果深度卷积 (分组卷积, groups=d_inner)
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_inner, bias=True,
        )

        # 选择性SSM
        self.ssm = SelectiveSSM(d_inner, d_state)

        # 输出投影
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(
        self, x: Tensor, state: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """前向传播。

        Args:
            x: 训练 (B,L,D) 或部署 (B,D)。
            state: 部署模式的状态字典 {"ssm_h": ..., "conv_buf": ...}。

        Returns:
            output: 与x同形状的输出。
            new_state: 更新后的状态字典。
        """
        if x.dim() == 2:
            return self._step(x, state)
        return self._forward_seq(x, state)

    def _forward_seq(
        self, x: Tensor, state: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """训练模式：处理完整序列。"""
        B, L, D = x.shape

        # 扩展投影并分割
        xz = self.in_proj(x)                   # (B, L, 2*d_inner)
        x_path, z = xz.chunk(2, dim=-1)        # 各 (B, L, d_inner)

        # 因果卷积 (转置为 B,C,L 格式)
        x_conv = x_path.transpose(1, 2)        # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[:, :, :L] # 截断因果padding
        x_conv = x_conv.transpose(1, 2)        # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # SSM
        ssm_h = state["ssm_h"] if state is not None else None
        y, ssm_h_final = self.ssm(x_conv, ssm_h)

        # 门控
        output = y * F.silu(z)

        # 输出投影
        output = self.out_proj(output)

        # 保存卷积缓冲区 (最后 d_conv-1 步，用于后续step模式)
        conv_buf = x_path[:, -(self.d_conv - 1):, :].transpose(1, 2).contiguous()

        new_state = {"ssm_h": ssm_h_final, "conv_buf": conv_buf}
        return output, new_state

    def _step(
        self, x: Tensor, state: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """部署模式：单步递归。"""
        B, D = x.shape
        d_inner = D  # 此处D实际是d_model，需要先投影

        # 注意: step模式下x是d_model维，需要先投影
        xz = self.in_proj(x)                   # (B, 2*d_inner)
        x_path, z = xz.chunk(2, dim=-1)        # 各 (B, d_inner)

        d_inner_actual = x_path.shape[-1]

        # 初始化状态
        if state is None:
            ssm_h = None
            conv_buf = torch.zeros(
                B, d_inner_actual, self.d_conv - 1,
                device=x.device, dtype=x.dtype,
            )
        else:
            ssm_h = state["ssm_h"]
            conv_buf = state["conv_buf"]

        # 因果卷积 (手动维护缓冲区)
        # 将新输入追加到缓冲区末尾
        conv_input = torch.cat([conv_buf, x_path.unsqueeze(-1)], dim=-1)  # (B, d_inner, d_conv)
        x_conv = F.conv1d(
            conv_input, self.conv1d.weight, self.conv1d.bias,
            groups=d_inner_actual,
        ).squeeze(-1)  # (B, d_inner)
        x_conv = F.silu(x_conv)

        # 更新卷积缓冲区
        new_conv_buf = conv_input[:, :, 1:]  # 滑窗: 去掉最早的一步

        # SSM单步
        y, ssm_h_new = self.ssm._step(x_conv, ssm_h)

        # 门控 + 输出投影
        output = y * F.silu(z)
        output = self.out_proj(output)

        new_state = {"ssm_h": ssm_h_new, "conv_buf": new_conv_buf}
        return output, new_state


class MambaEncoder(nn.Module):
    """Mamba编码器：多层MambaBlock堆叠，pre-norm架构。"""

    def __init__(
        self, d_model: int, d_state: int = 16,
        n_layers: int = 4, d_conv: int = 4,
    ):
        """
        Args:
            d_model: 模型维度。
            d_state: SSM隐状态维度。
            n_layers: 层数。
            d_conv: 因果卷积核宽度。
        """
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv)
            for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(d_model)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self, x: Tensor,
        states: Optional[list] = None,
    ) -> Tuple[Tensor, list]:
        """训练模式：处理完整序列。

        Args:
            x: (B, L, d_model)
            states: 各层的状态列表，None则自动初始化。

        Returns:
            output: (B, L, d_model)
            new_states: 各层的最终状态列表。
        """
        if states is None:
            states = [None] * len(self.layers)

        new_states = []
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            residual = x
            x = norm(x)
            x, state_i = layer(x, states[i])
            x = x + residual  # 残差连接
            new_states.append(state_i)

        x = self.final_norm(x)
        return x, new_states

    def step(
        self, x: Tensor, states: list,
    ) -> Tuple[Tensor, list]:
        """部署模式：单步递归。

        Args:
            x: (B, d_model)
            states: 各层的状态列表。

        Returns:
            output: (B, d_model)
            new_states: 更新后的各层状态列表。
        """
        new_states = []
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            residual = x
            x = norm(x)
            x, state_i = layer._step(x, states[i])
            x = x + residual
            new_states.append(state_i)

        x = self.final_norm(x)
        return x, new_states
