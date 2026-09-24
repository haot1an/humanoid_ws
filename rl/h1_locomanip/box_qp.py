"""批量盒约束 QP：min ½xᵀPx + gᵀx, lo ≤ x ≤ hi（P 对称正定），训练端替代 OSQP。

方法：原始积极集（projected Newton），每轮对全部问题同时：
    1) 积极变量固定在界上，解自由变量的线性方程（固定的行/列替换为单位阵，批量 n×n 求解）
    2) grad = P x + g 检查 KKT：在下界且 grad < 0、在上界且 grad > 0 的变量释放；越界的自由变量加入积极集
    3) 全部问题的积极集都不再变化时停止
与 OSQP 的一致性见 EXP-008（max 差 2.4e-4 rad/s，来自 OSQP 容差）；ADMM 版本收敛过慢已弃用。
"""
import torch


def solve_box_qp(P, g, lo, hi, max_iter=20):
    """P: (B,n,n)  g, lo, hi: (B,n)。返回 (x, 使用的轮数, 未收敛问题数)。"""
    n = P.shape[-1]
    eye = torch.eye(n, dtype=P.dtype, device=P.device).expand_as(P)
    x = torch.linalg.solve(P, -g)
    at_lo, at_hi = x < lo, x > hi
    for it in range(max_iter):
        free = ~(at_lo | at_hi)
        x_fixed = torch.where(at_lo, lo, torch.where(at_hi, hi, torch.zeros_like(g)))
        ff = free.unsqueeze(-1) & free.unsqueeze(-2)
        M = torch.where(ff, P, torch.zeros_like(P)) + torch.where(free.unsqueeze(-1), torch.zeros_like(P), eye)
        rhs = torch.where(free, -g - (P @ x_fixed.unsqueeze(-1)).squeeze(-1), x_fixed)
        x = torch.linalg.solve(M, rhs)
        grad = (P @ x.unsqueeze(-1)).squeeze(-1) + g
        new_lo = (at_lo & (grad >= 0)) | (free & (x < lo))
        new_hi = (at_hi & (grad <= 0)) | (free & (x > hi))
        changed = (new_lo != at_lo).any(-1) | (new_hi != at_hi).any(-1)
        if not changed.any():
            return torch.clamp(x, lo, hi), it + 1, 0
        at_lo, at_hi = new_lo, new_hi
    return torch.clamp(x, lo, hi), max_iter, int(changed.sum())
