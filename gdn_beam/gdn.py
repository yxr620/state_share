"""Dense GDN recurrence for prompt prefill and numerical verification."""

import torch
from torch import Tensor


def check_inputs(q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor) -> None:
    if q.ndim != 4 or k.shape != q.shape:
        raise ValueError("q and k must both have shape [requests, tokens or beams, Hk, K]")
    if v.ndim != 4 or v.shape[:2] != q.shape[:2]:
        raise ValueError("v must have shape [requests, tokens or beams, Hv, V]")
    if min(*q.shape, *v.shape) < 1 or v.size(2) % q.size(2):
        raise ValueError("all dimensions must be positive and Hv must be divisible by Hk")
    if alpha.shape != v.shape[:-1] or beta.shape != alpha.shape:
        raise ValueError("alpha and beta must have shape [requests, tokens or beams, Hv]")
    for tensor in (q, k, v, alpha, beta):
        if tensor.device != q.device or tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("inputs must share a device and use FP32, FP16 or BF16")


def dense_step(state: Tensor, q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor):
    """State has shape [..., Hv, V, K]. Return the new state and its readout."""
    group_size = v.size(-2) // k.size(-2)
    q = q.float().repeat_interleave(group_size, dim=-2)
    k = k.float().repeat_interleave(group_size, dim=-2)
    decayed = state * alpha.float()[..., None, None]
    prediction = (decayed * k.unsqueeze(-2)).sum(dim=-1)
    correction = beta.float().unsqueeze(-1) * (v.float() - prediction)
    updated = decayed + correction.unsqueeze(-1) * k.unsqueeze(-2)
    return updated, (updated * q.unsqueeze(-2)).sum(dim=-1)


@torch.no_grad()
def prefill(q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor) -> tuple[Tensor, Tensor]:
    """Scan a nonempty prompt from zero. Return S_ctx [R,Hv,V,K] and the last output [R,Hv,V]."""
    check_inputs(q, k, v, alpha, beta)
    state = torch.zeros((q.size(0), v.size(2), v.size(3), k.size(3)), device=q.device, dtype=torch.float32)
    for t in range(q.size(1)):
        state, output = dense_step(state, q[:, t], k[:, t], v[:, t], alpha[:, t], beta[:, t])
    return state, output
