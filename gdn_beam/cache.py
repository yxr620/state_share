"""Physical transition storage and logical beam ancestry for one GDN layer."""

from dataclasses import dataclass
from importlib.util import find_spec

import torch
from torch import Tensor

from .gdn import check_inputs


@dataclass(frozen=True)
class Records:
    # Each tensor begins with [depth, request, physical beam slot].
    alpha: Tensor
    delta: Tensor
    key: Tensor


def _torch_decode(context, records, ancestry, q, k, v, alpha, beta):
    requests, width, depth = ancestry.shape
    group_size = v.size(2) // k.size(2)
    q_grouped = q.float().repeat_interleave(group_size, dim=2)
    k_grouped = k.float().repeat_interleave(group_size, dim=2)
    with torch.autocast(device_type=context.device.type, enabled=False):
        uq = torch.einsum("rhvk,rbhk->rbhv", context, q_grouped)
        uk = torch.einsum("rhvk,rbhk->rbhv", context, k_grouped)

    request = torch.arange(requests, device=context.device)[:, None]
    for j in range(depth):
        slot = ancestry[:, :, j].long()
        past_key = records.key[j, request, slot].repeat_interleave(group_size, dim=2)
        decay = records.alpha[j, request, slot].unsqueeze(-1)
        delta = records.delta[j, request, slot]
        uq = decay * uq + delta * (past_key * q_grouped).sum(-1, keepdim=True)
        uk = decay * uk + delta * (past_key * k_grouped).sum(-1, keepdim=True)

    decay = alpha.float().unsqueeze(-1)
    delta = beta.float().unsqueeze(-1) * (v.float() - decay * uk)
    output = decay * uq + delta * (k_grouped * q_grouped).sum(-1, keepdim=True)
    records.key[depth, :, :width].copy_(k)
    records.alpha[depth, :, :width].copy_(alpha)
    records.delta[depth, :, :width].copy_(delta)
    return output


class SharedState:
    """One request batch at a common decode depth. Call step once for each consumed SID token."""

    def __init__(self, context: Tensor, *, max_beams: int, max_steps: int,
                 key_heads: int | None = None, backend: str = "auto"):
        if context.ndim != 4 or min(context.shape) < 1:
            raise ValueError("context must have shape [requests, Hv, V, K] with positive dimensions")
        if context.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError("context must use FP32, FP16 or BF16")
        requests, value_heads, value_dim, key_dim = context.shape
        key_heads = value_heads if key_heads is None else key_heads
        if key_heads < 1 or value_heads % key_heads:
            raise ValueError("key_heads must be a positive divisor of Hv")
        if max_beams < 1 or max_steps < 1:
            raise ValueError("max_beams and max_steps must be positive")

        if backend not in ("auto", "torch", "triton"):
            raise ValueError("backend must be auto, torch or triton")
        cuda_supported = context.is_cuda and torch.version.cuda is not None and key_dim <= 256
        if backend == "auto":
            backend = "triton" if cuda_supported and find_spec("triton") is not None else "torch"
        if backend == "triton":
            if not cuda_supported:
                raise ValueError("the Triton backend requires an NVIDIA CUDA device and K <= 256")
            from .kernel import triton_decode
            self._decode = triton_decode
        else:
            self._decode = _torch_decode
        self.backend = backend
        self._context = context.detach().float().contiguous()
        leading = (max_steps, requests, max_beams)
        options = {"dtype": torch.float32, "device": context.device}
        self._records = Records(torch.empty((*leading, value_heads), **options),
                                torch.empty((*leading, value_heads, value_dim), **options),
                                torch.empty((*leading, key_heads, key_dim), **options))
        self._ancestry: Tensor | None = None

    @property
    def steps(self) -> int:
        return 0 if self._ancestry is None else self._ancestry.size(2) + 1

    @torch.no_grad()
    def step(self, q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor,
             parents: Tensor | None = None) -> Tensor:
        """Consume one token per live beam and return [R,W,Hv,V]. Parents index the previous call."""
        check_inputs(q, k, v, alpha, beta)
        requests, width, key_heads, key_dim = k.shape
        if (requests, v.size(2), v.size(3), key_dim) != self._context.shape:
            raise ValueError("input geometry differs from the context")
        if k.device != self._context.device or key_heads != self._records.key.size(3):
            raise ValueError("input device and key heads must match the cache")
        if self.steps >= self._records.key.size(0) or width > self._records.key.size(2):
            raise ValueError("decode exceeds max_steps or max_beams")

        if self._ancestry is None:
            if parents is not None:
                raise ValueError("the first step starts from the shared context and takes no parents")
            ancestry = torch.empty((requests, width, 0), dtype=torch.int32, device=k.device)
        else:
            if parents is None or parents.shape != (requests, width):
                raise ValueError("provide one previous-step parent slot for each new beam")
            if parents.device != k.device or parents.dtype not in (torch.int32, torch.int64):
                raise ValueError("parents must be integer indices on the input device")
            if torch.any((parents < 0) | (parents >= self._ancestry.size(1))):
                raise ValueError("parent slot outside the previous active width")
            # Copy the integer ancestry and append the parent's physical slot.
            request = torch.arange(requests, device=k.device)[:, None]
            history = self._ancestry[request, parents.long()]
            ancestry = torch.cat((history, parents.int().unsqueeze(-1)), dim=2)

        output = self._decode(self._context, self._records, ancestry, q, k, v, alpha, beta)
        self._ancestry = ancestry
        return output
