"""FullState controls for the single-request Qwen benchmark."""

import torch
import triton
import triton.language as tl


@triton.jit
def _full_step(State, Next, Out, Q, K, V, Alpha, Beta, Parents,
               HEADS: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
               FIRST: tl.constexpr, REORDERED: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BV + tl.arange(0, BV)
    col = tl.arange(0, BK)
    beam, head = tl.program_id(1), tl.program_id(2)
    if REORDERED:
        parent = beam
    elif FIRST:
        parent = 0
    else:
        parent = tl.load(Parents + beam)
    offset = row[:, None] * DK + col[None, :]
    mask = (row[:, None] < DV) & (col[None, :] < DK)
    state = tl.load(State + (parent * HEADS + head) * DV * DK + offset, mask, 0)
    slot = beam * HEADS + head
    q = tl.load(Q + slot * DK + col, col < DK, 0).to(tl.float32)
    k = tl.load(K + slot * DK + col, col < DK, 0).to(tl.float32)
    v = tl.load(V + slot * DV + row, row < DV, 0).to(tl.float32)
    a = tl.load(Alpha + slot).to(tl.float32)
    b = tl.load(Beta + slot).to(tl.float32)
    state = state * a
    correction = b * (v - tl.sum(state * k[None, :], axis=1))
    state = state + correction[:, None] * k[None, :]
    tl.store(Next + slot * DV * DK + offset, state, mask)
    tl.store(Out + slot * DV + row, tl.sum(state * q[None, :], axis=1), row < DV)


class FullState:
    """Use identical state buffers and update math with indexed or explicitly reordered parents."""

    def __init__(self, context, width, *, explicit_reorder=False):
        self.context = context.float().contiguous()
        _, heads, dv, dk = context.shape
        self.buffers = [self.context.new_empty((width, heads, dv, dk)) for _ in range(2)]
        self.depth = 0
        self.explicit_reorder = explicit_reorder
        self.reorder_pending = explicit_reorder

    def reorder(self, parents):
        """Materialize one dense parent matrix per child, including the initial prompt fan-out."""
        if not self.reorder_pending:
            raise RuntimeError("unexpected FullState reorder")
        previous = self.context if self.depth == 0 else self.buffers[(self.depth - 1) % 2]
        if self.depth and torch.any((parents < 0) | (parents >= previous.size(0))):
            raise ValueError("invalid parent slot")
        torch.index_select(previous, 0, parents, out=self.buffers[self.depth % 2])
        self.reorder_pending = False

    def step(self, q, k, v, alpha, beta, parents):
        _, width, heads, dk = k.shape
        dv = v.size(-1)
        first = self.depth == 0
        target = self.buffers[self.depth % 2]
        if self.explicit_reorder:
            if self.reorder_pending:
                raise RuntimeError("reorder the beam states before the model forward")
            # Each block owns disjoint value rows of one beam, so the update can run in place.
            previous = target
        else:
            if not first and torch.any((parents < 0) | (parents >= width)):
                raise ValueError("invalid parent slot")
            previous = self.context if first else self.buffers[(self.depth - 1) % 2]
        out = torch.empty(v.shape, dtype=torch.float32, device=v.device)
        value_tile = 32
        _full_step[(triton.cdiv(dv, value_tile), width, heads)](
            previous, target, out, q.contiguous(), k.contiguous(),
            v.contiguous(), alpha.contiguous(), beta.contiguous(), previous if first else parents,
            heads, dk, dv, first, self.explicit_reorder, value_tile, triton.next_power_of_2(dk), num_warps=4,
        )
        self.depth += 1
        self.reorder_pending = self.explicit_reorder
        return out
