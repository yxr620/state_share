"""Fused context projection, transition replay, readout and record append."""

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_kernel(
    Context, AlphaLog, DeltaLog, KeyLog, Ancestry, Q, K, V, Alpha, Beta, Out,
    REQUESTS: tl.constexpr, WIDTH: tl.constexpr, CAPACITY: tl.constexpr, DEPTH: tl.constexpr,
    HK: tl.constexpr, HV: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr,
    BB: tl.constexpr, BV: tl.constexpr, BK: tl.constexpr,
):
    beam = tl.program_id(0) * BB + tl.arange(0, BB)
    row = tl.program_id(1) * BV + tl.arange(0, BV)
    col = tl.arange(0, BK)
    request = (tl.program_id(2) // HV).to(tl.int64)
    vh = tl.program_id(2) % HV
    kh = vh // (HV // HK)
    active = beam < WIDTH
    key_mask = active[:, None] & (col[None, :] < DK)
    value_mask = active[:, None] & (row[None, :] < DV)

    # One [BV, DK] load per block. All BB beams use the same context tile.
    ctx = tl.load(Context + ((request * HV + vh) * DV + row[:, None]) * DK + col[None, :],
                  (row[:, None] < DV) & (col[None, :] < DK), other=0)
    live = request * WIDTH + beam
    key_offset = ((live * HK + kh) * DK)[:, None] + col[None, :]
    q = tl.load(Q + key_offset, key_mask, other=0).to(tl.float32)
    k = tl.load(K + key_offset, key_mask, other=0).to(tl.float32)
    uq = tl.sum(ctx[None, :, :] * q[:, None, :], axis=2)
    uk = tl.sum(ctx[None, :, :] * k[:, None, :], axis=2)

    for j in tl.static_range(DEPTH):
        slot = tl.load(Ancestry + live * DEPTH + j, active, other=0)
        record = (j * REQUESTS + request) * CAPACITY + slot
        past_k = tl.load(KeyLog + ((record * HK + kh) * DK)[:, None] + col[None, :],
                         key_mask, other=0)
        decay = tl.load(AlphaLog + record * HV + vh, active, other=0)
        delta = tl.load(DeltaLog + ((record * HV + vh) * DV)[:, None] + row[None, :],
                        value_mask, other=0)
        uq = decay[:, None] * uq + delta * tl.sum(past_k * q, axis=1)[:, None]
        uk = decay[:, None] * uk + delta * tl.sum(past_k * k, axis=1)[:, None]

    value_offset = ((live * HV + vh) * DV)[:, None] + row[None, :]
    v = tl.load(V + value_offset, value_mask, other=0).to(tl.float32)
    alpha = tl.load(Alpha + live * HV + vh, active, other=0).to(tl.float32)
    beta = tl.load(Beta + live * HV + vh, active, other=0).to(tl.float32)
    delta = beta[:, None] * (v - alpha[:, None] * uk)
    output = alpha[:, None] * uq + delta * tl.sum(k * q, axis=1)[:, None]
    tl.store(Out + value_offset, output, value_mask)

    record = (DEPTH * REQUESTS + request) * CAPACITY + beam
    tl.store(DeltaLog + ((record * HV + vh) * DV)[:, None] + row[None, :], delta, value_mask)
    # Disjoint writers even when several value tiles and value heads share a key.
    if tl.program_id(1) == 0:
        tl.store(AlphaLog + record * HV + vh, alpha, active)
        if vh % (HV // HK) == 0:
            tl.store(KeyLog + ((record * HK + kh) * DK)[:, None] + col[None, :], k, key_mask)


def triton_decode(context, records, ancestry, q, k, v, alpha, beta):
    requests, width, depth = ancestry.shape
    _, _, key_heads, key_dim = k.shape
    _, _, value_heads, value_dim = v.shape
    beam_tile, value_tile = 4, 8
    out = torch.empty(v.shape, device=v.device, dtype=torch.float32)
    grid = (triton.cdiv(width, beam_tile), triton.cdiv(value_dim, value_tile), requests * value_heads)
    with torch.cuda.device_of(q):
        _decode_kernel[grid](
            context, records.alpha, records.delta, records.key, ancestry,
            q.contiguous(), k.contiguous(), v.contiguous(), alpha.contiguous(), beta.contiguous(), out,
            requests, width, records.key.size(2), depth, key_heads, value_heads, key_dim, value_dim,
            beam_tile, value_tile, triton.next_power_of_2(key_dim),
        )
    return out
