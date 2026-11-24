# Shared-state beam search for Gated DeltaNet

This package implements beam decoding for Gated DeltaNet (GDN). Prompt prefill produces a single state matrix per request and value head. During beam search, every live beam contributes a small transition record at each step. Children of the same parent share those records, even when their positions in the beam change. The package includes a PyTorch implementation, a fused Triton kernel for NVIDIA GPUs, and a small autoregressive example that generates ranked semantic IDs (SIDs).

## Compare shared-state decoding and FullState on Qwen3.5-0.8B

From this checkout, use Python 3.10 or newer and a CUDA-enabled PyTorch installation on Linux. The following single-request comparison runs on a 16 GB RTX 5060 Ti. It loads the official checkpoint, runs the 24-layer text decoder on a 1,024-token purchase-history prompt and generates four ordinary vocabulary tokens. To reuse downloaded weights, pass the checkpoint directory via `--model`. [Official model](https://huggingface.co/Qwen/Qwen3.5-0.8B)

```bash
python -m pip install -e '.[qwen]'
python examples/qwen_beam.py --prompt-length 1024 --widths 128 256 --new-tokens 4 --controls reorder fullstate
```

For each beam width, read the `reorder` line for the comparison with FullState that copies the selected parents' matrices before each forward. The `fullstate` line compares with a kernel that reads the parent matrices directly by index. Each line reports a speedup equal to the control’s median latency divided by the shared-state decoder’s median latency. A value above 1 means the shared-state decoder finished the measured decode loop faster on that run.

All three paths start from the same prompt cache. The 18 GDN layers use FP32 recurrent storage and return activations in the model dtype (BF16 by default). The six full-attention layers, convolution histories, MLPs, vocabulary projection and beam selection use the same code. Timing starts after prefill, cache setup and first-token selection. It covers the three subsequent decode steps, including cache reordering and vocabulary top-k. The benchmark also reports peak allocated CUDA tensor memory, candidate overlap and score differences. [Benchmark](examples/qwen_beam.py)

The `--controls` option selects which baselines to time alongside the shared-state decoder.

| Control | How each decode step obtains its parent state |
| --- | --- |
| `fullstate` (default) | The fused kernel reads each selected parent by index and writes a full state for each child. The first step reads the shared prompt state directly. |
| `reorder` | Before the model forward, `torch.index_select` copies the selected full matrices into child slots for every GDN layer, including the first fan-out from the prompt. The same Triton GDN update then runs in place. The two FullState controls reserve identical buffers and check that their final token IDs and scores agree exactly when timed together. |
| `native` | The installed Transformers implementation reorders its own cache and runs its own GDN update. Its recurrent cache dtype and availability of optional kernels can differ from the controlled FP32 paths. |

The optional `--check-native` flag compares greedy decoding with the installed Transformers recurrence. Use `--dtype float32 --widths 1 --check-native` for a comparison with FP32 weights, activations and states.

## Run the small SID example

From the directory containing this README, use Python 3.10 or newer with PyTorch installed. The CPU example runs with synthetic prompts and a randomly initialized one-layer model.

```bash
python -m pip install -e .
python -m gdn_beam.demo --device cpu
```

For an RTX 4090 on Linux, use a CUDA-enabled PyTorch installation. The following command explicitly selects the fused kernel. Triton compiles it on the first call for each input shape and replay depth.

```bash
python -m pip install -e '.[cuda]'
python -m gdn_beam.demo --device cuda --backend triton --requests 2 --prompt-length 64 --vocab-size 128 --beam-width 128 --sid-length 4
```

The demo prints the highest-scoring SIDs and their cumulative log probabilities for each request. Its toy catalog contains all fixed-length combinations of the per-position codes. Prefill scores the first digit. Each subsequent call consumes the selected digit, scores the next one, and takes the global top beams over all parent–digit pairs within the request. An SID of length `O` takes `O - 1` decode calls. The example uses generated weights so it can run independently of a recommender checkpoint or serving framework.

## Use the decoder with projected inputs

```python
from gdn_beam import SharedState, prefill

context, last_prompt_output = prefill(prompt_q, prompt_k, prompt_v, prompt_alpha, prompt_beta)
cache = SharedState(context, key_heads=prompt_k.shape[2], max_beams=128, max_steps=3)

# The prompt readout selects the first SID candidates. Supply their projections here.
out0 = cache.step(q0, k0, v0, alpha0, beta0)
# After the next top-k, parents[r, b] is child b's parent slot in out0 for request r.
out1 = cache.step(q1, k1, v1, alpha1, beta1, parents=parents)
```

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `context` | `[R, Hv, V, K]` | Read-only post-prefill state per request and value head |
| `q, k` | `[R, W, Hk, K]` | Projected query and key for each current token |
| `v` | `[R, W, Hv, V]` | Projected value |
| `alpha, beta` | `[R, W, Hv]` | Decay and correction gates after activation |
| `parents` | `[R, W]` | Integer slots in the preceding step's active width |
| Output | `[R, W, Hv, V]` | Updated GDN state projected onto the query |

`prefill` uses the same input layout with prompt length in place of `W`. Both entry points accept FP32, FP16 and BF16 inputs. Records, context and readouts use FP32. Inputs reside on one device. Each key head serves a contiguous group of `Hv / Hk` value heads. The caller supplies the model's query/key normalization and scaling. Keep the supplied context read-only while the cache is live.

Create one cache per GDN layer and request batch. Requests in a call have the same active width and decode depth. Width can grow or shrink between calls up to `max_beams`. The first call starts all beams at the shared context. Every later call takes the parent map from beam selection. The `auto` backend selects Triton on NVIDIA CUDA when available for `K <= 256` and PyTorch otherwise. Use `backend="torch"` or `backend="triton"` to select a path explicitly.

## How the state is represented

- **GDN transition records.** From the parent projection `uk = S_parent @ k`, compute `delta = beta * (v - alpha * uk)`. The triple `(alpha, delta, k)` encodes the update `S_new = alpha * S_parent + outer(delta, k)`. The store reserves one slab per step at maximum beam width. Each step writes its active slots once. Grouped value heads share a single stored key per key head. [Record store and decode step](gdn_beam/cache.py)
- **Projection replay.** Initialize `uq = S_ctx @ q` and `uk = S_ctx @ k`. For each ancestor record `(a, d, p)` in chronological order, update `uq = a * uq + d * dot(p, q)` and `uk = a * uk + d * dot(p, k)`. The current readout is `alpha * uq + delta * dot(k, q)`. A beam retains only its compact records and integer ancestry. [PyTorch replay](gdn_beam/cache.py)
- **Ancestry index.** A child gathers its parent's integer row and appends that parent's physical slot. For example, if three children select parents `[2, 0, 2]` after the first call, their rows are `[2]`, `[0]`, and `[2]`. The first and third child both read the same step-zero record. [Beam update](gdn_beam/cache.py)
- **Fused GPU execution.** A Triton program owns one request, one value head, a tile of value rows and a group of beams. It loads the context tile once for that group, replays the indexed records using two on-chip projection accumulators, then writes the output and current transition record. Value tiles write disjoint delta slices. A designated tile writes the gate and shared key. [Fused kernel](gdn_beam/kernel.py)

## Correctness checks

```bash
python -m unittest discover -s tests -v
```

The tests compare each decode step with the explicit GDN matrix recurrence. Cases cover multiple requests, grouped heads, changing widths, repeated and reordered parents, partial GPU tiles, and 128 beams with a 128 × 128 state. A separate test checks final SID rankings and scores against a dense beam search. CUDA tests run automatically when PyTorch can access an NVIDIA GPU. With Triton installed, its interpreter can also execute the fused kernel on CPU.

```bash
TRITON_INTERPRET=1 python -m unittest discover -s tests -v
```
