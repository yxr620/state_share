"""Time fixed-length text beam search on the text decoder of Qwen3.5-0.8B."""

import argparse
from contextlib import ExitStack
from copy import deepcopy
from statistics import median
from time import perf_counter
from unittest.mock import patch

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm

from full_state import FullState
from gdn_beam import SharedState


class DecodeRule:
    def __init__(self, state):
        self.state = state
        self.parents = None

    def __call__(self, query, key, value, *, g, beta, initial_state, output_final_state,
                 use_qk_l2norm_in_kernel):
        if query.size(1) != 1 or initial_state is not None or not use_qk_l2norm_in_kernel:
            raise ValueError("expected a single cached token with Qwen query/key normalization")
        # Transformers uses [W, 1, H, D]. SharedState uses [R=1, W, H, D].
        q = l2norm(query, dim=-1).float().transpose(0, 1) * (query.size(-1) ** -0.5)
        k = l2norm(key, dim=-1).transpose(0, 1)
        out = self.state.step(q, k, value.transpose(0, 1), g.exp().transpose(0, 1),
                              beta.transpose(0, 1), parents=self.parents)
        return out.transpose(0, 1).to(value.dtype), None


def external_state_update(state):
    if state is not None:
        raise ValueError("the external decode rule owns the recurrent state")


def install_rules(text, cache, mode, width, steps, stack):
    rules = []
    for block, layer in zip(text.layers, cache.layers):
        if block.layer_type != "linear_attention":
            continue
        # The prompt cache holds [R,H,K,V]. Each backend begins from the same FP32 [R,H,V,K].
        context = layer.recurrent_states.transpose(-1, -2).float().contiguous()
        if mode == "shared":
            state = SharedState(context, key_heads=context.size(1), max_beams=width, max_steps=steps, backend="triton")
        else:
            state = FullState(context, width, explicit_reorder=(mode == "reorder"))
        rule = DecodeRule(state)
        rules.append(rule)
        stack.enter_context(patch.object(block.linear_attn, "recurrent_gated_delta_rule", rule))
        # HF still maintains/reorders its convolution state. Full attention keeps the usual KV cache.
        layer.is_recurrent_states_initialized = False
        layer.recurrent_states = None
        layer.update_recurrent_state = external_state_update
    return rules


def prompt_ids(tokenizer, length):
    history = tokenizer.encode(" Recent purchases include coffee beans, paper filters, and an electric grinder.",
                               add_special_tokens=False)
    suffix = tokenizer.encode(" Recommend one related product. The next item is", add_special_tokens=False)
    if length < len(suffix):
        raise ValueError(f"prompt length must be at least {len(suffix)}")
    prefix = (history * (length // len(history) + 1))[:length - len(suffix)]
    return torch.tensor([prefix + suffix], device="cuda")


def decode(text, head, prompt_cache, first_logp, width, steps, mode):
    cache = deepcopy(prompt_cache)
    with ExitStack() as stack:
        rules = [] if mode == "native" else install_rules(text, cache, mode, width, steps, stack)
        scores, tokens = first_logp.topk(width)
        sequences = tokens[:, None]
        parent = torch.zeros_like(tokens)
        torch.cuda.synchronize()
        start = perf_counter()
        for t in range(steps):
            cache.reorder_cache(parent)
            for rule in rules:
                rule.parents = parent[None, :] if t else None
                if mode == "reorder":
                    rule.state.reorder(parent)
            result = text(input_ids=tokens[:, None], past_key_values=cache, use_cache=True)
            logits = head(result.last_hidden_state[:, -1])
            joint = torch.log_softmax(logits, dim=-1, dtype=torch.float32) + scores[:, None]
            scores, chosen = joint.flatten().topk(width)
            parent = chosen // logits.size(-1)
            tokens = chosen % logits.size(-1)
            sequences = torch.cat([sequences[parent], tokens[:, None]], dim=-1)
            del result, logits, joint
        torch.cuda.synchronize()
        elapsed_ms = (perf_counter() - start) * 1000
    return elapsed_ms, sequences.cpu(), scores.cpu()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B", help="model ID or local official checkpoint")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--widths", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--new-tokens", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--controls", nargs="+", choices=["fullstate", "reorder", "native"], default=["fullstate"],
                        help="compare SharedState to indexed FullState, explicitly reordered FullState, or Transformers")
    parser.add_argument("--check-native", action="store_true", help="also compare the top beam at width 1 with Transformers")
    args = parser.parse_args()
    if min(args.widths) < 1 or args.repeats < 1 or args.warmup < 0 or args.new_tokens < 2:
        parser.error("use positive widths and repeats, nonnegative warmup and at least two output tokens")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model, info = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), attn_implementation="sdpa", output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"]:
        raise RuntimeError(f"incomplete checkpoint: {info}")
    text, head = model.model.language_model.eval().cuda(), model.lm_head.eval().cuda()
    del model
    cfg = text.config
    if cfg.linear_num_key_heads != cfg.linear_num_value_heads or max(args.widths) > cfg.vocab_size:
        parser.error("this example requires equal key/value head counts and beam width at most the vocabulary size")
    prompt = prompt_ids(tokenizer, args.prompt_length)
    prefill = text(input_ids=prompt, use_cache=True)
    first_logp = head(prefill.last_hidden_state[:, -1])[0].float().log_softmax(-1)
    prompt_cache = prefill.past_key_values
    del prefill
    print(f"{torch.cuda.get_device_name()}, {args.dtype}, {len(text.layers)} layers, {args.prompt_length} prompt tokens, "
          f"{args.new_tokens - 1} post-prefill steps", flush=True)

    def run(width, mode):
        return decode(text, head, prompt_cache, first_logp, width, args.new_tokens - 1, mode)

    if args.check_native:
        _, ref_ids, ref_scores = run(1, "native")
        for mode in ("fullstate", "shared"):
            _, ids, scores = run(1, mode)
            print(f"greedy vs Transformers: {mode} same tokens={torch.equal(ids, ref_ids)}, "
                  f"native={tokenizer.decode(ref_ids[0])!r}, {mode}={tokenizer.decode(ids[0])!r}, "
                  f"top beam score difference={float((ref_scores - scores).abs().max()):.6g}", flush=True)

    for width in args.widths:
        modes = list(dict.fromkeys([*args.controls, "shared"]))
        times = {mode: [] for mode in modes}
        peaks = {mode: [] for mode in modes}
        outputs = {}
        for i in range(args.warmup + args.repeats):
            offset = i % len(modes)
            for mode in modes[offset:] + modes[:offset]:
                torch.cuda.reset_peak_memory_stats()
                ms, ids, scores = run(width, mode)
                if i >= args.warmup:
                    times[mode].append(ms)
                    peaks[mode].append(torch.cuda.max_memory_allocated() / 2**30)
                    outputs[mode] = (ids, scores)
        if "fullstate" in outputs and "reorder" in outputs:
            for indexed, reordered in zip(outputs["fullstate"], outputs["reorder"]):
                torch.testing.assert_close(indexed, reordered, atol=0, rtol=0)
        shared_ids, scores = outputs["shared"]
        shared_scores = dict(zip(map(tuple, shared_ids.tolist()), scores.tolist()))
        shared_ms = median(times["shared"])
        print(f"W={width}  SharedState={shared_ms:.2f} ms  peak={max(peaks['shared']):.2f} GiB", flush=True)
        for mode in modes[:-1]:
            ids, scores = outputs[mode]
            control_scores = dict(zip(map(tuple, ids.tolist()), scores.tolist()))
            common = shared_scores.keys() & control_scores.keys()
            diff = max((abs(shared_scores[s] - control_scores[s]) for s in common), default=float("nan"))
            ms = median(times[mode])
            print(f"  {mode}={ms:.2f} ms  peak={max(peaks[mode]):.2f} GiB  SharedState speedup={ms/shared_ms:.3f}x  "
                  f"same top1={torch.equal(shared_ids[0], ids[0])}  shared={len(common)}/{width}  "
                  f"max shared score difference={diff:.6g}", flush=True)


if __name__ == "__main__":
    main()
