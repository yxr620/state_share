import os
import unittest

import torch
from torch.nn import functional as F

from gdn_beam import SharedState, prefill
from gdn_beam.demo import ToyRecommender, beam_search
from gdn_beam.gdn import dense_step


INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
NATIVE_CUDA = torch.version.cuda is not None and torch.cuda.is_available() and not INTERPRET


def inputs(requests, width, hk, hv, dk, dv, device, dtype=torch.float32):
    def random(*shape):
        return torch.randn(*shape, device=device)

    q = F.normalize(random(requests, width, hk, dk), dim=-1)
    k = F.normalize(random(requests, width, hk, dk), dim=-1)
    # Include non-contiguous value and gate inputs from a larger projection.
    v = random(requests, width, hv, dv * 2).to(dtype)[..., ::2]
    alpha, beta = random(requests, width, hv, 2).sigmoid().to(dtype).unbind(-1)
    return q.to(dtype), k.to(dtype), v, alpha, beta


class TestDecode(unittest.TestCase):
    def check_tree(self, device, backend, dtype=torch.float32, interpret=False):
        torch.manual_seed(41)
        r, hk, hv, dk, dv = 2, 2, 4, 13, 11
        context, _ = prefill(*inputs(r, 7, hk, hv, dk, dv, device, dtype))
        original = context.clone()
        cache = SharedState(context, key_heads=hk, max_steps=4, max_beams=7, backend=backend)
        if interpret:
            from gdn_beam.kernel import triton_decode
            cache._decode = triton_decode
        # Each request has its own parent map. Exercise expansion, contraction, and repeated slots.
        lineages = [None, [[2, 0, 2, 1, 2], [1, 1, 0, 2, 1]],
                    [[4, 0], [1, 3]], [[1, 1, 0, 1], [0, 1, 1, 1]]]
        request = torch.arange(r, device=device)[:, None]
        state = context.unsqueeze(1).expand(-1, 3, -1, -1, -1)
        snapshots = []
        logs = (cache._records.alpha, cache._records.delta, cache._records.key)
        for t, lineage in enumerate(lineages):
            parents = None if lineage is None else torch.tensor(lineage, device=device)
            if parents is not None:
                state = state[request, parents]
            width = state.size(1)
            data = inputs(r, width, hk, hv, dk, dv, device, dtype)
            state, expected = dense_step(state, *data)
            actual = cache.step(*data, parents=parents)
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            snapshots.append([log[t, :, :width].clone() for log in logs])
            self.assertEqual(cache.steps, t + 1)
        for t, snapshot in enumerate(snapshots):
            for log, previous in zip(logs, snapshot):
                self.assertTrue(torch.equal(log[t, :, :previous.size(1)], previous))
        self.assertTrue(torch.equal(context, original))

    def test_cpu_projection_replay(self):
        self.check_tree("cpu", "torch")

    def test_mixed_precision_inputs(self):
        self.check_tree("cpu", "torch", torch.bfloat16)

    @unittest.skipUnless(NATIVE_CUDA, "requires NVIDIA GPU in native mode")
    def test_cuda_projection_replay(self):
        for dtype in (torch.float32, torch.float16):
            with self.subTest(dtype=dtype):
                self.check_tree("cuda", "triton", dtype)

    @unittest.skipUnless(INTERPRET, "requires Triton interpreter mode")
    def test_fused_kernel_in_interpreter(self):
        self.check_tree("cpu", "torch", interpret=True)

    def check_large_beam(self, device, backend):
        torch.manual_seed(19)
        width, hk, hv, dk, dv = 128, 1, 2, 128, 128
        context = torch.randn(1, hv, dv, dk, device=device)
        cache = SharedState(context, key_heads=hk, max_steps=3, max_beams=width, backend=backend)
        state = context.unsqueeze(1).expand(-1, width, -1, -1, -1)
        for t in range(3):
            parents = None
            if t:
                parents = torch.randint(width, (1, width), device=device)
                state = state[:, parents[0]]
            data = inputs(1, width, hk, hv, dk, dv, device)
            state, expected = dense_step(state, *data)
            torch.testing.assert_close(cache.step(*data, parents=parents), expected, atol=2e-5, rtol=2e-5)

    def test_large_beam_cpu(self):
        self.check_large_beam("cpu", "torch")

    @unittest.skipUnless(NATIVE_CUDA, "requires NVIDIA GPU in native mode")
    def test_large_beam_cuda(self):
        self.check_large_beam("cuda", "triton")

    def test_invalid_parent_or_capacity_preserves_history(self):
        data = inputs(1, 2, 1, 1, 4, 3, "cpu")
        context = torch.zeros(1, 1, 3, 4)
        cache = SharedState(context, max_steps=2, max_beams=2)
        cache.step(*data)
        for parents in (None, torch.tensor([[-1, 0]]), torch.tensor([[2, 1]])):
            with self.assertRaises(ValueError):
                cache.step(*data, parents=parents)
        self.assertEqual(cache.steps, 1)
        with self.assertRaises(ValueError):
            cache.step(*inputs(1, 3, 1, 1, 4, 3, "cpu"), parents=torch.zeros(1, 3, dtype=torch.long))
        cache.step(*data, parents=torch.tensor([[1, 1]], dtype=torch.int32))
        with self.assertRaises(ValueError):
            cache.step(*data, parents=torch.tensor([[0, 1]]))
        self.assertEqual(cache.steps, 2)

    def test_scalar_prefill(self):
        q = torch.tensor([2., 3.]).reshape(1, 2, 1, 1)
        k = torch.ones_like(q)
        v = torch.tensor([4., 10.]).reshape_as(q)
        alpha, beta = torch.ones(1, 2, 1) * 0.5, torch.ones(1, 2, 1) * 0.25
        state, last = prefill(q, k, v, alpha, beta)
        # S0 = 1; S1 = 0.5 + 0.25*(10 - 0.5) = 2.875.
        self.assertEqual(state.item(), 2.875)
        self.assertEqual(last.item(), 8.625)

    @torch.no_grad()
    def test_ranked_sids_against_dense_search(self):
        torch.manual_seed(7)
        model = ToyRecommender(vocabulary=3, digits=3).eval()
        prompt = torch.tensor([[0, 2, 1]])
        context, output = prefill(*model.inputs(prompt, 0))
        for width in (2, 27):
            beams = [((), 0., context, output)]
            for position in range(3):
                candidates = []
                for prefix, score, state, last in beams:
                    for token, log_p in enumerate(model.log_probs(last, position)[0].tolist()):
                        candidates.append((prefix + (token,), score + log_p, state))
                candidates.sort(key=lambda candidate: candidate[1], reverse=True)
                beams = []
                for prefix, score, parent in candidates[:width]:
                    data = model.inputs(torch.tensor([prefix[-1]]), position + 1)
                    state, last = dense_step(parent, *data)
                    beams.append((prefix, score, state, last))
            sids, scores = beam_search(model, prompt, width)
            self.assertEqual(sids[0].tolist(), [list(b[0]) for b in beams])
            torch.testing.assert_close(scores[0], torch.tensor([b[1] for b in beams]))
