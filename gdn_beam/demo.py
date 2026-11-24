"""Generate ranked semantic IDs with a small, randomly initialized GDN."""

import argparse

import torch
from torch import nn
from torch.nn import functional as F

from .cache import SharedState
from .gdn import prefill


class ToyRecommender(nn.Module):
    def __init__(self, vocabulary: int, digits: int):
        super().__init__()
        self.vocabulary = vocabulary
        self.key_heads, self.value_heads = 2, 4
        self.key_dim, self.value_dim = 16, 16
        self.embedding = nn.Embedding((digits + 1) * vocabulary, 32)
        key_width = self.key_heads * self.key_dim
        value_width = self.value_heads * self.value_dim
        self.projections = nn.Linear(32, 2 * key_width + value_width + 2 * self.value_heads)
        self.readouts = nn.ModuleList(nn.Linear(value_width, vocabulary) for _ in range(digits))

    def inputs(self, tokens, position):
        # Position 0 holds prompt embeddings, 1..digits hold the SID codebooks.
        hidden = self.embedding(tokens + position * self.vocabulary)
        key_width = self.key_heads * self.key_dim
        sizes = (key_width, key_width, self.value_heads * self.value_dim, self.value_heads, self.value_heads)
        q, k, v, alpha, beta = self.projections(hidden).split(sizes, dim=-1)
        heads = (self.key_heads, self.key_dim)
        return (F.normalize(q.unflatten(-1, heads), dim=-1),
                F.normalize(k.unflatten(-1, heads), dim=-1),
                v.unflatten(-1, (self.value_heads, self.value_dim)), alpha.sigmoid(), beta.sigmoid())

    def log_probs(self, output, position):
        return self.readouts[position](output.flatten(-2)).log_softmax(dim=-1)


@torch.no_grad()
def beam_search(model, prompt, width, backend="auto"):
    """Return SID sequences [R,W,O] and cumulative log probabilities [R,W], best first."""
    if width < 1:
        raise ValueError("beam width must be positive")
    context, last = prefill(*model.inputs(prompt, position=0))
    scores, first = model.log_probs(last, 0).topk(min(width, model.vocabulary))
    sequences = first.unsqueeze(-1)
    digits = len(model.readouts)
    if digits == 1:
        return sequences, scores

    cache = SharedState(context, max_beams=width, max_steps=digits - 1,
                  key_heads=model.key_heads, backend=backend)
    parents = None
    for position in range(1, digits):
        output = cache.step(*model.inputs(sequences[:, :, -1], position), parents=parents)
        candidates = scores.unsqueeze(-1) + model.log_probs(output, position)
        candidates = candidates.flatten(1)
        scores, selected = candidates.topk(min(width, candidates.size(1)))
        parents = selected // model.vocabulary
        next_token = selected % model.vocabulary
        history = sequences.gather(1, parents.unsqueeze(-1).expand(-1, -1, position))
        sequences = torch.cat((history, next_token.unsqueeze(-1)), dim=-1)
    return sequences, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--backend", choices=("auto", "torch", "triton"), default="auto")
    parser.add_argument("--requests", type=int, default=2)
    parser.add_argument("--prompt-length", type=int, default=12)
    parser.add_argument("--beam-width", type=int, default=8)
    parser.add_argument("--sid-length", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=16, help="codes per SID position")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if min(args.requests, args.prompt_length, args.beam_width, args.sid_length, args.vocab_size) < 1:
        parser.error("all sizes must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but PyTorch cannot access an NVIDIA device")
    if args.backend == "triton" and args.device != "cuda":
        parser.error("use --device cuda with --backend triton")
    torch.manual_seed(args.seed)
    model = ToyRecommender(args.vocab_size, args.sid_length).eval().to(args.device)
    prompt = torch.randint(args.vocab_size, (args.requests, args.prompt_length), device=args.device)
    sequences, scores = beam_search(model, prompt, args.beam_width, args.backend)
    for request in range(args.requests):
        print(f"Request {request} (top {min(3, scores.size(1))} of {scores.size(1)} beams)")
        for sid, score in zip(sequences[request, :3].tolist(), scores[request, :3].tolist()):
            print(f"  SID {sid}  log probability {score:.5f}")


if __name__ == "__main__":
    main()
