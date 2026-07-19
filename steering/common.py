"""
common.py — shared model loading + option-logprob scoring for the 27D steering loop.

The measurement protocol (RESEARCH-PROGRAM.md §4): score the FULL OPTION TEXT by
teacher-forced log-probability, never the answer letter / first token. This avoids the
A/B/C/D position bias that silently invalidates MCQ-style LLM psychometrics (Zheng 2024).

Works on a plain HuggingFace model OR a repeng-wrapped ControlModel (same forward API),
so the identical readout is used steered and unsteered.
"""
from __future__ import annotations
import os, json, math
from dataclasses import dataclass
from typing import Sequence
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = os.environ.get("STEERING_MODEL", "meta-llama/Meta-Llama-3.1-8B")


def _patch_repeng():
    """repeng's ControlModule.forward crashes on batched input: it reshapes the shared
    [1,seq] position_ids to the full batch for a padding mask (and assumes LEFT padding).
    We right-pad and read logits only at content positions, so steering trailing pad
    tokens is harmless (causal attention; never read). Replace forward to drop that mask.
    """
    from repeng import control as _c

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        control = self.params.control
        if control is None:
            return output
        if len(control.shape) == 1:
            control = control.reshape(1, 1, -1)
        modified = output[0] if isinstance(output, tuple) else output
        assert len(control.shape) == len(modified.shape)
        control = control.to(modified.device)
        norm_pre = torch.norm(modified, dim=-1, keepdim=True)
        modified = self.params.operator(modified, control)        # mask = 1 (all positions)
        if self.params.normalize:
            norm_post = torch.norm(modified, dim=-1, keepdim=True)
            modified = modified / norm_post * norm_pre
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    _c.ControlModule.forward = forward


_patch_repeng()


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model(path: str = DEFAULT_MODEL, dtype=torch.float16, device: str | None = None):
    device = device or pick_device()
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
    model.to(device)
    model.eval()
    model.config.pad_token_id = tok.pad_token_id
    return model, tok, device


@torch.no_grad()
def score_options(model, tok, prompt: str, options: Sequence[str],
                  device: str | None = None) -> list[dict]:
    """Teacher-forced log-prob of each option as a continuation of `prompt`.

    Right-padded single batched forward (one model call scores all options). Returns,
    per option: summed continuation logprob, token count, and length-normalised logprob.
    Compare options by `logprob_sum` for a hard pick; use a softmax over `logprob_sum`
    (or `logprob_mean`) for a smooth expected-value readout.
    """
    device = device or next(model.parameters()).device
    prompt_ids = tok(prompt, add_special_tokens=True).input_ids
    n_prompt = len(prompt_ids)

    seqs, opt_spans = [], []
    for opt in options:
        opt_ids = tok(opt, add_special_tokens=False).input_ids
        if len(opt_ids) == 0:                       # guard empty option
            opt_ids = [tok.eos_token_id]
        seqs.append(prompt_ids + opt_ids)
        opt_spans.append((n_prompt, n_prompt + len(opt_ids)))

    max_len = max(len(s) for s in seqs)
    pad_id = tok.pad_token_id
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, : len(s)] = 1
    input_ids, attn = input_ids.to(device), attn.to(device)

    logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
    logprobs = torch.log_softmax(logits, dim=-1)    # [B, L, V] in fp32

    out = []
    for i, (a, b) in enumerate(opt_spans):
        # token at position t is predicted by logits at t-1
        tok_logp = logprobs[i, a - 1 : b - 1, :]
        target = input_ids[i, a:b]
        lp = tok_logp.gather(-1, target.unsqueeze(-1)).squeeze(-1).sum().item()
        n = b - a
        out.append({"option": options[i], "logprob_sum": lp,
                    "n_tokens": n, "logprob_mean": lp / n})
    return out


def softmax(xs: list[float], temp: float = 1.0) -> list[float]:
    m = max(xs)
    es = [math.exp((x - m) / temp) for x in xs]
    z = sum(es)
    return [e / z for e in es]


if __name__ == "__main__":
    # smoke test: load model, score a trivially-discriminable continuation set
    import time
    t0 = time.time()
    model, tok, device = load_model()
    print(f"loaded on {device} in {time.time()-t0:.1f}s; dtype={next(model.parameters()).dtype}")
    prompt = "The capital of France is"
    res = score_options(model, tok, prompt, [" Paris", " London", " Tokyo"])
    for r in res:
        print(f"  {r['option']!r:12} sum={r['logprob_sum']:8.3f} mean={r['logprob_mean']:7.3f}")
    probs = softmax([r["logprob_sum"] for r in res])
    print("  softmax(sum):", [f"{p:.3f}" for p in probs])
    assert res[0]["logprob_sum"] > res[1]["logprob_sum"] > res[2]["logprob_sum"], "sanity FAILED"
    print("SANITY PASS: Paris > London > Tokyo")
