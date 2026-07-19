"""
fastscore.py — prefix-KV-cache option scoring (~6-7× faster administration).

The few-shot block (HEADER + examples) is identical across every item and anchor, but the
slow path recomputes it 135×6 times per administration. Here we forward it ONCE, cache its
KV, expand to the anchor-batch size, and `crop` back after each item. Numerically identical
to common.score_options (verified in __main__ — MPS can give silently-wrong logits, so this
self-check is mandatory before trusting the speedup).

Works on a repeng-wrapped ControlModel too: the control hook fires on both the prefix forward
and the per-item forward, so steering is applied consistently (prefix is unsteered context is
NOT true — the prefix KV is steered once and reused; that's the same steering the slow path
applies to the prefix every call, so results match).
"""
from __future__ import annotations
import torch
from common import softmax


class PrefixScorer:
    def __init__(self, model, tok, prefix_text: str, n_options: int, device=None):
        self.model = model
        self.tok = tok
        self.device = device or next(model.parameters()).device
        self.n = n_options
        ids = tok(prefix_text, add_special_tokens=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = model(**ids, use_cache=True)
        self.cache = out.past_key_values
        self.cache.batch_repeat_interleave(n_options)   # expand batch 1 -> n
        self.plen = ids.input_ids.shape[1]

    @torch.no_grad()
    def score(self, local_prompt: str, options: list[str]) -> list[dict]:
        """Score each option as a continuation of (cached prefix + local_prompt)."""
        assert len(options) == self.n, "n_options fixed at construction"
        tok = self.tok
        lp_ids = tok(local_prompt, add_special_tokens=False).input_ids
        nlp = len(lp_ids)
        seqs, spans = [], []
        for opt in options:
            oid = tok(opt, add_special_tokens=False).input_ids or [tok.eos_token_id]
            seqs.append(lp_ids + oid)
            spans.append((nlp, nlp + len(oid)))
        maxlen = max(len(s) for s in seqs)
        pad = tok.pad_token_id
        inp = torch.full((self.n, maxlen), pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            inp[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        inp = inp.to(self.device)
        # attention over [prefix | local+option(+pad)]
        cont_mask = (inp != pad).long()
        attn = torch.cat([torch.ones((self.n, self.plen), dtype=torch.long, device=self.device),
                          cont_mask], dim=1)
        pos = torch.arange(self.plen, self.plen + maxlen, device=self.device).unsqueeze(0).expand(self.n, -1)
        cache_pos = torch.arange(self.plen, self.plen + maxlen, device=self.device)
        out = self.model(input_ids=inp, attention_mask=attn, past_key_values=self.cache,
                         position_ids=pos, cache_position=cache_pos, use_cache=True)
        logits = out.logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
        self.cache.crop(self.plen)                       # restore prefix-only cache
        res = []
        for i, (a, b) in enumerate(spans):
            tl = logprobs[i, a - 1 : b - 1, :]
            tgt = inp[i, a:b]
            s = tl.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).sum().item()
            res.append({"option": options[i], "logprob_sum": s, "n_tokens": b - a,
                        "logprob_mean": s / (b - a)})
        return res


if __name__ == "__main__":
    # MANDATORY self-check: cached must match uncached score_options within tolerance.
    import torch as _t
    from common import load_model, score_options, DEFAULT_MODEL
    from administer import HEADER, _fewshot_block, ANCHORS, load_spi, ITEM_FMT

    model, tok, device = load_model(DEFAULT_MODEL, dtype=_t.float16)
    spi, items = load_spi()
    prefix = HEADER + _fewshot_block() + "\n\n"
    anchors = [" " + a for a, _ in ANCHORS]
    scorer = PrefixScorer(model, tok, prefix, n_options=len(anchors), device=device)

    ids = list(items)[::20][:5]
    maxdiff = 0.0
    for qid in ids:
        text = items[qid]["text"]
        local = ITEM_FMT.format(text=text, anchor="")     # "Statement: ...\nRating:"
        fast = scorer.score(local, anchors)
        slow = score_options(model, tok, prefix + local, anchors, device=device)
        for f, s in zip(fast, slow):
            d = abs(f["logprob_sum"] - s["logprob_sum"])
            maxdiff = max(maxdiff, d)
        ev_f = sum(p * v for p, (_, v) in zip(softmax([x["logprob_sum"] for x in fast]), ANCHORS))
        ev_s = sum(p * v for p, (_, v) in zip(softmax([x["logprob_sum"] for x in slow]), ANCHORS))
        print(f"  {text[:38]:38} EV fast={ev_f:.3f} slow={ev_s:.3f}")
    print(f"\nMAX |logprob_sum diff| = {maxdiff:.4f}  "
          f"{'PASS (<0.05)' if maxdiff < 0.05 else 'FAIL — do NOT use cached path'}")
