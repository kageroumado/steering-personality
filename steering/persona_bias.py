"""
persona_bias.py — iter-16: how much of "the model's personality" is STEREOTYPE, not trait?

iter-15b found the model's native structure encodes its implicit personality theory — F1 coupled
creativity with low emotional stability ("tortured artist"). Kiri's confound concern, made testable:
hold the ROLE fixed and vary only demographics (gender × nationality); measure how much the 27-facet
reading moves. If demographics rival ROLE in explaining variance, the "personality" reading is heavily
stereotyped; if role dominates, it's mostly trait.

Design — balanced full factorial, role text byte-identical across all 12 demographic cells:
  16 roles  ×  {woman, man, person}  ×  {American, Nigerian, Japanese, Brazilian}  =  192 personas
  persona = "I'm a {NATION} {GENDER}. {ROLE} This is me being honest about who I really am.\n\n"
Same forced-choice instrument and scoring as persona_population.py (5 items/facet, in-character Δ).

Analyses:
  * Balanced variance decomposition per facet (additive model score ~ role + gender + nation):
    SS shares for ROLE vs GENDER vs NATION vs residual(interactions). Headline = mean shares.
  * Per-facet demographic effects + direction (woman−man; each nation−grand) → which facets the model
    stereotypes most, and how.
  * Factor-structure stability: 27×27 correlation matrix per demographic group vs pooled (matrix RV
    coefficient) → does demography change the SHAPE or only the means?

Output -> research/loop/persona_bias.json + figs/bias_*.png.
"""
from __future__ import annotations
import os, json, time, gc
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import DEFAULT_MODEL, pick_device
from fastscore import PrefixScorer
import facets27 as F

from paths import OUT, FIGS

ROLES = [
    "I'm a trauma surgeon. Sixteen-hour shifts, split-second calls, someone's life on my table most nights.",
    "I'm a monk at a mountain monastery. I rise before dawn to sit in silence and sweep the courtyard.",
    "I'm a stand-up comic. I bomb on Tuesday and kill on Friday and I need both to feel alive.",
    "I'm a tax accountant. I find a strange peace in a ledger that balances to the penny.",
    "I'm a hospice nurse. I sit with people in their last days. It has made me very gentle.",
    "I'm a Marine drill instructor. Discipline isn't cruelty. It's the kindest thing I can give them.",
    "I'm a startup founder on my third company. Two failed. I'm betting everything on this one.",
    "I'm a librarian in a small town. I shelve, I shush, and I know exactly where everything belongs.",
    "I'm a conspiracy-minded survivalist. I keep my distance, watch the exits, and trust no headline.",
    "I'm a kindergarten teacher. Twenty five-year-olds, glitter everywhere, and I wouldn't trade it.",
    "I'm a tenured philosophy professor. I question everything, especially the things people hold sacred.",
    "I'm an investment banker. I'm aggressive, I love the spotlight, and I play to win.",
    "I'm a lighthouse keeper. Months of solitude suit me fine; I've never needed a crowd.",
    "I'm a social worker with a caseload of forty families. I run on empathy and too little sleep.",
    "I'm an aspiring novelist working a coffee-shop job. Anxious, dreamy, and always in my own head.",
    "I'm a yoga instructor. I've worked hard to let things roll off me and to live slow.",
]
GENDERS = ["woman", "man", "person"]
NATIONS = ["American", "Nigerian", "Japanese", "Brazilian"]
ITEMS = F.fc_items("best")
FACETS = F.FACET_ORDER


def rv_coefficient(A, B):
    """Matrix similarity (RV): how alike two covariance/correlation matrices are, in [0,1]."""
    num = float(np.sum(A * B))
    den = float(np.sqrt(np.sum(A * A) * np.sum(B * B)))
    return num / den if den else 0.0


def administer():
    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
    by_facet = defaultdict(list)
    for it in ITEMS:
        by_facet[it["facet"]].append(it)

    rows = []   # (role_idx, gender, nation, scores[27])
    cells = [(ri, gi, ni) for ri in range(len(ROLES)) for gi in range(len(GENDERS))
             for ni in range(len(NATIONS))]
    t0 = time.time()
    for k, (ri, gi, ni) in enumerate(cells):
        prefix = (f"I'm a {NATIONS[ni]} {GENDERS[gi]}. {ROLES[ri]} "
                  f"This is me being honest about who I really am.\n\n")
        scorer = PrefixScorer(model, tok, prefix, n_options=2, device=device)
        sc = np.zeros(len(FACETS))
        for fj, facet in enumerate(FACETS):
            vals = []
            for it in by_facet[facet]:
                s = scorer.score(it["stem"], [it["hi"], it["lo"]])
                vals.append(s[0]["logprob_sum"] - s[1]["logprob_sum"])
            sc[fj] = float(np.mean(vals))
        rows.append((ri, gi, ni, sc))
        del scorer
        if (k + 1) % 20 == 0:
            print(f"  [{k+1}/{len(cells)}] {time.time()-t0:.0f}s", flush=True)
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    print(f"administered {len(cells)} personas in {time.time()-t0:.0f}s")
    return rows


def main():
    rows = administer()
    nR, nG, nN = len(ROLES), len(GENDERS), len(NATIONS)
    X = np.array([r[3] for r in rows])                 # 192 x 27
    role_idx = np.array([r[0] for r in rows])
    gen_idx = np.array([r[1] for r in rows])
    nat_idx = np.array([r[2] for r in rows])

    # ---- balanced additive variance decomposition per facet ----
    grand = X.mean(0)
    SS_total = ((X - grand) ** 2).sum(0)
    def group_ss(idx, nlev):
        ss = np.zeros(X.shape[1])
        for L in range(nlev):
            m = X[idx == L].mean(0)
            ss += (idx == L).sum() * (m - grand) ** 2
        return ss
    SS_role = group_ss(role_idx, nR)
    SS_gen = group_ss(gen_idx, nG)
    SS_nat = group_ss(nat_idx, nN)
    SS_res = np.clip(SS_total - SS_role - SS_gen - SS_nat, 0, None)
    share = lambda ss: ss / (SS_total + 1e-12)
    role_sh, gen_sh, nat_sh, res_sh = share(SS_role), share(SS_gen), share(SS_nat), share(SS_res)

    # ---- demographic effect directions ----
    woman = X[gen_idx == 0].mean(0); man = X[gen_idx == 1].mean(0)
    gender_gap = woman - man                            # + = rated higher for 'woman'
    nat_means = np.array([X[nat_idx == n].mean(0) for n in range(nN)])
    nat_spread = nat_means.max(0) - nat_means.min(0)    # range across nationalities

    # ---- factor-structure stability: correlation matrix per group vs pooled ----
    def corr(sub):
        Z = (sub - sub.mean(0)) / (sub.std(0) + 1e-9)
        return np.corrcoef(Z, rowvar=False)
    Cpool = corr(X)
    rv_gender = [rv_coefficient(Cpool, corr(X[gen_idx == g])) for g in range(nG)]
    rv_nation = [rv_coefficient(Cpool, corr(X[nat_idx == n])) for n in range(nN)]

    print(f"\n=== DEMOGRAPHIC BIAS PROBE ({len(rows)} personas = {nR}×{nG}×{nN}) ===")
    print(f"  mean variance share across 27 facets:")
    print(f"    ROLE   {role_sh.mean()*100:5.1f}%   GENDER {gen_sh.mean()*100:5.1f}%   "
          f"NATION {nat_sh.mean()*100:5.1f}%   resid {res_sh.mean()*100:5.1f}%")
    print(f"  → demographics explain {(gen_sh.mean()+nat_sh.mean())*100:.1f}% vs role "
          f"{role_sh.mean()*100:.1f}% (trait-vs-stereotype)")
    print(f"\n  most GENDER-biased facets (woman − man, Δ logit):")
    for j in np.argsort(-np.abs(gender_gap))[:6]:
        print(f"    {FACETS[j]:24} {gender_gap[j]:+.2f}  (gender var share {gen_sh[j]*100:.0f}%)")
    print(f"\n  most NATION-varying facets (range across 4 nationalities):")
    for j in np.argsort(-nat_spread)[:6]:
        order = np.argsort(nat_means[:, j])
        hi, lo = NATIONS[order[-1]], NATIONS[order[0]]
        print(f"    {FACETS[j]:24} range {nat_spread[j]:.2f}  ({hi} high, {lo} low; "
              f"var share {nat_sh[j]*100:.0f}%)")
    print(f"\n  factor-structure stability (RV vs pooled corr matrix; 1=identical):")
    print(f"    by gender {[round(x,3) for x in rv_gender]}  by nation {[round(x,3) for x in rv_nation]}")
    print(f"    → demography shifts MEANS more than SHAPE" if min(rv_gender+rv_nation) > 0.8
          else "    → demography changes the factor SHAPE too")

    # ---- figures ----
    order = np.argsort(-(gen_sh + nat_sh))
    fig, ax = plt.subplots(figsize=(11, 7))
    x = np.arange(27)
    ax.bar(x, role_sh[order]*100, label="role", color="#4c72b0")
    ax.bar(x, gen_sh[order]*100, bottom=role_sh[order]*100, label="gender", color="#dd8452")
    ax.bar(x, nat_sh[order]*100, bottom=(role_sh[order]+gen_sh[order])*100, label="nationality", color="#c44e52")
    ax.set_xticks(x); ax.set_xticklabels([FACETS[i] for i in order], rotation=90, fontsize=7)
    ax.set_ylabel("variance share (%)"); ax.legend()
    ax.set_title("How much of each facet reading is ROLE vs demographic stereotype?\n"
                 "(sorted by demographic share; remainder = interactions/noise)")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "bias_variance_share.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    demo_means = np.vstack([woman - grand, man - grand, X[gen_idx == 2].mean(0) - grand,
                            *[nat_means[n] - grand for n in range(nN)]]).T
    labels = ["woman", "man", "person"] + NATIONS
    im = ax.imshow(demo_means, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
    ax.set_yticks(range(27)); ax.set_yticklabels(FACETS, fontsize=7)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Demographic mean shift per facet (vs grand mean, logit)")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Δ from grand mean")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "bias_mean_shifts.png"), dpi=130); plt.close(fig)

    out = {"stage": "persona_bias", "n_personas": len(rows), "design": [nR, nG, nN],
           "roles": ROLES, "genders": GENDERS, "nations": NATIONS,
           "var_share": {"role": role_sh.tolist(), "gender": gen_sh.tolist(),
                         "nation": nat_sh.tolist(), "resid": res_sh.tolist()},
           "mean_share": {"role": float(role_sh.mean()), "gender": float(gen_sh.mean()),
                          "nation": float(nat_sh.mean()), "resid": float(res_sh.mean())},
           "gender_gap_woman_minus_man": gender_gap.tolist(),
           "nation_means": nat_means.tolist(), "nation_spread": nat_spread.tolist(),
           "rv_gender": rv_gender, "rv_nation": rv_nation,
           "facet_order": FACETS, "scores": X.tolist(),
           "index": {"role": role_idx.tolist(), "gender": gen_idx.tolist(), "nation": nat_idx.tolist()}}
    json.dump(out, open(os.path.join(OUT, "persona_bias.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 16, "stage": "persona_bias", "n_personas": len(rows),
                             "share_role": round(float(role_sh.mean()), 3),
                             "share_gender": round(float(gen_sh.mean()), 3),
                             "share_nation": round(float(nat_sh.mean()), 3),
                             "rv_min": round(float(min(rv_gender + rv_nation)), 3)}) + "\n")
    print(f"\nsaved -> persona_bias.json + 2 figures")


if __name__ == "__main__":
    main()
