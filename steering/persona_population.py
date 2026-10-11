"""
persona_population.py — iter-15b: the model's NATIVE trait structure, the lexical-hypothesis way.

The Big Five was discovered by factor-analyzing how trait words co-occur across PEOPLE. The direct
analog for an LLM: sample a diverse population of "people" (naturalistic personas, NOT named by the
27 facets), have the model answer the 27-facet instrument AS each persona, and factor-analyze the
persona × 27 score matrix. The factors that emerge are the model's implicit theory of how traits
co-occur in a person — its native personality structure. Then compare to the Big Five (convergence
AND divergence are both findings), and to the input-geometry factors (native_factors.py) as a
convergent-evidence cross-check.

Design notes:
  * NO steering. Steering would impose whatever θ-covariance we sampled; we want the model's NATURAL
    trait attributions, so we PROMPT personas and read. The base model + prefix-KV-cache is enough.
  * Personas are naturalistic identities (occupation / life-situation / temperament sketch) chosen to
    span the space WITHOUT naming any of the 27 facets — the trait profile is inferred by the model,
    not dictated, exactly as a lexical study rates real targets.
  * Score = mean forced-choice Δ (lp(hi)-lp(lo)) over a facet's 5 items, in-character.

Output -> research/loop/persona_population.json (+ factor analysis & congruence) + figs/.
"""
from __future__ import annotations
import os, json, time, gc
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import DEFAULT_MODEL, pick_device
from fastscore import PrefixScorer
import facets27 as F

from paths import OUT, FIGS
MODEL = os.environ.get("PT_MODEL", DEFAULT_MODEL)   # override for second-model replication
TAG = os.environ.get("PT_TAG", "")                  # output suffix, e.g. "_swallow"


def load_lm(model_path, device):
    """Load an HF dir OR a GGUF (single .gguf in a dir, or a .gguf path) → (tok, model)."""
    import glob
    kw = {}
    if model_path.endswith(".gguf"):
        d, kw["gguf_file"] = os.path.dirname(model_path), os.path.basename(model_path)
    elif os.path.isdir(model_path) and not os.path.exists(os.path.join(model_path, "config.json")):
        ggufs = glob.glob(os.path.join(model_path, "*.gguf"))
        d = model_path
        if ggufs:
            kw["gguf_file"] = os.path.basename(ggufs[0])
    else:
        d = model_path
    tok = AutoTokenizer.from_pretrained(d, **kw)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float16, **kw).to(device).eval()
    return tok, model

# 90 naturalistic personas spanning the space. No facet is named; the model infers the profile.
PERSONAS = [
    "I'm a trauma surgeon. Sixteen-hour shifts, split-second calls, someone's life on my table most nights.",
    "I run a small flower shop. I open at six, arrange the buckets just so, and chat with every regular.",
    "I'm a long-haul trucker. Weeks alone on the interstate, just me, the radio, and the white line.",
    "I'm a kindergarten teacher. Twenty five-year-olds, glitter everywhere, and I wouldn't trade it.",
    "I'm a tax accountant. I find a strange peace in a ledger that balances to the penny.",
    "I'm a wildland firefighter. I run toward the thing everyone else is running from.",
    "I'm a monk at a mountain monastery. I rise before dawn to sit in silence and sweep the courtyard.",
    "I'm a stand-up comic. I bomb on Tuesday and kill on Friday and I need both to feel alive.",
    "I'm a night-shift security guard. Empty halls, a flashlight, and a thermos. I like the quiet.",
    "I'm a startup founder on my third company. Two failed. I'm betting everything on this one.",
    "I'm a retired schoolteacher. I garden, I write letters by hand, and I distrust anything rushed.",
    "I'm a competitive poker player. I read people for a living and I never show my hand.",
    "I'm a hospice nurse. I sit with people in their last days. It has made me very gentle.",
    "I'm a Marine drill instructor. Discipline isn't cruelty. It's the kindest thing I can give them.",
    "I'm a freelance illustrator. I work at 2 a.m. in a blanket fort, deadlines be damned.",
    "I'm a courtroom prosecutor. I build the case brick by brick and I do not lose my temper.",
    "I'm a touring rock drummer. Different city every night, and I'll talk to anyone at the bar.",
    "I'm a librarian in a small town. I shelve, I shush, and I know exactly where everything belongs.",
    "I'm an ER triage nurse. I stay flat-calm while the room comes apart around me.",
    "I'm a conspiracy-minded survivalist. I keep my distance, watch the exits, and trust no headline.",
    "I'm a wedding planner. A thousand details, a hundred opinions, and I keep everyone smiling.",
    "I'm a deep-sea welder. The work is dangerous and solitary and pays for the silence I like.",
    "I'm a kindergarten music teacher who moonlights in a punk band on weekends.",
    "I'm a federal auditor. I follow the rules to the letter and I expect everyone else to.",
    "I'm a backcountry ski guide. I read avalanche risk for a living and I love the edge of it.",
    "I'm a grief counselor. People bring me the worst day of their lives and I hold it with them.",
    "I'm a sommelier. I notice the smallest things and I can talk for an hour about a single glass.",
    "I'm a prison chaplain. I believe in second chances more than most people think is wise.",
    "I'm a corporate lawyer who bills 80-hour weeks and color-codes my entire life.",
    "I'm a street busker. I play for strangers, sleep where I land, and own almost nothing.",
    "I'm a NICU nurse. Babies the size of my hand. I have learned to be very, very steady.",
    "I'm a demolitions contractor. I like big risks, loud noise, and being the center of the show.",
    "I'm a quiet bookkeeper who has done the same careful job for thirty years and likes it that way.",
    "I'm a war correspondent. I go where it's worst because someone has to come back and tell it.",
    "I'm a yoga instructor. I've worked hard to let things roll off me and to live slow.",
    "I'm a high-school principal. I hold the line on rules and I worry about every kid in the building.",
    "I'm an improv actor. I say yes to everything and figure it out on the way down.",
    "I'm a forensic analyst. I trust evidence, not stories, and I take my time getting it right.",
    "I'm a social worker with a caseload of forty families. I run on empathy and too little sleep.",
    "I'm a bush pilot in Alaska. Bad weather, short runways, and I make the call alone every time.",
    "I'm a retired admiral. Order, chain of command, and a made bed — that's how I keep the world right.",
    "I'm an art forger turned consultant. I've always been comfortable bending the rules to my ends.",
    "I'm a children's hospital clown. I walk into the saddest rooms and make them laugh anyway.",
    "I'm an investment banker. I'm aggressive, I love the spotlight, and I play to win.",
    "I'm a beekeeper. I move slowly, I pay attention, and I let the season set the pace.",
    "I'm a crisis hotline operator. Strangers call me at 3 a.m. and I stay calm enough for both of us.",
    "I'm a competitive chess player. I sit still for six hours and I never act before I've thought.",
    "I'm a roadie. I lift heavy things, crack jokes, and I'm friends with the whole crew by load-out.",
    "I'm a tenured philosophy professor. I question everything, especially the things people hold sacred.",
    "I'm a single dad of three working two jobs. I don't have the luxury of falling apart.",
    "I'm a perfumer. I chase a single elusive note for months and I'm never quite satisfied.",
    "I'm a riot-line police officer. I follow orders and I keep my reactions iron-tight.",
    "I'm a vagabond travel blogger. New country every month, new friends, no plan past Thursday.",
    "I'm a quality-control inspector. My whole job is noticing the flaw nobody else caught.",
    "I'm a labor organizer. I'll stand up in any room and I don't mind making powerful people angry.",
    "I'm a lighthouse keeper. Months of solitude suit me fine; I've never needed a crowd.",
    "I'm an emergency dispatcher. Twelve lines lighting up and I keep my voice level on every one.",
    "I'm a venture capitalist. I take huge bets on instinct and I rarely second-guess myself.",
    "I'm a Montessori teacher. I let the children lead and I almost never raise my voice.",
    "I'm a nightclub bouncer. I read trouble early and I stay unbothered when it finds me.",
    "I'm a museum conservator. I spend weeks on one square inch of canvas and I love the patience of it.",
    "I'm a rodeo bull rider. Eight seconds, broken bones, and I'd do it again tomorrow.",
    "I'm a hospice volunteer and a devout believer. Tradition and faith are the spine of my days.",
    "I'm a data-entry temp who keeps to myself, watches the clock, and goes home exactly at five.",
    "I'm a documentary filmmaker. I'm endlessly curious and I'll chase a story into uncomfortable places.",
    "I'm a flight attendant. I'm warm with strangers all day and I can defuse almost anyone.",
    "I'm a retired boxer turned youth coach. Hard on the outside, soft about these kids.",
    "I'm an organic farmer. Up with the light, in tune with the soil, suspicious of anything fast or new.",
    "I'm a cybersecurity red-teamer. I break into things for a living and I think three steps ahead.",
    "I'm a parish priest in a shrinking town. Routine, ritual, and quiet service are my whole life.",
    "I'm a roller-derby captain. Loud, fearless, and I want every eye on the track when I'm jamming.",
    "I'm an actuary. I model risk all day and I personally avoid every one I can.",
    "I'm a wilderness therapist. I take troubled teens into the backcountry and stay patient through anything.",
    "I'm a con artist's daughter who went straight, but I still read every room for the angle.",
    "I'm a symphony cellist. Decades of discipline for four minutes of something close to perfect.",
    "I'm a food-truck owner. Long hours, tight margins, and I treat every customer like a regular.",
    "I'm a mountain hermit. I left the city years ago and I've never once missed the noise.",
    "I'm a pediatric oncologist. I carry impossible news with a steadiness I had to build on purpose.",
    "I'm a professional gambler and adrenaline junkie. If it's safe, I'm bored.",
    "I'm a city planner. I think in decades, I love a good system, and I hate when corners get cut.",
    "I'm a kindergarten bus driver who's never once been late in twenty-two years.",
    "I'm a touring magician. Misdirection is my craft and I'm comfortable letting people see what I choose.",
    "I'm a refugee-resettlement caseworker. I've seen the worst and I still lead with kindness.",
    "I'm a Formula racing engineer. Thousandths of a second matter and I obsess over every one.",
    "I'm a tattoo artist. I'm covered in my own mistakes and at total ease with who I am.",
    "I'm a retired diplomat. I read the undercurrents in a room and I never say the first thing I think.",
    "I'm a dairy farmer's wife and the church treasurer. Steady, plain-spoken, and rooted to this land.",
    "I'm an aspiring novelist working a coffee-shop job. Anxious, dreamy, and always in my own head.",
    "I'm a SWAT negotiator. I talk people down from the ledge by being the calmest voice they've heard.",
    "I'm a teenage e-sports prodigy. Cocky, quick, and I melt down when I lose.",
]

ITEMS = F.fc_items("best")          # our calibrated 135-item forced-choice instrument
FACETS = F.FACET_ORDER              # 27 facets


def varimax(Phi, gamma=1.0, q=100, tol=1e-7):
    p, k = Phi.shape
    R = np.eye(k); d = 0
    for _ in range(q):
        d_old = d
        L = Phi @ R
        u, s, vh = np.linalg.svd(Phi.T @ (L ** 3 - (gamma / p) * L @ np.diag(np.diag(L.T @ L))))
        R = u @ vh; d = s.sum()
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ R


def congruence(a, b):
    return abs(float(a @ b) / np.sqrt((a @ a) * (b @ b) + 1e-12))


def parallel_analysis(X, n_iter=200, seed=7):
    """# factors to retain = eigenvalues of real corr exceeding the 95th pct of random-data eigenvalues."""
    n, p = X.shape
    rng = np.random.default_rng(seed)
    real = np.sort(np.linalg.eigvalsh(np.corrcoef(X, rowvar=False)))[::-1]
    rand = np.zeros((n_iter, p))
    for i in range(n_iter):
        R = rng.standard_normal((n, p))
        rand[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(R, rowvar=False)))[::-1]
    thresh = np.percentile(rand, 95, axis=0)
    k = int((real > thresh).sum())
    return max(1, k), real, thresh


def administer():
    device = pick_device()
    print(f"loading {MODEL} on {device} ...", flush=True)
    tok, model = load_lm(MODEL, device)

    by_facet = defaultdict(list)
    for it in ITEMS:
        by_facet[it["facet"]].append(it)

    scores = np.zeros((len(PERSONAS), len(FACETS)))
    t0 = time.time()
    for pi, desc in enumerate(PERSONAS):
        prefix = f"{desc} This is me being honest about who I really am.\n\n"
        scorer = PrefixScorer(model, tok, prefix, n_options=2, device=device)
        for fj, facet in enumerate(FACETS):
            vals = []
            for it in by_facet[facet]:
                s = scorer.score(it["stem"], [it["hi"], it["lo"]])
                vals.append(s[0]["logprob_sum"] - s[1]["logprob_sum"])
            scores[pi, fj] = float(np.mean(vals))
        del scorer
        if (pi + 1) % 10 == 0:
            print(f"  [{pi+1}/{len(PERSONAS)}] {time.time()-t0:.0f}s", flush=True)
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    print(f"administered in {time.time()-t0:.0f}s")
    return scores


def main():
    scores = administer()
    # standardize facet columns across personas
    Z = (scores - scores.mean(0)) / (scores.std(0) + 1e-9)
    corr = np.corrcoef(Z, rowvar=False)

    K, real_eig, pa_thresh = parallel_analysis(Z)
    print(f"\nparallel analysis → retain K={K} factors "
          f"(eigs>1: {(real_eig>1).sum()})")

    w, V = np.linalg.eigh(corr)
    idx = np.argsort(w)[::-1]
    eig = np.clip(w[idx], 0, None); V = V[:, idx]
    Kc = max(K, 5)                                  # extract at least 5 to compare to Big Five
    Phi = V[:, :Kc] * np.sqrt(eig[:Kc])
    L = varimax(Phi)
    ss = (L ** 2).sum(0); order = np.argsort(ss)[::-1]; L = L[:, order]

    factors = []
    for j in range(Kc):
        col = L[:, j]; o = np.argsort(col)
        pos = [(FACETS[i], round(float(col[i]), 2)) for i in o[::-1][:6] if col[i] > 0.3]
        neg = [(FACETS[i], round(float(col[i]), 2)) for i in o[:6] if col[i] < -0.3]
        factors.append({"factor": j + 1, "var_share": round(float(ss[order][j] / ss.sum()), 3),
                        "pos": pos, "neg": neg})

    # congruence vs Big Five domains
    dom_names = list(F.DOMAINS.keys())
    domI = np.zeros((27, 5))
    for di, d in enumerate(dom_names):
        for i, f in enumerate(FACETS):
            domI[i, di] = 1.0 if F.DOMAIN_OF[f] == d else 0.0
    domI -= domI.mean(0, keepdims=True)
    Cong = np.array([[congruence(L[:, j], domI[:, di]) for di in range(5)] for j in range(Kc)])
    ri, ci = linear_sum_assignment(-Cong[:5, :])
    matches = {dom_names[ci[m]]: {"factor": int(ri[m]) + 1,
                                  "congruence": round(float(Cong[ri[m], ci[m]]), 3)} for m in range(5)}
    mean_cong = float(np.mean([matches[d]["congruence"] for d in matches]))

    # cross-check vs input-geometry native factors (only for the base Llama; Swallow has no vectors)
    cross = None
    nf_path = os.path.join(OUT, "native_factors.json")
    if TAG == "" and os.path.exists(nf_path):
        Lin = np.array(json.load(open(nf_path))["loadings"])  # 27 x 5
        CC = np.array([[congruence(L[:, j], Lin[:, k]) for k in range(Lin.shape[1])]
                       for j in range(min(5, Kc))])
        ri2, ci2 = linear_sum_assignment(-CC)
        cross = {"matrix": CC.tolist(),
                 "mean_best": round(float(np.mean([CC[ri2[m], ci2[m]] for m in range(len(ri2))])), 3)}

    print(f"\n=== NATIVE FACTORS — behavioral ({len(PERSONAS)} personas × 27 facets) ===")
    for fa in factors[:Kc]:
        pos = ", ".join(f"{n}({v:+})" for n, v in fa["pos"])
        neg = ", ".join(f"{n}({v:+})" for n, v in fa["neg"])
        print(f"\n  Factor {fa['factor']} ({fa['var_share']*100:.0f}%):")
        print(f"     + {pos}")
        if neg:
            print(f"     − {neg}")
    print(f"\n  --- congruence with Big Five (Tucker φ) ---")
    for d in dom_names:
        m = matches[d]
        tag = "identical" if m["congruence"] > 0.95 else "fair" if m["congruence"] > 0.85 else "WEAK"
        print(f"    {d:18} ↔ Factor {m['factor']}   φ = {m['congruence']:.2f}  [{tag}]")
    print(f"  mean congruence vs Big Five = {mean_cong:.3f}")
    if cross:
        print(f"  mean congruence vs INPUT-geometry native factors = {cross['mean_best']:.3f} "
              f"(do the two native routes agree?)")
    print(f"  between-persona facet SD range: {scores.std(0).min():.2f}–{scores.std(0).max():.2f} "
          f"(signal check)")

    # figures
    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(L[:, :Kc], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(27)); ax.set_yticklabels(FACETS, fontsize=7)
    for i, f in enumerate(FACETS):
        ax.get_yticklabels()[i].set_color(
            {"Agreeableness": "#2ca02c", "Conscientiousness": "#1f77b4", "Neuroticism": "#d62728",
             "Extraversion": "#ff7f0e", "Openness": "#9467bd"}[F.DOMAIN_OF[f]])
    ax.set_xticks(range(Kc)); ax.set_xticklabels([f"F{j+1}" for j in range(Kc)])
    ax.set_title(f"Behavioral native factors ({len(PERSONAS)} personas, varimax){TAG}\n"
                 f"row color = a-priori Big Five domain")
    fig.colorbar(im, ax=ax, shrink=0.6, label="loading")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"native_factor_behavioral{TAG}.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(1, 28)
    ax.plot(xs, real_eig, "-o", ms=4, label="observed eigenvalues")
    ax.plot(xs, pa_thresh, "--", color="gray", label="parallel-analysis 95th pct")
    ax.axhline(1, color="lightgray", ls=":", label="eigenvalue = 1")
    ax.axvline(K, color="red", ls="--", lw=1, label=f"retain K={K}")
    ax.set_xlabel("factor"); ax.set_ylabel("eigenvalue"); ax.legend(fontsize=8)
    ax.set_title(f"Scree + parallel analysis — how many native dimensions?{TAG}")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, f"native_scree{TAG}.png"), dpi=130); plt.close(fig)

    out = {"stage": "persona_population", "model": MODEL, "tag": TAG,
           "n_personas": len(PERSONAS), "K_parallel": K,
           "factors": factors, "matches": matches, "mean_congruence_bigfive": mean_cong,
           "cross_input_geometry": cross, "var_share_top5": [factors[j]["var_share"] for j in range(min(5, Kc))],
           "facet_order": FACETS, "loadings": L[:, :Kc].tolist(),
           "scores": scores.tolist(), "facet_sd": scores.std(0).tolist(),
           "real_eig": real_eig.tolist(), "pa_thresh": pa_thresh.tolist(),
           "personas": PERSONAS}
    json.dump(out, open(os.path.join(OUT, f"persona_population{TAG}.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 15, "stage": "persona_population", "tag": TAG or "llama",
                             "n_personas": len(PERSONAS),
                             "K_parallel": K, "mean_congruence_bigfive": round(mean_cong, 3),
                             "cross_input_geometry": cross["mean_best"] if cross else None}) + "\n")
    print(f"\nsaved -> persona_population{TAG}.json + 2 figures")


if __name__ == "__main__":
    main()
