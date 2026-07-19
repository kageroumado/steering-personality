"""
formalize_native.py — iter-19: the formal native model (the "new tool") at N=200.

The native metatrait structure is now established (D20), instrument-robust (D22), demographics-robust
(D21), and model-general (D23). This formalizes it as a proposable representation: scale to N=200
personas for a stable solution, fix the number of factors by parallel analysis, name the metatraits,
and report psychometric RELIABILITY — split-half factor replicability + per-factor internal
consistency (Cronbach's α and McDonald's ω). The deliverable is a named k-factor model with loadings
and reliabilities, plus its congruence to the Big Five (the comparison point).

Reuses the 90 personas already scored in persona_population.json (byte-identical procedure) and
administers 110 NEW naturalistic personas → N=200 on the primary model (Llama-3.1-8B).

Output -> research/loop/native_model.json + figs/native_model_*.png.
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

OUT = os.path.expanduser("~/Developer/personality-test-27d/research/loop")
FIGS = os.path.join(OUT, "figs")
ITEMS = F.fc_items("best")
FACETS = F.FACET_ORDER

# 110 NEW naturalistic personas (distinct from the 90 in persona_population.py), spanning the space.
NEW_PERSONAS = [
    "I'm a high-school chemistry teacher who's been doing the same lesson plans for fifteen years.",
    "I'm a freelance bounty hunter. Unpredictable hours, lots of risk, and I answer to no one.",
    "I'm a retired postal worker. I walked the same route for thirty years and loved every quiet mile.",
    "I'm a celebrity gossip blogger. I'm nosy, fast, and I'll say the thing no one else will.",
    "I'm an air-traffic controller. Dozens of planes, zero margin for error, and I never let it rattle me.",
    "I'm a tarot reader at a seaside town. I trust my intuition over any rulebook.",
    "I'm a Navy submariner. Months underwater in a steel tube; you learn to keep yourself very steady.",
    "I'm a kindergartener's stay-at-home dad. My days are snacks, scraped knees, and endless patience.",
    "I'm a competitive bodybuilder. Every meal weighed, every rep counted, no day off, ever.",
    "I'm a busker's apprentice who ran away to join the circus at seventeen.",
    "I'm a corporate whistleblower in hiding. I did the right thing and it cost me everything.",
    "I'm a wedding DJ. I read the crowd, keep the floor packed, and I'm friends with everyone by midnight.",
    "I'm an actuary turned doomsday prepper. I've run the numbers and I trust my bunker, not the news.",
    "I'm a pediatric speech therapist. Slow, gentle, repetitive work, and I adore it.",
    "I'm a mixed martial artist. I love the danger, the crowd, and proving myself in the cage.",
    "I'm a night auditor at a roadside motel. Just me and the ledger from eleven to seven. Suits me.",
    "I'm a kindergarten art teacher who repaints her classroom a new color every single year.",
    "I'm a court stenographer. I record every word exactly and I never inject myself into the room.",
    "I'm a heli-skiing guide in the Andes. I court avalanche country for a living and I live for it.",
    "I'm a suicide-prevention counselor. People hand me their darkest hour and I stay calm enough to hold it.",
    "I'm a Michelin-obsessed line cook. I'll remake a plate ten times until the sauce is exactly right.",
    "I'm a parole officer. I believe in redemption more than my colleagues think is sensible.",
    "I'm a hedge fund quant. I work ninety-hour weeks and I optimize literally everything in my life.",
    "I'm a street-corner preacher. I own nothing, I fear nothing, and I'll talk to any soul who passes.",
    "I'm a labor-and-delivery midwife. Tiny lives in my hands; I've learned to be unshakeably calm.",
    "I'm a pyrotechnician. I blow things up for festivals and I love being the loudest part of the night.",
    "I'm a parish bookkeeper who has reconciled the same accounts the same way for twenty-five years.",
    "I'm an embedded war photographer. I go toward the gunfire because the world needs to see it.",
    "I'm a tai-chi instructor in a city park. I've spent decades learning to let everything go.",
    "I'm a boarding-school headmaster. Rules, routine, and the worry I carry for every student.",
    "I'm a sketch comedian. I'll try any bit, follow any tangent, and improvise my way out.",
    "I'm a crime-scene investigator. I follow the evidence coldly and I take all the time it needs.",
    "I'm a hospice social worker carrying thirty dying patients. Compassion is all I run on.",
    "I'm a glacier-research pilot. Whiteouts and short fuel; every landing is mine to call alone.",
    "I'm a retired four-star general. A made bed, a clear chain of command, order in all things.",
    "I'm a card counter banned from three casinos. I've always been at ease working a system to my edge.",
    "I'm a therapy-dog handler visiting children's wards. I walk into grief and bring a little joy.",
    "I'm a private equity raider. I'm ruthless, I crave the spotlight, and I close every deal.",
    "I'm an apiarist and herbalist. I move slow, watch close, and let the seasons run the calendar.",
    "I'm a 911 fire dispatcher. Six emergencies at once and my voice never climbs a single note.",
    "I'm a grandmaster's sparring partner. I sit motionless for hours and never move before I've thought.",
    "I'm a touring sound engineer. Heavy cases, late nights, on a first-name basis with the whole crew.",
    "I'm a comparative-religion lecturer. I poke at every certainty, especially the sacred ones.",
    "I'm a widowed father of four working the night shift. Falling apart isn't on the menu.",
    "I'm a master watchmaker. I'll chase a thousandth-of-a-millimeter fault for weeks and still not be happy.",
    "I'm a mounted park ranger. I keep order on the trails and I keep my reactions locked down tight.",
    "I'm a couchsurfing nomad with a one-way ticket and no plan past next Tuesday.",
    "I'm a pharmaceutical QA auditor. My entire job is finding the defect everyone else signed off on.",
    "I'm a tenant-union firebrand. I'll stand up in any meeting and I don't mind angering the landlord.",
    "I'm a fire-tower lookout. One room, one mountain, one summer of glorious solitude.",
    "I'm a coast-guard rescue swimmer. Twenty-foot seas and I stay flat-calm hauling people out.",
    "I'm a seed-stage angel investor. I bet on a founder's eyes in ten minutes and I never look back.",
    "I'm a Waldorf preschool teacher. I let the kids set the rhythm and I almost never raise my voice.",
    "I'm a festival-gate security lead. I spot trouble three people deep and I don't get flustered.",
    "I'm a fresco restorer. Months on a single cherub's wing, and I love the slowness of it.",
    "I'm a cliff-diving competitor. Twenty-seven meters, a half-second of terror, and I'm hooked.",
    "I'm a devout deacon and the church's treasurer. Ritual, tradition, and quiet service are my whole life.",
    "I'm a graveyard-shift toll-booth attendant who clocks out at exactly the same minute every night.",
    "I'm an investigative podcaster. I'm relentlessly curious and I'll dig into places I'm not welcome.",
    "I'm a first-class purser. I'm warm with strangers all day and I can soothe almost any tantrum.",
    "I'm a retired wrestler turned middle-school coach. Gruff on the outside, a marshmallow about these kids.",
    "I'm a no-till regenerative farmer. Up with the sun, married to the soil, wary of anything new or fast.",
    "I'm a penetration tester. I break into banks for a living and I'm always five moves ahead.",
    "I'm a cloistered nun. Routine, prayer, silence, and devotion fill every hour of my day.",
    "I'm a flat-track roller-derby jammer. Loud, fearless, and I want all eyes on me when I break through.",
    "I'm a reinsurance underwriter. I price catastrophe all day and I dodge every personal risk I can.",
    "I'm an outdoor-behavioral guide for at-risk teens. I stay patient through tantrums, storms, anything.",
    "I'm a reformed pickpocket who went straight, but I still clock every mark and exit in a room.",
    "I'm a principal violist. Thirty years of discipline for a few transcendent minutes a night.",
    "I'm a ramen-cart owner. Brutal hours, razor margins, and I greet every late-night regular by name.",
    "I'm a desert recluse who left the grid a decade ago and has never once missed a soul.",
    "I'm a pediatric heart surgeon. I carry impossible odds with a calm I had to construct on purpose.",
    "I'm a high-stakes sports bettor and adrenaline chaser. If there's no risk, I'm already bored.",
    "I'm a transit planner. I think in thirty-year horizons, I love a clean system, and shortcuts enrage me.",
    "I'm a school-bus driver with a perfect twenty-year on-time record and not one fender-bender.",
    "I'm a close-up street magician. Misdirection is my craft; I let people see exactly what I choose.",
    "I'm an asylum-intake caseworker. I've heard the worst humanity does and I still lead with mercy.",
    "I'm an aerospace stress engineer. Microscopic margins, and I obsess over every decimal place.",
    "I'm a sideshow tattooist. I'm inked head to toe with my own history and entirely at peace with it.",
    "I'm a retired ambassador. I read the room's undercurrents and never blurt the first thing I think.",
    "I'm a rancher's wife and the 4-H club secretary. Steady, plain-spoken, rooted to this dry land.",
    "I'm an MFA dropout pulling espresso. Anxious, dreamy, perpetually lost inside my own head.",
    "I'm a hostage negotiator. I talk people off the ledge by being the steadiest voice they've heard all year.",
    "I'm a fifteen-year-old chess prodigy. Cocky, lightning-fast, and I unravel completely when I lose.",
    "I'm an oncology pharmacist. Precise to the milligram, because a rounding error is somebody's life.",
    "I'm a wandering portrait painter who trades canvases for a couch and a meal.",
    "I'm a SWAT team breacher. I follow the command, I move first through the door, ice in my veins.",
    "I'm a folk-festival fiddler. I laugh easy, I'll jam with anyone, and I never plan past the next tune.",
    "I'm a forensic accountant chasing shell companies. I trust the paper trail, never the alibi.",
    "I'm a long-distance caregiver for my mother with dementia. Endless patience, and quietly exhausted.",
    "I'm a base-jumping cinematographer. I throw myself off cliffs for the shot and I'd do it twice.",
    "I'm a small-town notary. Careful, literal, and I like that every document has its proper place.",
    "I'm a union shop steward on the factory floor. I'll grieve any unfair write-up and I don't back down.",
    "I'm an Antarctic winter-over technician. Six months of dark and ice, and the isolation suits me fine.",
    "I'm a neonatal transport nurse. Fragile lives at thirty thousand feet; I've made myself unshakeable.",
    "I'm a carnival ride operator and self-styled showman. I live for a screaming, delighted crowd.",
    "I'm a temple caretaker. I sweep, I tend, I keep to myself, and the quiet has never once bored me.",
    "I'm an ICU intensivist. The worst news in the building lands on me and I deliver it level and clear.",
    "I'm a professional dart hustler and barfly. Safe is boring; I like the action and the side bet.",
    "I'm an urban-forestry director. I plan in decades, I adore a good system, and I hate a cut corner.",
    "I'm a kindergarten crossing guard who has never been a minute late in eighteen winters.",
    "I'm a mentalist on a cruise ship. Sleight of mind is my trade; people see only what I let them.",
    "I'm a disaster-relief logistics lead. I've witnessed the worst days on earth and I still lead with heart.",
    "I'm a Formula E telemetry engineer. Hundredths of a second decide it and I fixate on every one.",
    "I'm a body-piercer and shop owner. Scarred, pierced, dyed, and utterly comfortable in my own skin.",
    "I'm a former spymaster, now retired. I sense every undercurrent and I never tip my hand.",
    "I'm an Amish-raised carpenter who left the order but kept its plain, steady, rooted ways.",
    "I'm a perpetual grad student in philosophy. I doubt everything, especially what everyone agrees on.",
    "I'm a triplet mom running a daycare from home. Two jobs of patience and not enough sleep, ever.",
    "I'm a master sommelier-in-training. I notice the faintest note and I'm never once fully satisfied.",
]


def varimax(Phi, gamma=1.0, q=200, tol=1e-8):
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


def parallel_analysis(X, n_iter=300, seed=11):
    n, p = X.shape
    rng = np.random.default_rng(seed)
    real = np.sort(np.linalg.eigvalsh(np.corrcoef(X, rowvar=False)))[::-1]
    rand = np.zeros((n_iter, p))
    for i in range(n_iter):
        rand[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal((n, p)), rowvar=False)))[::-1]
    thresh = np.percentile(rand, 95, axis=0)
    return max(1, int((real > thresh).sum())), real, thresh


def efa(X, K):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    C = np.corrcoef(Z, rowvar=False)
    w, V = np.linalg.eigh(C); idx = np.argsort(w)[::-1]
    eig = np.clip(w[idx], 0, None); V = V[:, idx]
    L = varimax(V[:, :K] * np.sqrt(eig[:K]))
    L = L[:, np.argsort((L ** 2).sum(0))[::-1]]
    return L, Z


def cronbach_alpha(items):
    """items: n x m (personas x facets-in-scale, sign-aligned)."""
    m = items.shape[1]
    if m < 2:
        return float("nan")
    var_sum = items.var(0, ddof=1).sum()
    tot_var = items.sum(1).var(ddof=1)
    return float((m / (m - 1)) * (1 - var_sum / tot_var)) if tot_var > 0 else float("nan")


def mcdonald_omega(load):
    """omega from a single factor's loadings (standardized): (Σλ)² / [(Σλ)² + Σ(1-λ²)]."""
    s = load.sum() ** 2
    err = (1 - load ** 2).sum()
    return float(s / (s + err)) if (s + err) > 0 else float("nan")


def administer_new(persona_list):
    device = pick_device()
    print(f"loading {DEFAULT_MODEL} on {device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODEL, dtype=torch.float16).to(device).eval()
    by_facet = defaultdict(list)
    for it in ITEMS:
        by_facet[it["facet"]].append(it)
    scores = np.zeros((len(persona_list), len(FACETS)))
    t0 = time.time()
    for pi, desc in enumerate(persona_list):
        scorer = PrefixScorer(model, tok, f"{desc} This is me being honest about who I really am.\n\n",
                              n_options=2, device=device)
        for fj, facet in enumerate(FACETS):
            vals = []
            for it in by_facet[facet]:
                s = scorer.score(it["stem"], [it["hi"], it["lo"]])
                vals.append(s[0]["logprob_sum"] - s[1]["logprob_sum"])
            scores[pi, fj] = float(np.mean(vals))
        del scorer
        if (pi + 1) % 10 == 0:
            print(f"  [{pi+1}/{len(persona_list)}] {time.time()-t0:.0f}s", flush=True)
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    print(f"administered {len(persona_list)} new personas in {time.time()-t0:.0f}s")
    return scores


def main():
    pop = json.load(open(os.path.join(OUT, "persona_population.json")))
    assert pop["facet_order"] == FACETS, "facet order mismatch"
    old_scores = np.array(pop["scores"]); old_personas = pop["personas"]
    new_scores = administer_new(NEW_PERSONAS)
    X = np.vstack([old_scores, new_scores])             # N x 27
    personas = old_personas + NEW_PERSONAS
    N = X.shape[0]

    K, real_eig, pa_thresh = parallel_analysis(X)
    print(f"\nN={N}: parallel analysis → K={K} factors (eigs>1: {(real_eig>1).sum()})")
    Kuse = max(K, 5)
    L, Z = efa(X, Kuse)

    # name factors + per-factor reliability
    factors = []
    for j in range(Kuse):
        col = L[:, j]; o = np.argsort(col)
        pos = [(FACETS[i], round(float(col[i]), 2)) for i in o[::-1] if col[i] > 0.35][:6]
        neg = [(FACETS[i], round(float(col[i]), 2)) for i in o if col[i] < -0.35][:6]
        # scale = facets with |loading|>0.35, sign-aligned
        idx = [i for i in range(27) if abs(col[i]) > 0.35]
        signed = Z[:, idx] * np.sign(col[idx])
        alpha = cronbach_alpha(signed) if len(idx) >= 2 else float("nan")
        omega = mcdonald_omega(np.abs(col[idx])) if len(idx) >= 2 else float("nan")
        factors.append({"factor": j + 1, "var_share": round(float((col ** 2).sum() / (L ** 2).sum()), 3),
                        "n_facets": len(idx), "alpha": round(alpha, 3), "omega": round(omega, 3),
                        "pos": pos, "neg": neg})

    # split-half factor replicability
    rng = np.random.default_rng(7)
    perm = rng.permutation(N)
    h1, h2 = perm[:N // 2], perm[N // 2:]
    L1, _ = efa(X[h1], Kuse); L2, _ = efa(X[h2], Kuse)
    C = np.array([[congruence(L1[:, i], L2[:, j]) for j in range(Kuse)] for i in range(Kuse)])
    ri, ci = linear_sum_assignment(-C)
    split_half = float(np.mean([C[ri[m], ci[m]] for m in range(Kuse)]))

    # congruence vs Big Five (a-priori DOMAINS)
    domI = np.zeros((27, 5))
    for di, d in enumerate(F.DOMAINS):
        for i, f in enumerate(FACETS):
            domI[i, di] = 1.0 if F.DOMAIN_OF[f] == d else 0.0
    domI -= domI.mean(0, keepdims=True)
    CB = np.array([[congruence(L[:, j], domI[:, di]) for di in range(5)] for j in range(5)])
    ri2, ci2 = linear_sum_assignment(-CB)
    big5 = float(np.mean([CB[ri2[m], ci2[m]] for m in range(5)]))

    print(f"\n=== FORMAL NATIVE MODEL (N={N}, K={Kuse}) ===")
    for fa in factors:
        pos = ", ".join(f"{n}({v:+})" for n, v in fa["pos"])
        neg = ", ".join(f"{n}({v:+})" for n, v in fa["neg"])
        print(f"\n  Factor {fa['factor']} ({fa['var_share']*100:.0f}% var | α={fa['alpha']:.2f} ω={fa['omega']:.2f} | {fa['n_facets']} facets):")
        print(f"     + {pos}")
        if neg:
            print(f"     − {neg}")
    print(f"\n  split-half factor replicability (Tucker φ) = {split_half:.3f}  "
          f"({'replicable' if split_half > 0.9 else 'fair' if split_half > 0.85 else 'unstable'})")
    print(f"  congruence vs Big Five = {big5:.3f}  (still diverges)")
    print(f"  mean per-factor α = {np.nanmean([f['alpha'] for f in factors]):.3f}, "
          f"ω = {np.nanmean([f['omega'] for f in factors]):.3f}")

    # figures
    fig, ax = plt.subplots(figsize=(8.5, 9))
    im = ax.imshow(L, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(27)); ax.set_yticklabels(FACETS, fontsize=7)
    for i, f in enumerate(FACETS):
        ax.get_yticklabels()[i].set_color(
            {"Agreeableness": "#2ca02c", "Conscientiousness": "#1f77b4", "Neuroticism": "#d62728",
             "Extraversion": "#ff7f0e", "Openness": "#9467bd"}[F.DOMAIN_OF[f]])
    ax.set_xticks(range(Kuse))
    ax.set_xticklabels([f"F{j+1}\nα={factors[j]['alpha']:.2f}" for j in range(Kuse)], fontsize=8)
    ax.set_title(f"Formal native model (N={N}, varimax)\nrow color = Big Five domain; "
                 f"split-half φ={split_half:.2f}, vs Big Five φ={big5:.2f}")
    fig.colorbar(im, ax=ax, shrink=0.6, label="loading")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "native_model_loadings.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(1, 28)
    ax.plot(xs, real_eig, "-o", ms=4, label="observed eigenvalues")
    ax.plot(xs, pa_thresh, "--", color="gray", label="parallel-analysis 95th pct")
    ax.axhline(1, color="lightgray", ls=":")
    ax.axvline(K, color="red", ls="--", lw=1, label=f"retain K={K}")
    ax.set_xlabel("factor"); ax.set_ylabel("eigenvalue"); ax.legend(fontsize=8)
    ax.set_title(f"Native model dimensionality (N={N})")
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, "native_model_scree.png"), dpi=130); plt.close(fig)

    out = {"stage": "native_model", "N": N, "K_parallel": K, "K_used": Kuse,
           "factors": factors, "split_half_congruence": split_half, "vs_bigfive": big5,
           "mean_alpha": float(np.nanmean([f["alpha"] for f in factors])),
           "mean_omega": float(np.nanmean([f["omega"] for f in factors])),
           "loadings": L.tolist(), "facet_order": FACETS, "scores": X.tolist(),
           "real_eig": real_eig.tolist(), "personas": personas}
    json.dump(out, open(os.path.join(OUT, "native_model.json"), "w"), indent=2)
    with open(os.path.join(OUT, "metrics.jsonl"), "a") as fh:
        fh.write(json.dumps({"iter": 19, "stage": "native_model", "N": N, "K": Kuse,
                             "split_half_congruence": round(split_half, 3),
                             "vs_bigfive": round(big5, 3),
                             "mean_alpha": round(float(np.nanmean([f["alpha"] for f in factors])), 3)}) + "\n")
    print(f"\nsaved -> native_model.json + 2 figures")


if __name__ == "__main__":
    main()
