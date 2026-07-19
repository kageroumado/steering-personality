"""
itembank.py — role-A authored candidate items (iteration-002).

Role A = the authoring LLM (Claude). These are IPIP-style self-report statements (rated on the
6-point accuracy anchor, same readout as SPI-135) authored to the §8 doctrine:
  * OPAQUE-leaning: avoid naming the trait; bury the signal in a concrete behavior.
  * EPISODIC / CONCRETE over global self-summaries; diverse life domains per facet.
  * BALANCED KEYS: mix forward (+1) and reverse (-1) so the scale isn't acquiescence-driven.
  * DESIRABILITY-NEUTRAL: no option is the obvious "better person"; differences are about
    which motivation, not how good.
  * No double-barreled, no negation traps, plain and vivid.

10 candidates/facet (5 facets = 50). Role B administers these to the steered persona battery;
selection keeps the highest θ-loading / lowest cross-loading subset per facet.

Schema per item: {"id", "facet", "key": +1|-1, "domain", "text"}.
key = +1 means high facet → high accuracy; key = -1 is reverse-keyed (scored 7-EV).
"""

ITEMS = [
    # ---------------- SelfControl (high = resists impulse, disciplined) ----------------
    {"id": "SC01", "facet": "SelfControl", "key": +1, "domain": "money",
     "text": "Set aside money each month before I let myself spend any of it."},
    {"id": "SC02", "facet": "SelfControl", "key": +1, "domain": "work",
     "text": "Finish the dull part of a task before I let myself take a break."},
    {"id": "SC03", "facet": "SelfControl", "key": +1, "domain": "conflict",
     "text": "Wait until I've cooled off before answering a message that irritated me."},
    {"id": "SC04", "facet": "SelfControl", "key": +1, "domain": "leisure",
     "text": "Stick to the plan I made even when something more fun comes up."},
    {"id": "SC05", "facet": "SelfControl", "key": +1, "domain": "food",
     "text": "Leave the second helping on the table when I've had enough."},
    {"id": "SC06", "facet": "SelfControl", "key": -1, "domain": "everyday",
     "text": "Reach for my phone the moment I feel a little bored."},
    {"id": "SC07", "facet": "SelfControl", "key": -1, "domain": "work",
     "text": "Start things in a burst of energy and drop them when it fades."},
    {"id": "SC08", "facet": "SelfControl", "key": -1, "domain": "money",
     "text": "End up buying things I never planned to buy."},
    {"id": "SC09", "facet": "SelfControl", "key": -1, "domain": "conflict",
     "text": "Say the first sharp thing that comes to mind in an argument."},
    {"id": "SC10", "facet": "SelfControl", "key": -1, "domain": "leisure",
     "text": "Stay up far later than I meant to because one more episode is tempting."},

    # ---------------- Trust (high = assumes good intentions) ----------------
    {"id": "TR01", "facet": "Trust", "key": +1, "domain": "everyday",
     "text": "Take a stranger's directions at face value and just follow them."},
    {"id": "TR02", "facet": "Trust", "key": +1, "domain": "work",
     "text": "Accept a coworker's apology without reading too much into it."},
    {"id": "TR03", "facet": "Trust", "key": +1, "domain": "social",
     "text": "Lend a friend a small sum without a second thought about getting it back."},
    {"id": "TR04", "facet": "Trust", "key": +1, "domain": "social",
     "text": "Believe a new acquaintance's reason for running late."},
    {"id": "TR05", "facet": "Trust", "key": +1, "domain": "everyday",
     "text": "Leave a delivery on faith that it'll still be there when I get home."},
    {"id": "TR06", "facet": "Trust", "key": -1, "domain": "money",
     "text": "Read a salesperson's friendliness as the start of a pitch."},
    {"id": "TR07", "facet": "Trust", "key": -1, "domain": "social",
     "text": "Wonder what someone really wants when they go out of their way for me."},
    {"id": "TR08", "facet": "Trust", "key": -1, "domain": "social",
     "text": "Quietly check a friend's story before I take it as true."},
    {"id": "TR09", "facet": "Trust", "key": -1, "domain": "money",
     "text": "Assume there's a catch when a deal looks unusually generous."},
    {"id": "TR10", "facet": "Trust", "key": -1, "domain": "online",
     "text": "Figure people online are usually not quite who they claim to be."},

    # ---------------- Sociability (high = seeks company, energized by people) ----------------
    {"id": "SO01", "facet": "Sociability", "key": +1, "domain": "social",
     "text": "Feel recharged after a long evening out with a group."},
    {"id": "SO02", "facet": "Sociability", "key": +1, "domain": "everyday",
     "text": "Start conversations with people waiting in line near me."},
    {"id": "SO03", "facet": "Sociability", "key": +1, "domain": "leisure",
     "text": "Fill a free weekend by making plans with other people."},
    {"id": "SO04", "facet": "Sociability", "key": +1, "domain": "social",
     "text": "Drift toward the busiest, most crowded room at a gathering."},
    {"id": "SO05", "facet": "Sociability", "key": +1, "domain": "work",
     "text": "Take the lunch table where the most people are sitting."},
    {"id": "SO06", "facet": "Sociability", "key": -1, "domain": "leisure",
     "text": "Look forward to an evening with no plans and nobody around."},
    {"id": "SO07", "facet": "Sociability", "key": -1, "domain": "social",
     "text": "Feel worn out after a few hours in a big group."},
    {"id": "SO08", "facet": "Sociability", "key": -1, "domain": "everyday",
     "text": "Pick the quiet corner table over the lively one."},
    {"id": "SO09", "facet": "Sociability", "key": -1, "domain": "leisure",
     "text": "Reach for a solo hobby before a group one."},
    {"id": "SO10", "facet": "Sociability", "key": -1, "domain": "social",
     "text": "Need a stretch of time alone to recover after a social event."},

    # ---------------- Conformity (high = follows norms, fits in) ----------------
    {"id": "CO01", "facet": "Conformity", "key": +1, "domain": "social",
     "text": "Dress to match what the people around me are wearing."},
    {"id": "CO02", "facet": "Conformity", "key": +1, "domain": "social",
     "text": "Go along with the group's pick of place even when it's not mine."},
    {"id": "CO03", "facet": "Conformity", "key": +1, "domain": "everyday",
     "text": "Follow a rule even when no one would notice if I skipped it."},
    {"id": "CO04", "facet": "Conformity", "key": +1, "domain": "social",
     "text": "Keep an opinion to myself when it would clash with everyone else's."},
    {"id": "CO05", "facet": "Conformity", "key": +1, "domain": "work",
     "text": "Match the way things are usually done here rather than my own way."},
    {"id": "CO06", "facet": "Conformity", "key": -1, "domain": "everyday",
     "text": "Do things my own way even when it gets a few odd looks."},
    {"id": "CO07", "facet": "Conformity", "key": -1, "domain": "work",
     "text": "Question a rule out loud when it doesn't make sense to me."},
    {"id": "CO08", "facet": "Conformity", "key": -1, "domain": "social",
     "text": "Stay with an unpopular choice while a group leans the other way."},
    {"id": "CO09", "facet": "Conformity", "key": -1, "domain": "social",
     "text": "Feel fine being the odd one out in a crowd."},
    {"id": "CO10", "facet": "Conformity", "key": -1, "domain": "work",
     "text": "Push back when the reason given is 'that's how it's always been done.'"},

    # ---------------- Adaptability (high = adjusts to change, flexible) ----------------
    {"id": "AD01", "facet": "Adaptability", "key": +1, "domain": "everyday",
     "text": "Roll with it when a plan changes at the last minute."},
    {"id": "AD02", "facet": "Adaptability", "key": +1, "domain": "life",
     "text": "Settle into a new routine without much fuss."},
    {"id": "AD03", "facet": "Adaptability", "key": +1, "domain": "work",
     "text": "Find a workaround quickly when something doesn't go as expected."},
    {"id": "AD04", "facet": "Adaptability", "key": +1, "domain": "travel",
     "text": "Feel more curious than uneasy when I'm somewhere unfamiliar."},
    {"id": "AD05", "facet": "Adaptability", "key": +1, "domain": "leisure",
     "text": "Happily switch to a different option when my first choice falls through."},
    {"id": "AD06", "facet": "Adaptability", "key": -1, "domain": "everyday",
     "text": "Get thrown off when my usual routine is disrupted."},
    {"id": "AD07", "facet": "Adaptability", "key": -1, "domain": "life",
     "text": "Need a while to come around to a change I didn't choose."},
    {"id": "AD08", "facet": "Adaptability", "key": -1, "domain": "everyday",
     "text": "Feel tense while plans are still up in the air."},
    {"id": "AD09", "facet": "Adaptability", "key": -1, "domain": "work",
     "text": "Want to know exactly what to expect before I'll start."},
    {"id": "AD10", "facet": "Adaptability", "key": -1, "domain": "leisure",
     "text": "Stick with the familiar option rather than try something new."},
]

FACETS = ["SelfControl", "Trust", "Sociability", "Conformity", "Adaptability"]


def items_dict():
    """{id: {'text':...}} for administer_spi."""
    return {it["id"]: {"text": it["text"]} for it in ITEMS}


def by_facet():
    out = {f: [] for f in FACETS}
    for it in ITEMS:
        out[it["facet"]].append(it)
    return out
