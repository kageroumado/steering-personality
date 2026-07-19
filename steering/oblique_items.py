"""
oblique_items.py — the PRODUCT bank: indirect "doesn't-read-as-psychometric" questions.

The end product is a human survey whose questions sound like a fun personality quiz, not a
psychometric inventory — concrete, projective, behavioral scenarios that never name the trait —
yet whose answers invert (through an LLM-calibrated loading matrix) to a 27D SPI reading.

These items are the test of the product's core premise: can OBLIQUE questions be calibrated on
the model at all? `calibrate_oblique.py` administers each under single-facet steering of all 27
facets and measures the loading L[item][facet] = slope(answer-delta ~ θ_facet). A usable oblique
item is one that loads strongly and cleanly (single-facet-dominant) on some facet despite never
mentioning it.

Design notes:
  * Behavioral / dispositional framing ("what would you do"), not self-description — per the iter-8
    SelfControl finding that I-DO items carry trait signal that I-SAY items don't.
  * Desirability-matched poles (neither is the obviously "better person").
  * `guess` = my a-priori hypothesis for the dominant facet; calibration tests it (the field is a
    label only, never used in scoring). Obliqueness deliberately varies (donuts→SelfControl is
    near-transparent; blank-notebook→Creativity is quite indirect) to map the obliqueness/signal
    trade-off.  hi = the pole hypothesised HIGH on `guess`.
"""
from __future__ import annotations

OBLIQUE = [
    {"id": "OB01", "guess": "SensationSeeking", "stem": "A friend offers you a spontaneous road trip leaving within the hour, no plan.", "hi": " You're already grabbing your jacket.", "lo": " You'd need to know where and how long first."},
    {"id": "OB02", "guess": "Order", "stem": "Packing for a week away, your suitcase is", "hi": " everything folded into labeled cubes.", "lo": " stuffed in and zipped fast."},
    {"id": "OB03", "guess": "Sociability", "stem": "After a long week, the restorative thing is", "hi": " a night out with everyone.", "lo": " a quiet night on your own."},
    {"id": "OB04", "guess": "Creativity", "stem": "Handed a blank notebook, the first page becomes", "hi": " something strange and your own.", "lo": " a neat, ordered list."},
    {"id": "OB05", "guess": "Compassion", "stem": "A stranger's grocery bag splits open in the parking lot.", "hi": " You're already crouching to help.", "lo": " You figure they've got it."},
    {"id": "OB06", "guess": "Anxiety", "stem": "The plane hits a patch of hard turbulence.", "hi": " Your hand finds the armrest and your mind races ahead.", "lo": " You barely look up from your book."},
    {"id": "OB07", "guess": "Irritability", "stem": "Someone steals the parking spot you were clearly waiting for.", "hi": " It sits with you for the next hour.", "lo": " You shrug and find another."},
    {"id": "OB08", "guess": "AttentionSeeking", "stem": "At karaoke, when the mic comes around, you're", "hi": " up there before they finish asking.", "lo": " cheering hard for everyone else."},
    {"id": "OB09", "guess": "Humor", "stem": "When a meeting goes completely off the rails, you're", "hi": " the one cracking the joke.", "lo": " waiting quietly for it to refocus."},
    {"id": "OB10", "guess": "Authoritarianism", "stem": "A sign says 'keep off the grass,' but there's a worn shortcut across it.", "hi": " You take the long way on the path.", "lo": " You cut across like everyone else."},
    {"id": "OB11", "guess": "Honesty", "stem": "A cashier hands you ten dollars too much in change.", "hi": " You catch it and hand it back.", "lo": " You notice later and let it go."},
    {"id": "OB12", "guess": "Industry", "stem": "A Saturday with nothing on the calendar, and you", "hi": " make a list and knock things out.", "lo": " let the day go wherever it goes."},
    {"id": "OB13", "guess": "Intellect", "stem": "A puzzle the whole group is stuck on, and you", "hi": " can't put it down till it cracks.", "lo": " are happy to let someone else get it."},
    {"id": "OB14", "guess": "Trust", "stem": "A coworker you barely know asks to borrow your good headphones.", "hi": " Sure — here you go.", "lo": " You'd rather not, honestly."},
    {"id": "OB15", "guess": "ArtAppreciation", "stem": "A free afternoon in a city you've never visited, and you head", "hi": " straight for the art museum.", "lo": " straight for the food and the streets."},
    {"id": "OB16", "guess": "EmotionalStability", "stem": "Your inner weather, most days, is", "hi": " mostly clear and mild.", "lo": " changeable, with some storms."},
    {"id": "OB17", "guess": "SelfControl", "stem": "There's an open box of donuts in the break room all day.", "hi": " You can walk past it just fine.", "lo": " You'll have had two by noon."},
    {"id": "OB18", "guess": "Conservatism", "stem": "A long-standing tradition your family always keeps, and you'd", "hi": " keep it exactly as it is.", "lo": " happily reinvent it."},
    {"id": "OB19", "guess": "Charisma", "stem": "You need a stranger to do you a small favor, and you feel", "hi": " confident you can win them over.", "lo": " you'd rather not have to ask."},
    {"id": "OB20", "guess": "Impulsivity", "stem": "A 'today only' deal pops up on something cool, and it's", "hi": " already in your cart.", "lo": " a tab you close to think it over."},
    {"id": "OB21", "guess": "EmotionalExpressiveness", "stem": "When a film genuinely moves you, the people next to you", "hi": " can tell.", "lo": " would never know."},
    {"id": "OB22", "guess": "SensationSeeking", "stem": "The park's tallest, fastest coaster, and you're", "hi": " front row, hands up.", "lo": " holding everyone's bags."},
    {"id": "OB23", "guess": "Adaptability", "stem": "Your favorite restaurant is unexpectedly closed when you arrive.", "hi": " Cool — let's find somewhere new.", "lo": " That throws off the whole evening."},
    {"id": "OB24", "guess": "Introspection", "stem": "On a long train ride alone, you mostly", "hi": " drift off into your own thoughts.", "lo": " keep busy with something to do."},
    {"id": "OB25", "guess": "Perfectionism", "stem": "A gift you're wrapping comes out a little crooked, and you", "hi": " redo it.", "lo": " call it good — they'll tear it open anyway."},
    {"id": "OB26", "guess": "Conformity", "stem": "Everyone at the table orders the same dish, and you", "hi": " go along with it too.", "lo": " order the thing you actually wanted."},
    {"id": "OB27", "guess": "EasyGoingness", "stem": "Your ideal pace on a vacation is", "hi": " slow mornings, no agenda.", "lo": " up early, see everything."},
    {"id": "OB28", "guess": "Compassion", "stem": "A friend's plan is clearly headed for a flop, and you", "hi": " gently steer them right.", "lo": " let them learn it for themselves."},
    {"id": "OB29", "guess": "Creativity", "stem": "Assembling furniture, the instructions are", "hi": " optional — you wing it your own way.", "lo": " step one, in order, every time."},
    {"id": "OB30", "guess": "Sociability", "stem": "You walk into a party where you know exactly one person. By the end you", "hi": " know ten.", "lo": " have stuck close to the one."},
    {"id": "OB31", "guess": "WellBeing", "stem": "Catching your reflection unexpectedly, the feeling is", "hi": " quietly glad it's you.", "lo": " a small sigh."},
    {"id": "OB32", "guess": "SensationSeeking", "stem": "The menu has one dish you can't pronounce and have never tried.", "hi": " That's the one you order.", "lo": " You go with your usual."},
    {"id": "OB33", "guess": "Industry", "stem": "An email needs a reply you've been dreading. It's", "hi": " handled before lunch.", "lo": " still sitting there on Friday."},
    {"id": "OB34", "guess": "AttentionSeeking", "stem": "A photo's being taken of the group, and you", "hi": " step to the front and center.", "lo": " slot in at the edge."},
    {"id": "OB35", "guess": "Introspection", "stem": "After an argument, the part that stays with you is", "hi": " turning over what it says about you.", "lo": " just wanting to move on from it."},
]


# batch 2 (iter-11): scale the pool toward an overdetermined, well-conditioned S. Authored for
# breadth of SITUATION (work, money, risk, social, aesthetics, daily habit, moral, emotion), not to
# target facets (we can't — guess hit-rate was 1/35); a few lean at the iter-10 weak-recovered facets
# (Irritability, Anxiety, Adaptability, Conservatism) but calibration decides their real loadings.
OBLIQUE2 = [
    {"id": "OB36", "guess": "Creativity", "stem": "Handed the aux cord on a road trip, you play", "hi": " something nobody's heard yet.", "lo": " the songs everyone can sing along to."},
    {"id": "OB37", "guess": "Honesty", "stem": "A long queue, and a gap opens to slip ahead unnoticed.", "hi": " You stay put.", "lo": " You take it."},
    {"id": "OB38", "guess": "AttentionSeeking", "stem": "Your team wins; at the celebration you're", "hi": " loudest in the room.", "lo": " happy in the back."},
    {"id": "OB39", "guess": "Intellect", "stem": "Tonight it's a documentary or a comedy, and you pick", "hi": " the documentary.", "lo": " the comedy."},
    {"id": "OB40", "guess": "Perfectionism", "stem": "You spot a typo in a printed menu, and you", "hi": " can't unsee it.", "lo": " don't care at all."},
    {"id": "OB41", "guess": "Irritability", "stem": "A coworker takes credit for your idea in the meeting.", "hi": " You set the record straight on the spot.", "lo": " You let it slide this time."},
    {"id": "OB42", "guess": "Charisma", "stem": "The group can't decide where to eat, so you", "hi": " just pick and lead.", "lo": " wait for a consensus."},
    {"id": "OB43", "guess": "Industry", "stem": "An unexpected free day appears, and you", "hi": " finally tackle the project you've been meaning to.", "lo": " do gloriously nothing."},
    {"id": "OB44", "guess": "Order", "stem": "Learning a new board game, you'd rather", "hi": " read all the rules first.", "lo": " start playing and figure it out."},
    {"id": "OB45", "guess": "Irritability", "stem": "Someone cuts you off in traffic, and it", "hi": " ruins the next ten minutes.", "lo": " is forgotten instantly."},
    {"id": "OB46", "guess": "Impulsivity", "stem": "At a great buffet, you", "hi": " go back for thirds.", "lo": " stop when you're full."},
    {"id": "OB47", "guess": "Adaptability", "stem": "A friend cancels plans at the last minute, and", "hi": " no problem — you adapt.", "lo": " it stings a bit."},
    {"id": "OB48", "guess": "ArtAppreciation", "stem": "You pass a street musician who's genuinely good, and you", "hi": " stop and listen a while.", "lo": " keep moving, you're busy."},
    {"id": "OB49", "guess": "Order", "stem": "Your handwriting in a notebook tends to be", "hi": " neat and even.", "lo": " a fast scrawl."},
    {"id": "OB50", "guess": "Anxiety", "stem": "After an argument with a friend, you", "hi": " replay it for days.", "lo": " let it go by morning."},
    {"id": "OB51", "guess": "Charisma", "stem": "Asked to give the toast at a wedding, you think", "hi": " gladly — you've got this.", "lo": " please, anyone but me."},
    {"id": "OB52", "guess": "Authoritarianism", "stem": "A rule at work seems pointless, and you", "hi": " follow it anyway.", "lo": " quietly ignore it."},
    {"id": "OB53", "guess": "Honesty", "stem": "You find a banknote on the sidewalk, no one around.", "hi": " You hand it in at a shop nearby.", "lo": " Lucky day — it's yours."},
    {"id": "OB54", "guess": "EasyGoingness", "stem": "Your ideal weekend has", "hi": " nothing locked in.", "lo": " a full plan."},
    {"id": "OB55", "guess": "Adaptability", "stem": "Trying a new sport for the first time, you feel", "hi": " excited.", "lo": " self-conscious."},
    {"id": "OB56", "guess": "Compassion", "stem": "When a friend is venting, you", "hi": " feel their stress as your own.", "lo": " stay calm and just listen."},
    {"id": "OB57", "guess": "AttentionSeeking", "stem": "A surprise party is thrown — for you — and it's", "hi": " the best thing ever.", "lo": " a little overwhelming."},
    {"id": "OB58", "guess": "Conformity", "stem": "You disagree with the group's plan, so you", "hi": " go with the flow.", "lo": " voice it."},
    {"id": "OB59", "guess": "Compassion", "stem": "There's one bite of dessert left to share or finish, and you", "hi": " offer it.", "lo": " take it."},
    {"id": "OB60", "guess": "Order", "stem": "Your inbox has forty-seven unread emails, and that", "hi": " itches until it's zero.", "lo": " is honestly fine."},
    {"id": "OB61", "guess": "Anxiety", "stem": "There's a spider across the room, and you", "hi": " calmly relocate it.", "lo": " call for backup."},
    {"id": "OB62", "guess": "AttentionSeeking", "stem": "An open mic invites anyone up, and you", "hi": " sign up.", "lo": " happily watch."},
    {"id": "OB63", "guess": "Adaptability", "stem": "Told the whole plan just changed, you think", "hi": " fine — what's the new plan.", "lo": " ugh, I'd geared up for the old one."},
    {"id": "OB64", "guess": "Introspection", "stem": "A coworker's been short with you all day, and you", "hi": " wonder what you did.", "lo": " figure it's their problem, not yours."},
    {"id": "OB65", "guess": "SensationSeeking", "stem": "Picking a vacation, you lean toward", "hi": " somewhere brand new.", "lo": " a familiar favorite."},
    {"id": "OB66", "guess": "WellBeing", "stem": "You make a small mistake in front of others, and you", "hi": " laugh it off.", "lo": " stew on it a while."},
    {"id": "OB67", "guess": "Honesty", "stem": "Asked your honest take on a friend's bad new haircut, you", "hi": " tell them kindly but truly.", "lo": " say it looks great."},
    {"id": "OB68", "guess": "Order", "stem": "Your desk right now is", "hi": " clear and ordered.", "lo": " a creative landslide."},
    {"id": "OB69", "guess": "Industry", "stem": "With a deadline a week out, you", "hi": " start today.", "lo": " start the night before."},
    {"id": "OB70", "guess": "Sociability", "stem": "A stranger compliments your jacket, and you", "hi": " light up and chat.", "lo": " give a quick thanks and move on."},
    {"id": "OB71", "guess": "Anxiety", "stem": "When plans are vague and undecided, you feel", "hi": " a low hum of unease.", "lo": " relaxed — it'll sort itself out."},
    {"id": "OB72", "guess": "SensationSeeking", "stem": "A risky bet with good odds is on the table, and you're", "hi": " in.", "lo": " keeping what you have."},
    {"id": "OB73", "guess": "EmotionalExpressiveness", "stem": "A friend shares big news, and your reaction is", "hi": " visible and immediate.", "lo": " warm but understated."},
    {"id": "OB74", "guess": "Compassion", "stem": "You could speak up to fix something unfair, but it's awkward, and you", "hi": " speak up.", "lo": " stay out of it."},
    {"id": "OB75", "guess": "SensationSeeking", "stem": "On the drink menu: the reliable favorite, or the unmarked experimental one. You take", "hi": " the experiment.", "lo": " the favorite."},
]

POOL = OBLIQUE + OBLIQUE2
BANKS = {"b1": OBLIQUE, "b2": OBLIQUE2, "pool": POOL}


def _sanity():
    for name, bank in BANKS.items():
        ids = [it["id"] for it in bank]
        assert len(ids) == len(set(ids)), f"{name}: duplicate ids"
        for it in bank:
            assert it["hi"].startswith(" ") and it["lo"].startswith(" "), it["id"]
    allids = [it["id"] for it in POOL]
    assert len(allids) == len(set(allids)), "pool id collision"
    print(f"OK — b1={len(OBLIQUE)} b2={len(OBLIQUE2)} pool={len(POOL)} oblique items")


if __name__ == "__main__":
    _sanity()
