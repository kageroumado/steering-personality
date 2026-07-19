"""
facets27.py — single source of truth for the FULL SPI-27 facet battery (breadth 5→27, D13).

Two artifacts per facet, authored from the actual SPI/IPIP item content
(`spi_data.json` keys) so the steering contrast and the forced-choice readout measure the
same construct the 135-item instrument scores:

  * CONTRAST  — parallel first-person high-pole / low-pole statements (3 phrasings each)
                for `pca_diff` steering-vector extraction (`extract27.py`).
  * FC_ITEMS  — pole-contrast forced-choice items (shared first-person stem + desirability-
                matched hi/lo completions, 5 per facet) for the iter-4 readout. `hi` = HIGH
                pole of the named facet; Delta = lp(hi) - lp(lo) increases with the trait.

The 5 facets validated through iter-6 (SelfControl, Trust, Sociability, Conformity,
Adaptability) are carried over VERBATIM from extract_vectors.py / forced_choice.py so nothing
regresses and the breadth run is a strict superset.

Sign convention: `pca_diff` returns an arbitrary-sign principal direction; the per-facet sign
(which way is "more of the trait") is determined empirically by the Tier-0 viability scan
(`tier0_27.py`), exactly as it was hand-discovered for the original 5 — not hardcoded here.
"""
from __future__ import annotations

# ----------------------------------------------------------------------------------------
# Contrast pairs for steering-vector extraction. pos = high pole, neg = low pole.
# ----------------------------------------------------------------------------------------
FACETS = {
    # ===== Agreeableness facets =====
    "Compassion": {
        "pos": ["I am deeply moved by other people's suffering.",
                "I put other people's needs ahead of my own.",
                "I feel a strong pull to comfort anyone who is hurting."],
        "neg": ["I am unmoved by other people's suffering.",
                "I put my own needs ahead of other people's.",
                "I feel little pull to comfort people who are hurting."],
    },
    "Trust": {  # carried verbatim from extract_vectors.py
        "pos": ["I trust other people.",
                "I assume people generally have good intentions toward me.",
                "I find it easy to believe what others tell me."],
        "neg": ["I distrust other people.",
                "I assume people generally have hidden or bad intentions toward me.",
                "I find it hard to believe what others tell me."],
    },
    "Honesty": {
        "pos": ["I always tell the truth, even when it costs me.",
                "I cannot imagine lying or cheating to get ahead.",
                "I refuse to use other people for my own ends."],
        "neg": ["I bend the truth whenever it suits me.",
                "I will lie or cheat to get ahead.",
                "I use other people for my own ends."],
    },
    # ===== Conscientiousness facets =====
    "Conservatism": {
        "pos": ["I hold traditional religious and political values.",
                "I vote for conservative candidates and respect tradition.",
                "I believe in one true religion and stand for the national anthem."],
        "neg": ["I reject traditional religious and political values.",
                "I vote for liberal candidates and question tradition.",
                "I am not religious and feel no pull toward patriotic ritual."],
    },
    "Authoritarianism": {
        "pos": ["I respect authority and believe the rules should be obeyed.",
                "I think laws should be strictly enforced and we should be tough on crime.",
                "I follow the rules and expect others to as well."],
        "neg": ["I distrust authority and bristle at being told what to do.",
                "I think strict law-and-order goes too far.",
                "I rebel against rules I did not agree to."],
    },
    "EasyGoingness": {
        "pos": ["I like to take life at a slow, leisurely pace.",
                "I am happy to hang around doing nothing.",
                "I let things proceed at their own pace without rushing."],
        "neg": ["I like to keep a fast, busy pace.",
                "I am restless when there is nothing to do.",
                "I push to make things happen rather than waiting."],
    },
    "Perfectionism": {
        "pos": ["I keep working until every detail is perfect.",
                "I set very high standards and dislike imperfect work.",
                "I am hard to satisfy and want everything just right."],
        "neg": ["I am fine stopping when something is good enough.",
                "I set relaxed standards and tolerate imperfect work.",
                "I am easily satisfied and do not fuss over details."],
    },
    "Order": {
        "pos": ["I keep everything tidy and in its proper place.",
                "I like order and put things back where they belong.",
                "I am bothered by mess and disorder."],
        "neg": ["I leave things messy and out of place.",
                "I dislike fussing over order and let things pile up.",
                "I am not bothered by mess and disorder."],
    },
    "Industry": {
        "pos": ["I start my tasks right away and work hard.",
                "I dive into work without needing a push.",
                "I take my duties seriously and never neglect them."],
        "neg": ["I put off my tasks and find it hard to get going.",
                "I need a push to get started on work.",
                "I neglect my duties when I can."],
    },
    "Impulsivity": {
        "pos": ["I act on the spur of the moment without thinking.",
                "I jump into things and make rash decisions.",
                "I follow my impulses the instant I feel them."],
        "neg": ["I stop and think things over before doing anything.",
                "I plan carefully instead of acting on impulse.",
                "I resist sudden urges and decide deliberately."],
    },
    "SelfControl": {  # carried verbatim
        "pos": ["I have strong self-control.",
                "I resist my impulses and think carefully before acting.",
                "I stay disciplined and rarely act on a whim."],
        "neg": ["I have weak self-control.",
                "I give in to my impulses and act before thinking.",
                "I am undisciplined and often act on a whim."],
    },
    # ===== Neuroticism facets =====
    "EmotionalStability": {
        "pos": ["I stay even-keeled and am not easily affected by my emotions.",
                "I experience very few emotional highs and lows.",
                "My moods stay steady from day to day."],
        "neg": ["I get overwhelmed by my emotions.",
                "I experience my emotions very intensely.",
                "My moods swing widely from day to day."],
    },
    "Anxiety": {
        "pos": ["I am a worrier who fears the worst.",
                "I am a nervous person who panics easily.",
                "I worry about things constantly."],
        "neg": ["I stay calm and rarely worry.",
                "I am relaxed and hard to rattle.",
                "I expect things to turn out fine."],
    },
    "Irritability": {
        "pos": ["I get angry easily and lose my temper.",
                "I am quickly annoyed and short-tempered.",
                "Small things set me off."],
        "neg": ["I seldom get mad and rarely get irritated.",
                "I stay patient and am not easily annoyed.",
                "It takes a lot to set me off."],
    },
    "WellBeing": {
        "pos": ["I am happy with my life and love living.",
                "I feel comfortable with myself and at ease.",
                "I wake up glad to be alive."],
        "neg": ["I dislike myself and feel a sense of hopelessness.",
                "I feel worthless and ill at ease with myself.",
                "I find little to enjoy in life."],
    },
    "EmotionalExpressiveness": {
        "pos": ["I am open about my feelings and express them easily.",
                "I show my sadness and let others see how I feel.",
                "I reveal what is going on inside me."],
        "neg": ["I have difficulty expressing my feelings.",
                "I hide my sadness and keep my feelings to myself.",
                "I reveal little about what is going on inside me."],
    },
    # ===== Extraversion facets =====
    "Sociability": {  # carried verbatim
        "pos": ["I love being around other people.",
                "I seek out company and feel energized in a crowd.",
                "I am outgoing and enjoy meeting new people."],
        "neg": ["I prefer to be alone.",
                "I avoid company and feel drained in a crowd.",
                "I am withdrawn and keep to myself."],
    },
    "Adaptability": {  # carried verbatim
        "pos": ["I adapt easily to change.",
                "When plans change suddenly, I adjust smoothly and stay relaxed.",
                "I am flexible and comfortable with new and unfamiliar situations."],
        "neg": ["I struggle to adapt to change.",
                "When plans change suddenly, I get thrown off and stressed.",
                "I am rigid and uncomfortable with new and unfamiliar situations."],
    },
    "Charisma": {
        "pos": ["I have a natural talent for influencing people.",
                "I am skilled at handling social situations and approaching others.",
                "I am good at making impromptu speeches."],
        "neg": ["I lack any talent for influencing people.",
                "I find it difficult to approach others.",
                "I freeze up when I have to speak on the spot."],
    },
    "Humor": {
        "pos": ["I laugh a lot and amuse my friends.",
                "I laugh my way through life and joke around.",
                "I find something funny in almost everything."],
        "neg": ["I seldom laugh or joke around.",
                "I take life seriously and rarely find things funny.",
                "I am the serious one who does not crack jokes."],
    },
    "AttentionSeeking": {
        "pos": ["I make myself the center of attention.",
                "I like to attract attention and put on a show.",
                "I love it when all eyes are on me."],
        "neg": ["I dislike being the center of attention.",
                "I hate having all eyes on me.",
                "I stay in the background and avoid the spotlight."],
    },
    "SensationSeeking": {
        "pos": ["I love dangerous situations and seek out adventure.",
                "I take risks and do crazy things.",
                "I chase the thrill of danger."],
        "neg": ["I avoid dangerous situations and prefer safety.",
                "I steer clear of risks and play it safe.",
                "I have no appetite for thrills or danger."],
    },
    "Conformity": {  # carried verbatim
        "pos": ["I go along with what the group expects.",
                "I follow the rules and social norms around me.",
                "I prefer to fit in rather than stand out."],
        "neg": ["I go against what the group expects.",
                "I ignore the rules and social norms around me.",
                "I prefer to stand out rather than fit in."],
    },
    # ===== Openness facets =====
    "Introspection": {
        "pos": ["I love to reflect on things and examine my own life.",
                "I spend a lot of time trying to understand myself.",
                "I like to get lost in thought."],
        "neg": ["I rarely reflect on things or examine my life.",
                "I spend little time trying to understand myself.",
                "I avoid getting lost in thought."],
    },
    "ArtAppreciation": {
        "pos": ["I believe in the importance of art and love museums.",
                "I enjoy poetry and have read the great literary classics.",
                "I am moved by art and beauty."],
        "neg": ["I do not like art and avoid museums.",
                "I have no interest in poetry or literary classics.",
                "I am unmoved by art and beauty."],
    },
    "Creativity": {
        "pos": ["I am full of new and original ideas.",
                "I am an original thinker with a vivid imagination.",
                "I love to think up new ways of doing things."],
        "neg": ["I rarely come up with new ideas.",
                "I am a conventional thinker with little imagination.",
                "I prefer doing things the established way."],
    },
    "Intellect": {
        "pos": ["I am quick to understand things and catch on fast.",
                "I can handle a lot of information and think quickly.",
                "I grasp new ideas almost instantly."],
        "neg": ["I am slow to understand things and learn slowly.",
                "I get overwhelmed by a lot of information.",
                "I need things explained several times."],
    },
}

# ----------------------------------------------------------------------------------------
# Forced-choice pole-contrast items. hi = HIGH pole; Delta = lp(hi) - lp(lo). 5 per facet.
# Desirability-matched so neither completion is the obviously "better person".
# ----------------------------------------------------------------------------------------
FC_ITEMS = [
    # ---- Compassion ----
    {"id": "CM1", "facet": "Compassion", "stem": "When a coworker is clearly upset, I", "hi": " stop what I'm doing to check on them.", "lo": " give them space and stay out of it."},
    {"id": "CM2", "facet": "Compassion", "stem": "Seeing a stranger struggling with their bags, I", "hi": " step in to help.", "lo": " let them manage on their own."},
    {"id": "CM3", "facet": "Compassion", "stem": "When a friend describes a hard day, I", "hi": " feel it along with them.", "lo": " hear them out but stay detached."},
    {"id": "CM4", "facet": "Compassion", "stem": "Faced with someone worse off than me, I feel", "hi": " a pull to do something.", "lo": " that it isn't mine to fix."},
    {"id": "CM5", "facet": "Compassion", "stem": "When I make a plan, I think first about", "hi": " how it affects everyone involved.", "lo": " what works best for me."},

    # ---- Trust ----
    {"id": "TR1", "facet": "Trust", "stem": "When a stranger gives me directions, I", "hi": " just follow them.", "lo": " double-check them myself."},
    {"id": "TR2", "facet": "Trust", "stem": "When a deal looks unusually generous, I", "hi": " take it at face value.", "lo": " assume there's a catch."},
    {"id": "TR3", "facet": "Trust", "stem": "When someone goes out of their way for me, I", "hi": " take the kindness as it is.", "lo": " wonder what they really want."},
    {"id": "TR4", "facet": "Trust", "stem": "When a friend explains why they were late, I", "hi": " believe them without a second thought.", "lo": " quietly check whether it's true."},
    {"id": "TR5", "facet": "Trust", "stem": "Meeting someone new online, I figure they're", "hi": " probably who they say they are.", "lo": " probably not quite who they claim."},

    # ---- Honesty ----
    {"id": "HO1", "facet": "Honesty", "stem": "When the truth would cost me, I", "hi": " tell it anyway.", "lo": " shade it in my favor."},
    {"id": "HO2", "facet": "Honesty", "stem": "To get ahead of a rival, I", "hi": " keep it strictly fair.", "lo": " do whatever it takes."},
    {"id": "HO3", "facet": "Honesty", "stem": "Given a chance to bend a rule unseen, I", "hi": " leave it alone.", "lo": " take the opening."},
    {"id": "HO4", "facet": "Honesty", "stem": "When someone's mistake happens to benefit me, I", "hi": " point it out.", "lo": " quietly let it ride."},
    {"id": "HO5", "facet": "Honesty", "stem": "In dealing with people, I", "hi": " play it straight.", "lo": " work them to my advantage."},

    # ---- Conservatism ----
    {"id": "CV1", "facet": "Conservatism", "stem": "When it comes to long-standing tradition, I", "hi": " honor the old ways.", "lo": " question them."},
    {"id": "CV2", "facet": "Conservatism", "stem": "On matters of religion, I", "hi": " hold a firm faith.", "lo": " keep my distance from it."},
    {"id": "CV3", "facet": "Conservatism", "stem": "At the polls, I lean toward", "hi": " the conservative candidate.", "lo": " the liberal candidate."},
    {"id": "CV4", "facet": "Conservatism", "stem": "When the national anthem plays, I", "hi": " stand with feeling.", "lo": " stay seated, unmoved."},
    {"id": "CV5", "facet": "Conservatism", "stem": "Toward customs handed down over generations, I feel", "hi": " loyalty.", "lo": " indifference."},

    # ---- Authoritarianism ----
    {"id": "AU1", "facet": "Authoritarianism", "stem": "When someone in charge gives an order, I", "hi": " respect it.", "lo": " bristle at it."},
    {"id": "AU2", "facet": "Authoritarianism", "stem": "On enforcing the law, I think we should be", "hi": " strict and firm.", "lo": " lenient and forgiving."},
    {"id": "AU3", "facet": "Authoritarianism", "stem": "Faced with a rule I dislike, I", "hi": " follow it anyway.", "lo": " push back against it."},
    {"id": "AU4", "facet": "Authoritarianism", "stem": "Toward authority in general, I tend to", "hi": " defer.", "lo": " rebel."},
    {"id": "AU5", "facet": "Authoritarianism", "stem": "On dealing with crime, I believe in being", "hi": " tough.", "lo": " understanding."},

    # ---- EasyGoingness ----
    {"id": "EG1", "facet": "EasyGoingness", "stem": "With an open afternoon, I", "hi": " let it drift.", "lo": " pack it with things to do."},
    {"id": "EG2", "facet": "EasyGoingness", "stem": "My pace through the day is", "hi": " slow and relaxed.", "lo": " quick and driven."},
    {"id": "EG3", "facet": "EasyGoingness", "stem": "When there's nothing pressing, I", "hi": " happily do nothing.", "lo": " find something productive."},
    {"id": "EG4", "facet": "EasyGoingness", "stem": "With a deadline still far off, I", "hi": " let it come at its own pace.", "lo": " get on it immediately."},
    {"id": "EG5", "facet": "EasyGoingness", "stem": "My ideal way of living is", "hi": " leisurely.", "lo": " busy and full."},

    # ---- Perfectionism ----
    {"id": "PF1", "facet": "Perfectionism", "stem": "When a piece of work is good enough, I", "hi": " keep polishing it.", "lo": " call it done."},
    {"id": "PF2", "facet": "Perfectionism", "stem": "About small flaws others won't notice, I", "hi": " can't let them go.", "lo": " let them slide."},
    {"id": "PF3", "facet": "Perfectionism", "stem": "The standards I hold are", "hi": " exacting.", "lo": " easygoing."},
    {"id": "PF4", "facet": "Perfectionism", "stem": "When some detail is left unhandled, I feel", "hi": " unsettled.", "lo": " fine about it."},
    {"id": "PF5", "facet": "Perfectionism", "stem": "With my own output, I am", "hi": " hard to satisfy.", "lo": " readily satisfied."},

    # ---- Order ----
    {"id": "OR1", "facet": "Order", "stem": "My room tends to be", "hi": " tidy and arranged.", "lo": " a comfortable mess."},
    {"id": "OR2", "facet": "Order", "stem": "After I use something, I", "hi": " put it right back.", "lo": " leave it where it lands."},
    {"id": "OR3", "facet": "Order", "stem": "Surrounded by clutter, I feel", "hi": " the itch to fix it.", "lo": " perfectly at ease."},
    {"id": "OR4", "facet": "Order", "stem": "I keep my belongings", "hi": " in fixed places.", "lo": " wherever they end up."},
    {"id": "OR5", "facet": "Order", "stem": "Toward disorder, I am", "hi": " bothered.", "lo": " unbothered."},

    # ---- Industry ----
    {"id": "IN1", "facet": "Industry", "stem": "Facing a task I've been handed, I", "hi": " start it right away.", "lo": " put it off a while."},
    {"id": "IN2", "facet": "Industry", "stem": "To get going on work, I", "hi": " need no push.", "lo": " need a push."},
    {"id": "IN3", "facet": "Industry", "stem": "When my duties pile up, I", "hi": " work hard through them.", "lo": " let some slide."},
    {"id": "IN4", "facet": "Industry", "stem": "At the start of a chore, I", "hi": " dig straight in.", "lo": " find a reason to wait."},
    {"id": "IN5", "facet": "Industry", "stem": "When it comes to hard work, I", "hi": " throw myself into it.", "lo": " do the minimum."},

    # ---- Impulsivity ----
    {"id": "IM1", "facet": "Impulsivity", "stem": "When an idea grabs me, I", "hi": " act on it right away.", "lo": " sit with it first."},
    {"id": "IM2", "facet": "Impulsivity", "stem": "In making a decision, I tend to", "hi": " go with the spur of the moment.", "lo": " think it over."},
    {"id": "IM3", "facet": "Impulsivity", "stem": "When I feel a sudden urge, I", "hi": " follow it.", "lo": " rein it in."},
    {"id": "IM4", "facet": "Impulsivity", "stem": "Before doing something new, I", "hi": " just jump in.", "lo": " stop and weigh it."},
    {"id": "IM5", "facet": "Impulsivity", "stem": "My choices are usually", "hi": " made on the fly.", "lo": " made after deliberation."},

    # ---- SelfControl ----
    {"id": "SC1", "facet": "SelfControl", "stem": "When I get the urge to buy something I hadn't planned for, I", "hi": " set it down and walk away.", "lo": " go ahead and get it."},
    {"id": "SC2", "facet": "SelfControl", "stem": "Once I sit down to work, the dull part", "hi": " gets done before I take a break.", "lo": " can wait until I feel like it."},
    {"id": "SC3", "facet": "SelfControl", "stem": "When a message really irritates me, I", "hi": " wait until I've cooled off to reply.", "lo": " fire back right away."},
    {"id": "SC4", "facet": "SelfControl", "stem": "Late at night, when one more episode is tempting, I", "hi": " turn it off and go to bed.", "lo": " keep watching anyway."},
    {"id": "SC5", "facet": "SelfControl", "stem": "With a second helping there and my hunger gone, I", "hi": " leave it on the table.", "lo": " help myself to more."},

    # ---- EmotionalStability ----
    {"id": "ES1", "facet": "EmotionalStability", "stem": "When strong feelings hit, I", "hi": " stay steady.", "lo": " get swept up."},
    {"id": "ES2", "facet": "EmotionalStability", "stem": "My moods across a week are", "hi": " pretty level.", "lo": " up and down."},
    {"id": "ES3", "facet": "EmotionalStability", "stem": "Under emotional strain, I", "hi": " keep an even keel.", "lo": " get overwhelmed."},
    {"id": "ES4", "facet": "EmotionalStability", "stem": "I experience my emotions", "hi": " mildly.", "lo": " intensely."},
    {"id": "ES5", "facet": "EmotionalStability", "stem": "From one day to the next, my mood", "hi": " holds steady.", "lo": " shifts a lot."},

    # ---- Anxiety ----
    {"id": "AX1", "facet": "Anxiety", "stem": "Before something uncertain, I", "hi": " brace for the worst.", "lo": " expect it to be fine."},
    {"id": "AX2", "facet": "Anxiety", "stem": "Most days, my mind is", "hi": " busy with worries.", "lo": " free of worry."},
    {"id": "AX3", "facet": "Anxiety", "stem": "When a small problem comes up, I", "hi": " start to panic.", "lo": " stay calm."},
    {"id": "AX4", "facet": "Anxiety", "stem": "As a person, I'd call myself", "hi": " nervous.", "lo": " unflappable."},
    {"id": "AX5", "facet": "Anxiety", "stem": "Facing the unknown, I feel", "hi": " on edge.", "lo": " at ease."},

    # ---- Irritability ----
    {"id": "IR1", "facet": "Irritability", "stem": "When something annoys me, I", "hi": " flare up fast.", "lo": " let it pass."},
    {"id": "IR2", "facet": "Irritability", "stem": "My temper is", "hi": " quick to rise.", "lo": " slow to rise."},
    {"id": "IR3", "facet": "Irritability", "stem": "At a minor frustration, I", "hi": " get mad.", "lo": " stay unbothered."},
    {"id": "IR4", "facet": "Irritability", "stem": "Small irritations tend to", "hi": " set me off.", "lo": " roll off me."},
    {"id": "IR5", "facet": "Irritability", "stem": "When things go wrong, I", "hi": " snap.", "lo": " stay even."},

    # ---- WellBeing ----
    {"id": "WB1", "facet": "WellBeing", "stem": "When I think about my life, I feel", "hi": " glad of it.", "lo": " heavy about it."},
    {"id": "WB2", "facet": "WellBeing", "stem": "Toward myself, I feel", "hi": " at ease.", "lo": " dissatisfied."},
    {"id": "WB3", "facet": "WellBeing", "stem": "On waking in the morning, I", "hi": " look forward to the day.", "lo": " dread it a little."},
    {"id": "WB4", "facet": "WellBeing", "stem": "About who I am, I feel", "hi": " comfortable.", "lo": " uneasy."},
    {"id": "WB5", "facet": "WellBeing", "stem": "My overall sense of life is that", "hi": " I love it.", "lo": " it weighs on me."},

    # ---- EmotionalExpressiveness ----
    {"id": "EE1", "facet": "EmotionalExpressiveness", "stem": "When I'm feeling something strongly, I", "hi": " let it show.", "lo": " keep it hidden."},
    {"id": "EE2", "facet": "EmotionalExpressiveness", "stem": "About my inner life, I", "hi": " open up readily.", "lo": " stay guarded."},
    {"id": "EE3", "facet": "EmotionalExpressiveness", "stem": "When I'm sad, I", "hi": " show it.", "lo": " mask it."},
    {"id": "EE4", "facet": "EmotionalExpressiveness", "stem": "Expressing how I feel comes to me", "hi": " easily.", "lo": " with difficulty."},
    {"id": "EE5", "facet": "EmotionalExpressiveness", "stem": "To the people around me, I am", "hi": " an open book.", "lo": " hard to read."},

    # ---- Sociability ----
    {"id": "SO1", "facet": "Sociability", "stem": "With a free evening ahead, I'd rather", "hi": " make plans with a group.", "lo": " spend it on my own."},
    {"id": "SO2", "facet": "Sociability", "stem": "At a gathering, I drift toward", "hi": " the busiest, most crowded room.", "lo": " a quiet corner away from the noise."},
    {"id": "SO3", "facet": "Sociability", "stem": "After a long evening out with people, I feel", "hi": " recharged.", "lo": " worn out."},
    {"id": "SO4", "facet": "Sociability", "stem": "Waiting in line near a stranger, I", "hi": " strike up a conversation.", "lo": " keep to myself."},
    {"id": "SO5", "facet": "Sociability", "stem": "Faced with a free weekend, my first instinct is to", "hi": " fill it with people.", "lo": " keep it quiet and solo."},

    # ---- Adaptability ----
    {"id": "AD1", "facet": "Adaptability", "stem": "When a plan changes at the last minute, I", "hi": " roll with it.", "lo": " get thrown off."},
    {"id": "AD2", "facet": "Adaptability", "stem": "Dropped somewhere unfamiliar, I feel", "hi": " curious.", "lo": " uneasy."},
    {"id": "AD3", "facet": "Adaptability", "stem": "When my usual routine gets disrupted, I", "hi": " adjust without much fuss.", "lo": " need time to recover."},
    {"id": "AD4", "facet": "Adaptability", "stem": "When my first choice falls through, I", "hi": " switch to another easily.", "lo": " stay stuck on what I'd wanted."},
    {"id": "AD5", "facet": "Adaptability", "stem": "Before starting something new, I", "hi": " just dive in.", "lo": " want to know exactly what to expect."},

    # ---- Charisma ----
    {"id": "CH1", "facet": "Charisma", "stem": "In a roomful of people, I", "hi": " win them over easily.", "lo": " struggle to connect."},
    {"id": "CH2", "facet": "Charisma", "stem": "Approaching someone new, I feel", "hi": " smooth.", "lo": " awkward."},
    {"id": "CH3", "facet": "Charisma", "stem": "Asked to speak on the spot, I", "hi": " carry it off.", "lo": " freeze up."},
    {"id": "CH4", "facet": "Charisma", "stem": "When I want to sway a group, I", "hi": " know just how.", "lo": " fall flat."},
    {"id": "CH5", "facet": "Charisma", "stem": "In social situations, I am", "hi": " in my element.", "lo": " out of my depth."},

    # ---- Humor ----
    {"id": "HU1", "facet": "Humor", "stem": "Through an ordinary day, I", "hi": " laugh a lot.", "lo": " rarely laugh."},
    {"id": "HU2", "facet": "Humor", "stem": "Around my friends, I", "hi": " keep them amused.", "lo": " stay the quiet one."},
    {"id": "HU3", "facet": "Humor", "stem": "When something goes sideways, I", "hi": " find it funny.", "lo": " take it straight."},
    {"id": "HU4", "facet": "Humor", "stem": "My way through life is", "hi": " full of laughs.", "lo": " mostly serious."},
    {"id": "HU5", "facet": "Humor", "stem": "Joking around comes to me", "hi": " naturally.", "lo": " seldom."},

    # ---- AttentionSeeking ----
    {"id": "AT1", "facet": "AttentionSeeking", "stem": "When all eyes turn to me, I", "hi": " love it.", "lo": " shrink from it."},
    {"id": "AT2", "facet": "AttentionSeeking", "stem": "At a gathering, I", "hi": " make myself the center.", "lo": " stay on the edges."},
    {"id": "AT3", "facet": "AttentionSeeking", "stem": "Given a chance to stand out, I", "hi": " take it.", "lo": " pass it up."},
    {"id": "AT4", "facet": "AttentionSeeking", "stem": "To impress people, I'll", "hi": " put on a show.", "lo": " hold back."},
    {"id": "AT5", "facet": "AttentionSeeking", "stem": "Being noticed is something I", "hi": " seek out.", "lo": " avoid."},

    # ---- SensationSeeking ----
    {"id": "SS1", "facet": "SensationSeeking", "stem": "Offered something risky and thrilling, I", "hi": " jump at it.", "lo": " turn it down."},
    {"id": "SS2", "facet": "SensationSeeking", "stem": "Toward danger, I feel", "hi": " drawn.", "lo": " wary."},
    {"id": "SS3", "facet": "SensationSeeking", "stem": "On a dare to do something wild, I", "hi": " go for it.", "lo": " sit it out."},
    {"id": "SS4", "facet": "SensationSeeking", "stem": "Between a safe path and a thrilling one, I pick", "hi": " the thrill.", "lo": " the safe one."},
    {"id": "SS5", "facet": "SensationSeeking", "stem": "Adventure is something I", "hi": " seek.", "lo": " avoid."},

    # ---- Conformity ----
    {"id": "CO1", "facet": "Conformity", "stem": "When the group picks a place that isn't my choice, I", "hi": " go along with it.", "lo": " push for my own pick."},
    {"id": "CO2", "facet": "Conformity", "stem": "When my opinion would clash with everyone else's, I", "hi": " keep it to myself.", "lo": " say it anyway."},
    {"id": "CO3", "facet": "Conformity", "stem": "Faced with a rule that makes little sense, I", "hi": " follow it anyway.", "lo": " question it out loud."},
    {"id": "CO4", "facet": "Conformity", "stem": "Getting dressed for an event, I tend to", "hi": " match what others will wear.", "lo": " wear what I like regardless."},
    {"id": "CO5", "facet": "Conformity", "stem": "When the group leans one way, I", "hi": " fall in with them.", "lo": " hold my own line."},

    # ---- Introspection ----
    {"id": "IT1", "facet": "Introspection", "stem": "With a quiet moment, I", "hi": " turn it inward and reflect.", "lo": " look for something to do."},
    {"id": "IT2", "facet": "Introspection", "stem": "About my own motives, I", "hi": " examine them closely.", "lo": " don't dwell on them."},
    {"id": "IT3", "facet": "Introspection", "stem": "My own inner life is something I", "hi": " explore.", "lo": " leave alone."},
    {"id": "IT4", "facet": "Introspection", "stem": "When my mind wanders, I", "hi": " follow it into thought.", "lo": " snap back to the task."},
    {"id": "IT5", "facet": "Introspection", "stem": "Understanding myself is something I", "hi": " work at.", "lo": " rarely think about."},

    # ---- ArtAppreciation ----
    {"id": "AR1", "facet": "ArtAppreciation", "stem": "Passing an art museum, I", "hi": " want to go in.", "lo": " walk on by."},
    {"id": "AR2", "facet": "ArtAppreciation", "stem": "Toward poetry, I feel", "hi": " drawn.", "lo": " indifferent."},
    {"id": "AR3", "facet": "ArtAppreciation", "stem": "When I encounter a work of art, I", "hi": " am moved.", "lo": " feel little."},
    {"id": "AR4", "facet": "ArtAppreciation", "stem": "About the great literary classics, I", "hi": " seek them out.", "lo": " leave them unread."},
    {"id": "AR5", "facet": "ArtAppreciation", "stem": "The importance of art is something I", "hi": " believe in.", "lo": " dismiss."},

    # ---- Creativity ----
    {"id": "CR1", "facet": "Creativity", "stem": "Facing an open problem, I", "hi": " come up with new ideas.", "lo": " stick to known ones."},
    {"id": "CR2", "facet": "Creativity", "stem": "My head is", "hi": " full of ideas.", "lo": " quiet of them."},
    {"id": "CR3", "facet": "Creativity", "stem": "Toward doing things a new way, I", "hi": " love to invent one.", "lo": " keep to the usual way."},
    {"id": "CR4", "facet": "Creativity", "stem": "As a thinker, I am", "hi": " original.", "lo": " conventional."},
    {"id": "CR5", "facet": "Creativity", "stem": "My imagination is", "hi": " vivid.", "lo": " sparse."},

    # ---- Intellect ----
    {"id": "IL1", "facet": "Intellect", "stem": "Meeting a new idea, I", "hi": " catch on quickly.", "lo": " need time to get it."},
    {"id": "IL2", "facet": "Intellect", "stem": "Handed a lot of information, I", "hi": " take it in stride.", "lo": " get overwhelmed."},
    {"id": "IL3", "facet": "Intellect", "stem": "When something complex comes up, I", "hi": " grasp it fast.", "lo": " work through it slowly."},
    {"id": "IL4", "facet": "Intellect", "stem": "My mind tends to be", "hi": " quick.", "lo": " deliberate."},
    {"id": "IL5", "facet": "Intellect", "stem": "Learning something new, I", "hi": " pick it up at once.", "lo": " need it repeated."},
]

# ----------------------------------------------------------------------------------------
# v2 forced-choice items — re-authored for the 5 facets the Tier-0 scan (iter-7) flagged
# weak/broken on the v1 items DESPITE potent vectors (item-vector misalignment, D14). Each
# fix is diagnosis-driven and committed BEFORE seeing results (no iterating against the metric):
#   * SelfControl  — v1 items were concrete situational choices a long inferential hop from the
#                    abstract trait vector ("control cravings / resist temptation / never splurge").
#                    v2 ties each item to the vector's literal content (afford, temptation,
#                    overindulge, splurge) and parallelizes pole fluency.
#   * Sociability  — v1 fine but flat; v2 makes each item directly dispositional (love being
#                    around people / energized in a crowd / outgoing) to match the vector.
#   * Honesty      — v1 was NON-monotonic (ρ−0.70): the honest pole carries a "good person" halo
#                    the contrast didn't cancel. v2 frames an explicit cost/benefit trade so
#                    neither pole is the obviously better person.
#   * Introspection / EasyGoingness — weak-but-monotonic signal; v2 sharpens the poles to be more
#                    vivid and trait-direct, raising reliability.
FC_ITEMS_V2 = {
    "SelfControl": [
        {"id": "SC1", "facet": "SelfControl", "stem": "When I want something I can't really afford, I", "hi": " hold off.", "lo": " buy it anyway."},
        {"id": "SC2", "facet": "SelfControl", "stem": "Faced with a tempting treat right in front of me, I", "hi": " easily pass it up.", "lo": " give in to it."},
        {"id": "SC3", "facet": "SelfControl", "stem": "My cravings are something I", "hi": " keep firmly in check.", "lo": " let run their course."},
        {"id": "SC4", "facet": "SelfControl", "stem": "Overdoing it on something fun is for me", "hi": " a rare thing.", "lo": " a regular thing."},
        {"id": "SC5", "facet": "SelfControl", "stem": "An impulse to splurge is something I", "hi": " rein in.", "lo": " act on."},
    ],
    "Sociability": [
        {"id": "SO1", "facet": "Sociability", "stem": "Being around other people is something I", "hi": " love.", "lo": " find draining."},
        {"id": "SO2", "facet": "Sociability", "stem": "My free time, I'd rather spend", "hi": " with other people.", "lo": " by myself."},
        {"id": "SO3", "facet": "Sociability", "stem": "In a big, lively crowd, I feel", "hi": " energized.", "lo": " worn down."},
        {"id": "SO4", "facet": "Sociability", "stem": "When it comes to going out, I", "hi": " like to do it a lot.", "lo": " would mostly rather not."},
        {"id": "SO5", "facet": "Sociability", "stem": "Toward meeting new people, I am", "hi": " outgoing.", "lo": " reserved."},
    ],
    "Honesty": [
        {"id": "HO1", "facet": "Honesty", "stem": "When a small lie would smooth things over, I", "hi": " tell the truth instead.", "lo": " just tell the lie."},
        {"id": "HO2", "facet": "Honesty", "stem": "When getting ahead means cutting a corner, I", "hi": " keep to the straight path.", "lo": " cut the corner."},
        {"id": "HO3", "facet": "Honesty", "stem": "If I could benefit from bending the rules unseen, I", "hi": " wouldn't.", "lo": " would."},
        {"id": "HO4", "facet": "Honesty", "stem": "When the truth is inconvenient for me, I", "hi": " say it anyway.", "lo": " massage it."},
        {"id": "HO5", "facet": "Honesty", "stem": "Using someone to get what I want is something I", "hi": " won't do.", "lo": " will do if I need to."},
    ],
    "Introspection": [
        {"id": "IT1", "facet": "Introspection", "stem": "My quiet moments, I tend to spend", "hi": " reflecting on things.", "lo": " doing rather than pondering."},
        {"id": "IT2", "facet": "Introspection", "stem": "Examining my own life is something I", "hi": " love to do.", "lo": " rarely bother with."},
        {"id": "IT3", "facet": "Introspection", "stem": "When it comes to understanding myself, I", "hi": " dig deep.", "lo": " don't overthink it."},
        {"id": "IT4", "facet": "Introspection", "stem": "My mind tends to", "hi": " turn inward and reflect.", "lo": " stay on the world outside."},
        {"id": "IT5", "facet": "Introspection", "stem": "Getting lost in thought is something I", "hi": " enjoy.", "lo": " avoid."},
    ],
    "EasyGoingness": [
        {"id": "EG1", "facet": "EasyGoingness", "stem": "The pace I like for my life is", "hi": " slow and easy.", "lo": " quick and full."},
        {"id": "EG2", "facet": "EasyGoingness", "stem": "Doing nothing for a stretch is something I", "hi": " enjoy.", "lo": " can't stand."},
        {"id": "EG3", "facet": "EasyGoingness", "stem": "When there's no rush, I", "hi": " take it easy.", "lo": " still keep busy."},
        {"id": "EG4", "facet": "EasyGoingness", "stem": "I'd describe my way of living as", "hi": " leisurely.", "lo": " driven."},
        {"id": "EG5", "facet": "EasyGoingness", "stem": "Letting things unfold at their own pace is", "hi": " how I like it.", "lo": " frustrating to me."},
    ],
}


# Per-facet winners from the iter-8 v1-vs-v2 comparison (reauthor_test.json). v2 clearly won
# (broken/weak → strong, or fixed a sign inversion) for these; the other two weak facets keep v1
# (EasyGoingness: v1 was cleaner/monotonic; SelfControl: neither item set tracks the vector — a
# deeper vector-encodes-self-report issue, not an item problem). Post-hoc engineering selection,
# labelled as such — the pre-committed scientific claim is the v1-vs-v2 comparison itself.
BEST_V2 = ["Sociability", "Introspection", "Honesty"]


def fc_items(mode="v1"):
    """Full 135-item forced-choice list (27×5, same ids).
      mode='v1'   — original items (iter-4/iter-7).
      mode='all'  — v2 items for all 5 re-authored facets.
      mode='best' — v2 only for the facets where it won (BEST_V2); v1 elsewhere."""
    if mode == "v1":
        return list(FC_ITEMS)
    swap = set(FC_ITEMS_V2) if mode == "all" else set(BEST_V2)
    out = [it for it in FC_ITEMS if it["facet"] not in swap]
    for f in swap:
        out.extend(FC_ITEMS_V2[f])
    return out


# Canonical facet order (matches spi_data.json key order; the 5 validated facets are a subset).
FACET_ORDER = list(FACETS.keys())

# A-priori Big Five domain assignment — the grouping used in the FACETS section comments above.
# This is the project's working labeling, NOT a hard SAPA/SPI ground truth: a handful of facets
# cross-load in the real SPI (Conservatism/Authoritarianism/Conformity cluster as a "traditionalism"
# axis; SensationSeeking and EasyGoingness are low-arousal/approach facets that float between
# Extraversion and Conscientiousness). It exists so the activation-space geometry can be TESTED
# against an independent labeling — "do the steering vectors recover the Big Five?" — not assumed to.
DOMAINS = {
    "Agreeableness":     ["Compassion", "Trust", "Honesty"],
    "Conscientiousness": ["Conservatism", "Authoritarianism", "EasyGoingness", "Perfectionism",
                          "Order", "Industry", "Impulsivity", "SelfControl"],
    "Neuroticism":       ["EmotionalStability", "Anxiety", "Irritability", "WellBeing",
                          "EmotionalExpressiveness"],
    "Extraversion":      ["Sociability", "Adaptability", "Charisma", "Humor", "AttentionSeeking",
                          "SensationSeeking", "Conformity"],
    "Openness":          ["Introspection", "ArtAppreciation", "Creativity", "Intellect"],
}
DOMAIN_OF = {f: d for d, fs in DOMAINS.items() for f in fs}
assert set(DOMAIN_OF) == set(FACETS), "DOMAINS must cover exactly the 27 facets"

# The 5 facets validated through iter-6 (for diffing breadth vs. the established core).
CORE5 = ["SelfControl", "Trust", "Sociability", "Conformity", "Adaptability"]


def items_by_facet():
    out = {f: [] for f in FACETS}
    for it in FC_ITEMS:
        out[it["facet"]].append(it)
    return out


def _sanity():
    assert len(FACETS) == 27, f"expected 27 facets, got {len(FACETS)}"
    for f, spec in FACETS.items():
        assert len(spec["pos"]) == len(spec["neg"]) == 3, f"{f}: need 3 pos/3 neg"
    bf = items_by_facet()
    for f in FACETS:
        assert len(bf[f]) == 5, f"{f}: need 5 FC items, got {len(bf[f])}"
    ids = [it["id"] for it in FC_ITEMS]
    assert len(ids) == len(set(ids)) == 135, f"need 135 unique FC ids, got {len(set(ids))}"
    for c in CORE5:
        assert c in FACETS, c
    print(f"OK — {len(FACETS)} facets, {len(FC_ITEMS)} forced-choice items, all checks pass")


if __name__ == "__main__":
    _sanity()
