"""Where this checkout keeps its inputs and outputs.

Resolution order for each location: an explicit environment variable, then the
directory inside this checkout, then the author's working tree. A clone with no
environment set therefore reads the committed `results/`, `vectors/` and
`instrument/` and writes alongside them, which is what `reproduce.py` relies on.

Environment variables: STEERING_REPO, STEERING_OUT, STEERING_VECTORS,
STEERING_VECTORS_INSTRUCT, STEERING_FIGS.
"""
from __future__ import annotations
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
CHECKOUT = os.path.dirname(_HERE)
_AUTHOR_TREE = os.path.expanduser("~/Developer/personality-test-27d")


def _resolve(env: str, inside_checkout: str, author_relative: str) -> str:
    override = os.environ.get(env)
    if override:
        return os.path.expanduser(override)
    local = os.path.join(CHECKOUT, inside_checkout)
    if os.path.isdir(local):
        return local
    return os.path.join(_AUTHOR_TREE, author_relative)


REPO = os.environ.get("STEERING_REPO") or (
    CHECKOUT if os.path.isdir(os.path.join(CHECKOUT, "results")) else _AUTHOR_TREE
)
OUT = _resolve("STEERING_OUT", "results", "research/loop")
VEC_DIR = _resolve("STEERING_VECTORS", "vectors", "steering/vectors")
VEC_DIR_INSTRUCT = _resolve("STEERING_VECTORS_INSTRUCT", "vectors_instruct", "steering/vectors_instruct")
FIGS = _resolve("STEERING_FIGS", "figures", "research/loop/figs")
INSTRUMENT = _resolve("STEERING_INSTRUMENT", "instrument", "instrument")
