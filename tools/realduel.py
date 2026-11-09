#!/usr/bin/env python3
"""
Bias-cancelled head-to-head on the REAL kaggle_environments engine (the true
gate). Same idea as tools/duel.py but uses the actual env, because sim2 can
disagree with real env for planet-draining strategies. For each seed it plays
BOTH slot assignments on the same map and sums A's results, cancelling the
diagonal/player-order slot bias. Run this on your Mac (needs kaggle-environments).

  python3 tools/realduel.py --a agents/v13_cap2_big085.py --b agents/v4_hammer.py \
      --players 2 --seeds 24 --jobs 6

RULE (matches our discipline): a patch must beat BOTH main(v4) and v3 here at
n>=32 before it touches main.py. n=8 single-slot is NOT a gate.
"""
import argparse
from collections import Counter
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).resolve().parents[1]
PY_TAG = f"py{sys.version_info.major}{sys.version_info.minor}"
VERSIONED_DEPS = ROOT / ".codex_deps" / PY_TAG
LEGACY_DEPS = ROOT / ".codex_deps" / "python"

if VERSIONED_DEPS.exists():
    sys.path.insert(0, str(VERSIONED_DEPS))
elif LEGACY_DEPS.exists():
    sys.path.insert(0, str(LEGACY_DEPS))

from kaggle_environments import make


def _resolve_agent_path(agent):
    path = Path(agent)
    if path.exists():
        return str(path.resolve())
    root_path = ROOT / agent
    if root_path.exists():
        return str(root_path.resolve())
    return agent


def _isolate_duplicate_file_agents(agent_refs):
    counts = Counter(ref for ref in agent_refs if Path(ref).is_file())
    duplicate_files = {ref for ref, count in counts.items() if count > 1}
    if not duplicate_files:
        return agent_refs, None

    tmp = tempfile.TemporaryDirectory(prefix="orbit_realduel_agents_")
    tmp_path = Path(tmp.name)
    isolated = []
    for slot, ref in enumerate(agent_refs):
        if ref in duplicate_files:
            src = Path(ref)
            dst = tmp_path / f"slot{slot}_{src.name}"
            shutil.copy2(src, dst)
            isolated.append(str(dst))
        else:
            isolated.append(ref)
    return isolated, tmp


def _winners(env, n):
    last = env.steps[-1]
    return [i for i in range(n) if last[i]["reward"] == 1]


def _game(a, b, seed, a_slot, players):
    agents = [None] * players
    for i in range(players):
        agents[i] = a if i == a_slot else b
    agents, tmp = _isolate_duplicate_file_agents(agents)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(agents)
    if tmp is not None:
        tmp.cleanup()
    w = _winners(env, players)
    a_win = a_slot in w
    b_win = any(i != a_slot for i in w)
    if a_win and b_win:
        return "tie"
    return "A" if a_win else "B"


def _pair(task):
    a, b, seed, players = task
    return [_game(a, b, seed, slot, players) for slot in range(players)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    a = _resolve_agent_path(args.a)
    b = _resolve_agent_path(args.b)
    tasks = [(a, b, args.seed0 + i, args.players) for i in range(args.seeds)]
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            res = pool.map(_pair, tasks)
    else:
        res = [_pair(t) for t in tasks]
    flat = [o for pair in res for o in pair]
    A = flat.count("A"); B = flat.count("B"); T = flat.count("tie")
    dec = A + B
    wr = A / dec if dec else 0.0
    ci = 1.96 * math.sqrt(wr * (1 - wr) / dec) if dec else 0.0
    lo = 100 * (wr - ci)
    verdict = "PROMOTE" if lo > 50 else ("REJECT" if 100 * (wr + ci) < 50 else "INCONCLUSIVE")
    print(f"[REAL ENV] A={args.a}  B={args.b}  games={len(flat)}")
    print(f"  A {A} | B {B} | ties {T}")
    print(f"  A decisive winrate = {100*wr:.0f}%  (95% CI +/-{100*ci:.0f}pp, n={dec})  -> {verdict}")


if __name__ == "__main__":
    main()
