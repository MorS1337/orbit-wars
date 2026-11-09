#!/usr/bin/env python3
"""SPSA black-box optimisation of the 2P producer config against the mirror.

Each iteration perturbs all parameters simultaneously (Rademacher delta),
measures the decisive winrate of theta+ and theta- against the stock mirror
on a FRESH seed block (16 games via tools/realduel.py), and steps along the
estimated gradient. State (theta, iteration, history) is checkpointed to
--state every iteration, so the run is resumable and the trajectory is
auditable.

  python3 tools/spsa_mirror.py --iters 150 --seeds-per-eval 8 --jobs 4 \
      --state analysis/spsa_state.json

Final candidate: build an agent from state["theta"] (tools prints the config)
and validate with the usual realduel/field_gate battery before any submit.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import field4p

ROOT = Path(__file__).resolve().parents[1]
BASE_AGENT = ROOT / "agents/v39_ms2p/main.py"
MIRROR = ROOT / "agents/v2_mirror/main.py"
ORBIT_LITE = ROOT / "agents/v16_endgame/orbit_lite"

# name: (stock, min, max, is_int, source_pattern)
PARAMS = {
    "horizon":         (13.0,  9.0, 18.0, True,  "    horizon=13,"),
    "max_sources":     ( 6.0,  4.0, 12.0, True,  "    max_sources_per_lane=6,"),
    "max_off_targets": (12.0,  6.0, 16.0, True,  "    max_offensive_targets=12,"),
    "max_def_targets": ( 2.0,  1.0,  5.0, True,  "    max_defensive_targets=2,"),
    "max_waves":       ( 6.0,  3.0,  9.0, True,  "    max_waves_per_turn=6,"),
    "roi":             ( 3.5,  1.5,  6.0, False, "    roi_threshold=1.5,"),
    "min_ships":       ( 4.0,  2.0, 10.0, False, "    min_ships_to_launch=4.0,"),
    "beta":            ( 2.2,  0.5,  4.5, False, "    reinforce_size_beta=2.2,"),
    "eta_free":        ( 3.0,  0.0,  8.0, False, "    reinforce_eta_free=3.0,"),
    "eta_scale":       (12.0,  3.0, 20.0, False, "    reinforce_eta_scale=12.0,"),
    "regroup_time":    ( 6.0,  4.0, 10.0, False, "    max_regroup_time=6.0,"),
    "regroup_delta":   (0.25, 0.05, 0.60, False, "    regroup_pressure_delta_min=0.25,"),
}
ORDER = list(PARAMS)


def clamp(theta):
    out = {}
    for k, v in theta.items():
        _s, lo, hi, is_int, _p = PARAMS[k]
        v = min(max(v, lo), hi)
        out[k] = v
    return out


def materialise(theta, dirpath: Path):
    """Write an agent dir for a parameter vector."""
    src = BASE_AGENT.read_text()
    for k, v in theta.items():
        stock, _lo, _hi, is_int, pat = PARAMS[k]
        field = pat.strip().split("=")[0]
        typ = "int" if is_int else "float"
        val = str(int(round(v))) if is_int else f"{v:.3f}"
        new = f"    {field}={val},"
        assert pat in src, pat
        src = src.replace(pat, new)
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "main.py").write_text(src)
    link = dirpath / "orbit_lite"
    if not link.exists():
        link.symlink_to(ORBIT_LITE.resolve())
    return dirpath / "main.py"


def evaluate(agent_path: Path, *, seeds: int, seed0: int, jobs: int) -> float:
    """Candidate win-rate vs the MIXED meta-zoo field, in [0, 1] (baseline ~0.25)."""
    r = field4p.evaluate(str(agent_path), pool=None, seeds=seeds,
                         seed0=seed0, jobs=jobs, rotate=True)
    return r["winrate"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--seeds-per-eval", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--a0", type=float, default=0.6, help="initial step scale")
    ap.add_argument("--c0", type=float, default=0.25, help="initial perturbation scale")
    ap.add_argument("--state", default=str(ROOT / "analysis/spsa_4p_field_state.json"))
    args = ap.parse_args()

    rng = random.Random(1337)
    state_path = Path(args.state)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        print(f"resuming at iter {state['iter']}, theta={state['theta']}")
    else:
        state = {"iter": 0, "theta": {k: PARAMS[k][0] for k in ORDER}, "history": []}

    # per-param scale: fraction of the parameter's range
    spans = {k: (PARAMS[k][2] - PARAMS[k][1]) for k in ORDER}
    work = Path(tempfile.mkdtemp(prefix="spsa_"))

    while state["iter"] < args.iters:
        it = state["iter"]
        # standard SPSA decay schedules
        a_k = args.a0 / (it + 10) ** 0.602
        c_k = args.c0 / (it + 1) ** 0.101
        delta = {k: rng.choice((-1.0, 1.0)) for k in ORDER}

        theta = state["theta"]
        t_plus = clamp({k: theta[k] + c_k * spans[k] * delta[k] for k in ORDER})
        t_minus = clamp({k: theta[k] - c_k * spans[k] * delta[k] for k in ORDER})
        p_plus = materialise(t_plus, work / "plus")
        p_minus = materialise(t_minus, work / "minus")

        seed0 = 50000 + it * 211  # fresh block every iteration
        w_plus = evaluate(p_plus, seeds=args.seeds_per_eval, seed0=seed0, jobs=args.jobs)
        w_minus = evaluate(p_minus, seeds=args.seeds_per_eval, seed0=seed0, jobs=args.jobs)

        g_scale = (w_plus - w_minus) / (2.0 * c_k)
        new_theta = clamp({
            k: theta[k] + a_k * g_scale * spans[k] * delta[k] for k in ORDER
        })
        state["theta"] = new_theta
        state["iter"] = it + 1
        state["history"].append({
            "iter": it, "w_plus": w_plus, "w_minus": w_minus,
            "theta": {k: round(v, 4) for k, v in new_theta.items()},
        })
        state_path.write_text(json.dumps(state, indent=1))
        print(
            f"iter {it:03d}  w+ {w_plus:.3f}  w- {w_minus:.3f}  "
            + " ".join(f"{k}={new_theta[k]:.2f}" for k in ORDER),
            flush=True,
        )

    print("done. final theta:", json.dumps(state["theta"], indent=1))


if __name__ == "__main__":
    main()
