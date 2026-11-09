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

ROOT = Path(__file__).resolve().parents[1]
BASE_AGENT = ROOT / "agents/v39_ms2p/main.py"
MIRROR = ROOT / "agents/v2_mirror/main.py"
ORBIT_LITE = ROOT / "agents/v16_endgame/orbit_lite"

# name: (stock, min, max, is_int, source_pattern)
PARAMS = {
    "horizon":            (19.0, 13.0, 24.0, True,  "horizon: int = 19"),
    "max_sources":        (16.0,  8.0, 20.0, True,  "max_sources_per_lane: int = 16"),
    "max_off_targets":    (15.0,  8.0, 20.0, True,  "max_offensive_targets: int = 15"),
    "max_def_targets":    ( 5.0,  2.0,  8.0, True,  "max_defensive_targets: int = 5"),
    "max_waves":          ( 8.0,  4.0, 10.0, True,  "max_waves_per_turn: int = 8"),
    "roi":                (1.454, 1.1,  2.2, False, "roi_threshold: float = 1.454"),
    "min_ships":          (7.982, 3.0, 11.0, False, "min_ships_to_launch: float = 7.982"),
    "beta":               (1.738, 0.5,  4.0, False, "reinforce_size_beta: float = 1.738"),
    "eta_free":           (8.000, 0.0, 12.0, False, "reinforce_eta_free: float = 8.000"),
    "eta_scale":          (4.215, 2.0, 18.0, False, "reinforce_eta_scale: float = 4.215"),
    "regroup_time":       (6.528, 4.0, 10.0, False, "max_regroup_time: float = 6.528"),
    "regroup_delta":      (0.244, 0.05, 0.60, False, "regroup_pressure_delta_min: float = 0.244"),
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
        field, _, _ = pat.partition(":")
        typ = "int" if is_int else "float"
        val = str(int(round(v))) if is_int else f"{v:.3f}"
        new = f"{field}: {typ} = {val}"
        assert pat in src, pat
        src = src.replace(pat, new)
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "main.py").write_text(src)
    link = dirpath / "orbit_lite"
    if not link.exists():
        link.symlink_to(ORBIT_LITE.resolve())
    return dirpath / "main.py"


def evaluate(agent_path: Path, *, seeds: int, seed0: int, jobs: int) -> float:
    """Bias-cancelled decisive winrate vs the mirror, in [0, 1]."""
    cmd = [
        sys.executable, str(ROOT / "tools/realduel.py"),
        "--a", str(agent_path), "--b", str(MIRROR),
        "--players", "2", "--seeds", str(seeds), "--seed0", str(seed0),
        "--jobs", str(jobs),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.search(r"A (\d+) \| B (\d+) \| ties (\d+)", out)
    if not m:
        raise RuntimeError(f"realduel parse failed:\n{out[-500:]}")
    a, b, _t = map(int, m.groups())
    return a / max(a + b, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--seeds-per-eval", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--a0", type=float, default=0.6, help="initial step scale")
    ap.add_argument("--c0", type=float, default=0.25, help="initial perturbation scale")
    ap.add_argument("--state", default=str(ROOT / "analysis/spsa_state.json"))
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
