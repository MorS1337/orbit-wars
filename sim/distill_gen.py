#!/usr/bin/env python3
"""Generate producer-vs-producer demonstrations via the REAL engine, in parallel.

Each step we record both seats' (raw obs snapshot, producer move list). Dumped
as pickled shards for the labeller/cloner. Run on the box (many CPU cores).
"""
import argparse
import os
import pickle
import sys
import time
import multiprocessing as mp

sys.path.insert(0, "/root/producer")   # producer/main.py + orbit_lite live here


def _load_producer():
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("prod_agent", "/root/producer/main.py")
    m = ilu.module_from_spec(spec)
    sys.modules["prod_agent"] = m
    spec.loader.exec_module(m)
    return m.agent


def _snap(obs):
    return {
        "step": int(obs.get("step", 0)),
        "av": float(obs.get("angular_velocity", 0.0)),
        "planets": [list(p) for p in obs.get("planets", [])],
        "fleets": [list(f) for f in obs.get("fleets", [])],
        "initial_planets": [list(p) for p in obs.get("initial_planets", [])],
    }


def gen_games(args):
    seed0, n_games, players = args
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    from kaggle_environments import make
    prod = _load_producer()
    demos = []
    for g in range(n_games):
        env = make("orbit_wars", configuration={"episodeSteps": 500, "seed": seed0 + g}, debug=False)
        env.run([prod] * players)
        for t in range(1, len(env.steps)):
            before = env.steps[t - 1][0]["observation"]
            snap = _snap(before)
            for seat in range(players):
                mv = env.steps[t][seat].get("action") or []
                if mv:                                   # only record steps with a launch
                    demos.append((snap, seat, players, [list(m) for m in mv]))
    return demos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=64)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out", default="/root/demos.pkl")
    args = ap.parse_args()
    workers = args.workers or min(mp.cpu_count(), 64)
    per = max(1, args.games // workers)
    tasks = [(1000 + i * per * 7, per, args.players) for i in range(workers)]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        results = pool.map(gen_games, tasks)
    demos = [d for r in results for d in r]
    pickle.dump(demos, open(args.out, "wb"))
    dt = time.time() - t0
    print(f"generated {len(demos)} launch-demos from ~{per*workers} games on {workers} workers "
          f"in {dt:.0f}s -> {len(demos)/dt:.0f} demos/s; saved {args.out}")


if __name__ == "__main__":
    main()
