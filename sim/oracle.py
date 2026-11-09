#!/usr/bin/env python3
"""Dump ground-truth transitions from the REAL orbit_wars engine.

Test-driven sim development (per Isaiah/Frog-Parade): get the oracle first,
then build the batched simulator to match it bit-for-bit.

Each transition = (before_obs, actions_per_player, after_obs). The real engine
computes steps[t] = interpreter(steps[t-1], action@t), so:
    before = steps[t-1].observation
    actions = [steps[t][i].action for i in players]
    after  = steps[t].observation

We dump full planet/fleet/comet state so the batched step can be checked in
isolation. Saved as a pickle list of dicts under sim/oracle_dumps/.
"""
import math
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".codex_deps", "py313"))
from kaggle_environments import make  # noqa: E402


def scripted(obs, cfg):
    """Deterministic varied launches to stress the step (per-seed reproducible)."""
    player = obs["player"]
    planets = obs["planets"]
    rng = random.Random((obs["step"] + 1) * 97 + player * 7919)
    moves = []
    for p in planets:
        if p[1] == player and p[5] > 0 and rng.random() < 0.5:
            moves.append([p[0], rng.uniform(0, 2 * math.pi),
                          max(1, int(p[5] * rng.uniform(0.3, 0.9)))])
    return moves


def _obs_snapshot(obs):
    return {
        "step": int(obs.get("step", 0)),
        "angular_velocity": float(obs.get("angular_velocity", 0.0)),
        "planets": [list(p) for p in obs.get("planets", [])],
        "fleets": [list(f) for f in obs.get("fleets", [])],
        "initial_planets": [list(p) for p in obs.get("initial_planets", [])],
        "comet_planet_ids": list(obs.get("comet_planet_ids", [])),
        "next_fleet_id": int(obs.get("next_fleet_id", 0)),
    }


def dump_episode(seed, steps=60, players=2):
    env = make("orbit_wars", configuration={"episodeSteps": steps, "seed": seed}, debug=True)
    env.run([scripted] * players)
    transitions = []
    for t in range(1, len(env.steps)):
        before = env.steps[t - 1][0]["observation"]
        after = env.steps[t][0]["observation"]
        actions = [env.steps[t][i].get("action") for i in range(players)]
        transitions.append({
            "before": _obs_snapshot(before),
            "actions": actions,
            "after": _obs_snapshot(after),
            "config": {"shipSpeed": env.configuration.shipSpeed,
                       "cometSpeed": env.configuration.cometSpeed,
                       "episodeSteps": env.configuration.episodeSteps},
        })
    return transitions


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "oracle_dumps")
    os.makedirs(out_dir, exist_ok=True)
    seeds = list(range(1000, 1010))
    for seed in seeds:
        trs = dump_episode(seed, steps=60, players=2)
        with open(os.path.join(out_dir, f"ep_{seed}.pkl"), "wb") as f:
            pickle.dump(trs, f)
        # quick alignment sanity: action launches at t should match fleet growth
        print(f"seed {seed}: {len(trs)} transitions, "
              f"pre-comet (step<49) = {sum(1 for x in trs if x['before']['step'] < 49)}")
    print(f"dumped {len(seeds)} episodes to {out_dir}")


if __name__ == "__main__":
    main()
