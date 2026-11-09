#!/usr/bin/env python3
"""Small local tournament runner for Orbit Wars agents."""

import argparse
from collections import Counter
import math
import os
import shutil
import statistics
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PY_TAG = f"py{sys.version_info.major}{sys.version_info.minor}"
VERSIONED_DEPS = ROOT / ".codex_deps" / PY_TAG
LEGACY_DEPS = ROOT / ".codex_deps" / "python"

if VERSIONED_DEPS.exists():
    sys.path.insert(0, str(VERSIONED_DEPS))
elif LEGACY_DEPS.exists():
    rpds_files = list((LEGACY_DEPS / "rpds").glob("rpds.cpython-*.so"))
    expected = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
    if not rpds_files or any(expected in path.name for path in rpds_files):
        sys.path.insert(0, str(LEGACY_DEPS))

try:
    from kaggle_environments import make  # noqa: E402
except ModuleNotFoundError as exc:
    install_dir = ROOT / ".codex_deps" / PY_TAG
    raise SystemExit(
        "kaggle-environments is not installed for this Python interpreter.\n"
        f"Current Python: {sys.executable} ({sys.version.split()[0]})\n"
        "Install local deps with:\n"
        f"  {sys.executable} -m pip install --target {install_dir} \"kaggle-environments>=1.28.0\"\n"
        "Then rerun:\n"
        f"  {sys.executable} tools/evaluate.py\n"
    ) from exc


def obs_get(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def total_scores(final_state, players):
    obs = final_state[0].observation
    scores = [0] * players
    for planet in obs_get(obs, "planets", []):
        owner = int(planet[1])
        if 0 <= owner < players:
            scores[owner] += int(planet[5])
    for fleet in obs_get(obs, "fleets", []):
        owner = int(fleet[1])
        if 0 <= owner < players:
            scores[owner] += int(fleet[6])
    return scores


def resolve_agent_path(agent):
    path = Path(agent)
    if path.exists():
        return str(path.resolve())
    root_path = ROOT / agent
    if root_path.exists():
        return str(root_path.resolve())
    return agent


def isolate_duplicate_file_agents(agent_refs):
    """Return refs with duplicate Python files copied to per-slot temp files.

    kaggle_environments may load equal file paths through the same Python module.
    Our agents keep per-game globals, so repeated local paths can accidentally
    share state between slots. Real Kaggle submissions run independently; this
    makes local multi-copy tests closer to that.
    """
    counts = Counter(ref for ref in agent_refs if Path(ref).is_file())
    duplicate_files = {ref for ref, count in counts.items() if count > 1}
    if not duplicate_files:
        return agent_refs, None

    tmp = tempfile.TemporaryDirectory(prefix="orbit_eval_agents_")
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


def run_game(agent, opponent, seed, players, our_slot, debug):
    env = make("orbit_wars", configuration={"seed": seed}, debug=debug)
    agent_ref = resolve_agent_path(agent)
    opponent_ref = resolve_agent_path(opponent)
    agents = [opponent_ref] * players
    agents[our_slot] = agent_ref
    agents, tmp = isolate_duplicate_file_agents(agents)
    final_state = env.run(agents)[-1]
    if tmp is not None:
        tmp.cleanup()
    rewards = [state.reward for state in final_state]
    statuses = [state.status for state in final_state]
    scores = total_scores(final_state, players)
    return {
        "seed": seed,
        "slot": our_slot,
        "reward": rewards[our_slot],
        "status": statuses[our_slot],
        "score": scores[our_slot],
        "scores": scores,
        "won": rewards[our_slot] == 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="main.py")
    parser.add_argument("--opponent", default="starter")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--players", type=int, choices=(2, 4), default=2)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.players == 4 and args.opponent == "starter":
        print("warning: starter is usually tuned for baseline play; use random/main.py for harsher 4p checks")

    results = []
    for i in range(args.episodes):
        seed = args.seed_start + i
        slots = [0] if args.no_rotate else list(range(args.players))
        for slot in slots:
            result = run_game(args.agent, args.opponent, seed, args.players, slot, args.debug)
            results.append(result)
            if not args.quiet:
                score_text = ",".join(str(x) for x in result["scores"])
                print(
                    f"seed={seed:04d} slot={slot} reward={result['reward']:>2} "
                    f"status={result['status']:<5} score={result['score']:>5} scores=[{score_text}]",
                    flush=True,
                )

    wins = sum(1 for result in results if result["won"])
    rewards = [result["reward"] for result in results]
    scores = [result["score"] for result in results]
    winrate = wins / max(1, len(results))
    ci = 1.96 * math.sqrt(winrate * (1 - winrate) / len(results)) if results else 0.0
    print()
    print(f"agent: {args.agent}")
    print(f"opponent: {args.opponent}")
    print(f"players: {args.players}")
    print(f"episodes: {args.episodes}")
    print(f"slots per seed: {1 if args.no_rotate else args.players}")
    print(f"games: {len(results)}")
    print(f"wins: {wins} ({winrate:.1%}, 95% CI +/- {ci:.1%})")
    print(f"mean reward: {statistics.mean(rewards):.3f}")
    print(f"mean score: {statistics.mean(scores):.1f}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
