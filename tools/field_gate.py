#!/usr/bin/env python3
"""Real-env field gate for Orbit Wars candidates.

This is the practical ladder proxy:
- 2P mode: bias-cancelled round-robin across candidates.
- 4P mode: each candidate plays a rotating mixed opponent pool.

Results are appended to CSV, so long runs can be resumed. Use small seed counts
for quick signal and larger counts only for finalists.
"""

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

from evaluate import (
    ROOT,
    isolate_duplicate_file_agents,
    make,
    resolve_agent_path,
    total_scores,
)


FIELDS = [
    "mode",
    "agent",
    "opponent",
    "seed",
    "slot",
    "game_idx",
    "reward",
    "score",
    "rank",
    "reward_win",
    "score_top",
    "strict_score_win",
    "decisive_win",
    "tie",
    "scores",
    "game_key",
    "key",
]


def short_name(spec):
    path = Path(spec)
    try:
        text = str(path.resolve().relative_to(ROOT))
    except Exception:
        text = str(spec)
    text = text.replace("external_kernels_new/", "new/")
    text = text.replace("external_kernels/", "ext/")
    text = text.replace("/extracted_main.py", "")
    text = text.replace("/extracted_submission.py", "")
    text = text.replace("agents/", "")
    text = text.replace(".py", "")
    return text


def bool_text(value):
    return "1" if value else "0"


def run_env(agent_refs, seed):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    isolated, tmp = isolate_duplicate_file_agents(agent_refs)
    try:
        final_state = env.run(isolated)[-1]
    finally:
        if tmp is not None:
            tmp.cleanup()
    rewards = [state.reward for state in final_state]
    scores = total_scores(final_state, len(agent_refs))
    return rewards, scores


def score_row(mode, agent_label, opponent_label, seed, slot, game_idx, rewards, scores):
    reward = rewards[slot]
    score = scores[slot]
    max_score = max(scores)
    top_tie = sum(s == max_score for s in scores)
    rank = 1 + sum(s > score for s in scores)
    reward_win = reward == 1
    score_top = score == max_score
    strict_score_win = score_top and top_tie == 1
    tie = score_top and top_tie > 1
    return {
        "mode": mode,
        "agent": agent_label,
        "opponent": opponent_label,
        "seed": seed,
        "slot": slot,
        "game_idx": game_idx,
        "reward": reward,
        "score": score,
        "rank": rank,
        "reward_win": bool_text(reward_win),
        "score_top": bool_text(score_top),
        "strict_score_win": bool_text(strict_score_win),
        "decisive_win": bool_text(strict_score_win),
        "tie": bool_text(tie),
        "scores": ";".join(str(s) for s in scores),
    }


def run_task(task):
    kind = task[0]
    if kind == "2p":
        _, key, a_spec, b_spec, a_ref, b_ref, seed, a_slot, game_idx = task
        refs = [b_ref, b_ref]
        refs[a_slot] = a_ref
        rewards, scores = run_env(refs, seed)
        row_a = score_row(
            "2p",
            short_name(a_spec),
            short_name(b_spec),
            seed,
            a_slot,
            game_idx,
            rewards,
            scores,
        )
        b_slot = 1 - a_slot
        a_win = rewards[a_slot] == 1
        b_win = rewards[b_slot] == 1
        tied = a_win and b_win
        row_a["tie"] = bool_text(tied)
        row_a["decisive_win"] = bool_text(a_win and not b_win)
        row_a["game_key"] = key
        row_a["key"] = f"{key}|A"

        row_b = score_row(
            "2p",
            short_name(b_spec),
            short_name(a_spec),
            seed,
            b_slot,
            game_idx,
            rewards,
            scores,
        )
        row_b["tie"] = bool_text(tied)
        row_b["decisive_win"] = bool_text(b_win and not a_win)
        row_b["game_key"] = key
        row_b["key"] = f"{key}|B"
        return [row_a, row_b]

    _, key, agent_spec, agent_ref, pool_specs, pool_refs, seed, slot, game_idx = task
    opponents = []
    offset = (seed + game_idx) % len(pool_refs)
    for i in range(3):
        opponents.append(pool_refs[(offset + i) % len(pool_refs)])
    refs = []
    opp_idx = 0
    for i in range(4):
        if i == slot:
            refs.append(agent_ref)
        else:
            refs.append(opponents[opp_idx])
            opp_idx += 1
    rewards, scores = run_env(refs, seed)
    row = score_row(
        "4p",
        short_name(agent_spec),
        "mixed_pool",
        seed,
        slot,
        game_idx,
        rewards,
        scores,
    )
    row["game_key"] = key
    row["key"] = key
    return [row]


def existing_game_keys(path):
    if not path.exists():
        return set()
    with path.open(newline="") as f:
        out = set()
        for row in csv.DictReader(f):
            if row.get("game_key"):
                out.add(row["game_key"])
            elif row.get("key"):
                out.add(row["key"])
        return out


def append_rows(path, rows, write_header=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header or not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_rows(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows):
    by = defaultdict(list)
    for row in rows:
        by[(row["mode"], row["agent"])].append(row)

    print("\n=== REAL FIELD GATE SUMMARY ===")
    for mode in ("2p", "4p"):
        mode_rows = [(agent, rs) for (m, agent), rs in by.items() if m == mode]
        if not mode_rows:
            continue
        print(f"\n[{mode.upper()}]")
        scored = []
        for agent, rs in mode_rows:
            n = len(rs)
            if mode == "2p":
                dec = [r for r in rs if r["tie"] != "1"]
                denom = len(dec)
                wins = sum(1 for r in dec if r["decisive_win"] == "1")
                rate = wins / denom if denom else 0.0
                ci = 1.96 * math.sqrt(rate * (1 - rate) / denom) if denom else 0.0
                scored.append((rate, agent, f"{wins}/{denom} decisive, ties {n - denom}, CI +/- {ci:.1%}"))
            else:
                strict = sum(1 for r in rs if r["strict_score_win"] == "1")
                reward = sum(1 for r in rs if r["reward_win"] == "1")
                ranks = [int(r["rank"]) for r in rs]
                scores = [int(float(r["score"])) for r in rs]
                rate = strict / n if n else 0.0
                ci = 1.96 * math.sqrt(rate * (1 - rate) / n) if n else 0.0
                detail = (
                    f"strict {strict}/{n}, reward {reward}/{n}, "
                    f"rank {statistics.mean(ranks):.2f}, score {statistics.mean(scores):.1f}, "
                    f"CI +/- {ci:.1%}"
                )
                scored.append((rate, agent, detail))
        for rate, agent, detail in sorted(scored, reverse=True):
            print(f"{agent:<38} {rate:>6.1%}  {detail}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("2p", "4p", "both"), default="both")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--pool", nargs="+", default=None, help="4P mixed opponent pool")
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--seed0", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--out", default="analysis/field_gate_results.csv")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    if args.fresh and out.exists():
        out.unlink()

    candidate_refs = [(spec, resolve_agent_path(spec)) for spec in args.candidates]
    pool_specs = args.pool or args.candidates
    pool_refs = [(spec, resolve_agent_path(spec)) for spec in pool_specs]

    tasks = []
    game_idx = 0
    if args.mode in ("2p", "both"):
        for (a_spec, a_ref), (b_spec, b_ref) in combinations(candidate_refs, 2):
            for seed in range(args.seed0, args.seed0 + args.seeds):
                for slot in (0, 1):
                    key = f"2p|{short_name(a_spec)}|{short_name(b_spec)}|{seed}|{slot}"
                    tasks.append(("2p", key, a_spec, b_spec, a_ref, b_ref, seed, slot, game_idx))
                    game_idx += 1

    if args.mode in ("4p", "both"):
        if len(pool_refs) < 3:
            raise SystemExit("4P mode needs at least 3 pool agents")
        for agent_spec, agent_ref in candidate_refs:
            effective_pool = [(s, r) for s, r in pool_refs if r != agent_ref]
            if len(effective_pool) < 3:
                effective_pool = pool_refs
            ps, pr = zip(*effective_pool)
            for seed in range(args.seed0, args.seed0 + args.seeds):
                for slot in range(4):
                    key = f"4p|{short_name(agent_spec)}|mixed|{seed}|{slot}"
                    tasks.append(("4p", key, agent_spec, agent_ref, ps, pr, seed, slot, game_idx))
                    game_idx += 1

    done = existing_game_keys(out)
    pending = [task for task in tasks if task[1] not in done]
    print(f"tasks: {len(tasks)} total, {len(done)} done, {len(pending)} pending")

    if pending:
        if args.jobs > 1:
            with Pool(args.jobs) as pool:
                for rows in pool.imap_unordered(run_task, pending):
                    append_rows(out, rows)
        else:
            for task in pending:
                append_rows(out, run_task(task))

    task_keys = {task[1] for task in tasks}
    rows = [row for row in load_rows(out) if (row.get("game_key") or row.get("key")) in task_keys]
    summarize(rows)
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
