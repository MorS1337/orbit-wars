#!/usr/bin/env python3
"""Real-env pool evaluator for Orbit Wars agents.

Unlike tools/evaluate.py's clone matchup, this supports a mixed opponent pool
and reports score ranks. That matters in 4P: identical deterministic bots can
tie and all receive reward=1, so reward-win alone is a noisy promotion signal.
"""

import argparse
import math
import statistics

from evaluate import (
    isolate_duplicate_file_agents,
    make,
    resolve_agent_path,
    total_scores,
)


def build_agents(agent_ref, opponent_refs, players, our_slot, seed, game_idx):
    agents = []
    offset = (seed + game_idx) % len(opponent_refs)
    opp_idx = 0
    for slot in range(players):
        if slot == our_slot:
            agents.append(agent_ref)
        else:
            agents.append(opponent_refs[(offset + opp_idx) % len(opponent_refs)])
            opp_idx += 1
    return agents


def run_game(agent_ref, opponent_refs, seed, players, our_slot, game_idx, debug):
    env = make("orbit_wars", configuration={"seed": seed}, debug=debug)
    agents = build_agents(agent_ref, opponent_refs, players, our_slot, seed, game_idx)
    agents, tmp = isolate_duplicate_file_agents(agents)
    final_state = env.run(agents)[-1]
    if tmp is not None:
        tmp.cleanup()

    rewards = [state.reward for state in final_state]
    statuses = [state.status for state in final_state]
    scores = total_scores(final_state, players)
    our_score = scores[our_slot]
    max_score = max(scores)
    rank = 1 + sum(score > our_score for score in scores)
    top_tie = sum(score == max_score for score in scores)
    return {
        "seed": seed,
        "slot": our_slot,
        "reward": rewards[our_slot],
        "status": statuses[our_slot],
        "score": our_score,
        "scores": scores,
        "reward_win": rewards[our_slot] == 1,
        "score_top": our_score == max_score,
        "strict_score_win": our_score == max_score and top_tie == 1,
        "rank": rank,
    }


def summarize(label, count, total):
    rate = count / max(1, total)
    ci = 1.96 * math.sqrt(rate * (1 - rate) / total) if total else 0.0
    print(f"{label}: {count}/{total} ({rate:.1%}, 95% CI +/- {ci:.1%})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="main.py")
    parser.add_argument("--opponents", nargs="+", required=True)
    parser.add_argument("--players", type=int, choices=(2, 4), default=4)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if len(args.opponents) < args.players - 1:
        raise SystemExit(f"need at least {args.players - 1} opponents for {args.players}P")

    agent_ref = resolve_agent_path(args.agent)
    opponent_refs = [resolve_agent_path(opponent) for opponent in args.opponents]

    results = []
    game_idx = 0
    for i in range(args.episodes):
        seed = args.seed_start + i
        slots = [0] if args.no_rotate else list(range(args.players))
        for slot in slots:
            result = run_game(agent_ref, opponent_refs, seed, args.players, slot, game_idx, args.debug)
            results.append(result)
            game_idx += 1
            if not args.quiet:
                score_text = ",".join(str(x) for x in result["scores"])
                print(
                    f"seed={seed:04d} slot={slot} reward={result['reward']:>2} "
                    f"rank={result['rank']} status={result['status']:<5} "
                    f"score={result['score']:>5} scores=[{score_text}]",
                    flush=True,
                )

    total = len(results)
    rewards = [result["reward"] for result in results]
    scores = [result["score"] for result in results]
    ranks = [result["rank"] for result in results]
    print()
    print(f"agent: {args.agent}")
    print(f"opponents: {', '.join(args.opponents)}")
    print(f"players: {args.players}")
    print(f"episodes: {args.episodes}")
    print(f"slots per seed: {1 if args.no_rotate else args.players}")
    print(f"games: {total}")
    summarize("reward wins", sum(result["reward_win"] for result in results), total)
    summarize("score-top games", sum(result["score_top"] for result in results), total)
    summarize("strict score wins", sum(result["strict_score_win"] for result in results), total)
    print(f"mean reward: {statistics.mean(rewards):.3f}")
    print(f"mean score: {statistics.mean(scores):.1f}")
    print(f"mean rank: {statistics.mean(ranks):.2f}")


if __name__ == "__main__":
    main()
