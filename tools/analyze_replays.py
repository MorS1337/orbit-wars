#!/usr/bin/env python3
"""Summarize Orbit Wars Kaggle replay JSON files."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as bot  # noqa: E402


SAMPLE_TURNS = (20, 50, 75, 100, 125, 150, 200, 300, 400)
WINDOWS = ((0, 50), (50, 100), (100, 150), (150, 220), (220, 500))


def obs_at(replay, turn):
    idx = min(max(0, turn), len(replay["steps"]) - 1)
    return replay["steps"][idx][0]["observation"]


def player_stats(obs, players):
    stats = {
        player: {
            "planets": 0,
            "prod": 0,
            "planet_ships": 0,
            "fleet_ships": 0,
            "ships": 0,
        }
        for player in range(players)
    }
    for planet in obs.get("planets", []):
        owner = int(planet[1])
        if 0 <= owner < players:
            item = stats[owner]
            item["planets"] += 1
            item["prod"] += int(planet[6])
            item["planet_ships"] += int(planet[5])
            item["ships"] += int(planet[5])
    for fleet in obs.get("fleets", []):
        owner = int(fleet[1])
        if 0 <= owner < players:
            item = stats[owner]
            item["fleet_ships"] += int(fleet[6])
            item["ships"] += int(fleet[6])
    for item in stats.values():
        ships = max(1, item["ships"])
        item["fleet_frac"] = item["fleet_ships"] / ships
    return stats


def format_stats(stats):
    return (
        f"p{stats['prod']:>2}/n{stats['planets']:>2}/"
        f"s{stats['ships']:>4}/f{stats['fleet_frac']:.0%}"
    )


def first_hit_for_action(obs, action):
    planets = bot.normalize_planets(obs.get("planets", []))
    source_id = int(action[0])
    source = next((planet for planet in planets if planet.id == source_id), None)
    if source is None:
        return None, None
    angle = float(action[1])
    ships = int(action[2])
    comet_lookup = bot.build_comet_lookup(obs.get("comets", []))
    hit_id, hit_turn = bot.simulate_first_hit(
        source,
        angle,
        ships,
        planets,
        float(obs.get("angular_velocity", 0.0)),
        comet_lookup,
        bot.MAX_PREDICT_HORIZON,
    )
    target = next((planet for planet in planets if planet.id == hit_id), None)
    return target, hit_turn


def action_breakdown(replay, slot):
    players = len(replay["steps"][0])
    counts = {
        window: {
            "moves": 0,
            "ships": 0,
            "neutral": 0,
            "enemy": 0,
            "self": 0,
            "miss": 0,
            "unknown": 0,
        }
        for window in WINDOWS
    }
    totals = {
        player: {"moves": 0, "ships": 0}
        for player in range(players)
    }

    for turn, step in enumerate(replay["steps"]):
        for player, state in enumerate(step):
            action = state.get("action") or []
            totals[player]["moves"] += len(action)
            totals[player]["ships"] += sum(int(move[2]) for move in action)

        state = step[slot]
        obs = state.get("observation", {})
        for move in state.get("action") or []:
            ships = int(move[2])
            for window in WINDOWS:
                if window[0] <= turn < window[1]:
                    bucket = counts[window]
                    break
            else:
                continue
            bucket["moves"] += 1
            bucket["ships"] += ships
            target, _ = first_hit_for_action(obs, move)
            if target is None:
                bucket["miss"] += 1
            elif target.owner == slot:
                bucket["self"] += 1
            elif target.owner == -1:
                bucket["neutral"] += 1
            elif 0 <= target.owner < players:
                bucket["enemy"] += 1
            else:
                bucket["unknown"] += 1
    return counts, totals


def ownership_changes(replay, slot):
    gained = Counter()
    lost = Counter()
    losses_by_turn = []
    gains_by_turn = []
    previous = {}
    for turn, step in enumerate(replay["steps"]):
        planets = step[0]["observation"].get("planets", [])
        current = {int(planet[0]): int(planet[1]) for planet in planets}
        if not previous:
            previous = current
            continue
        for planet_id, owner in current.items():
            old = previous.get(planet_id, owner)
            if old == owner:
                continue
            if old == slot:
                lost[owner] += 1
                losses_by_turn.append((turn, planet_id, owner))
            if owner == slot:
                gained[old] += 1
                gains_by_turn.append((turn, planet_id, old))
        previous = current
    return gained, lost, gains_by_turn, losses_by_turn


def incoming_pressure(replay, slot):
    pressure = Counter()
    soon_pressure = Counter()
    for turn, step in enumerate(replay["steps"]):
        my_planets = {
            int(planet[0])
            for planet in step[0]["observation"].get("planets", [])
            if int(planet[1]) == slot
        }
        if not my_planets:
            continue
        fleets = step[0]["observation"].get("fleets", [])
        for fleet in fleets:
            owner = int(fleet[1])
            if owner < 0 or owner == slot:
                continue
            pressure[owner] += int(fleet[6])
            if turn <= 90:
                soon_pressure[owner] += int(fleet[6])
    return pressure, soon_pressure


def classify_issue(replay, slot, winner):
    players = len(replay["steps"][0])
    final_turn = len(replay["steps"]) - 1
    labels = []

    at50 = player_stats(obs_at(replay, 50), players)
    at75 = player_stats(obs_at(replay, 75), players)
    at100 = player_stats(obs_at(replay, 100), players)
    at125 = player_stats(obs_at(replay, 125), players)

    enemy50 = max((at50[p]["prod"] for p in range(players) if p != slot), default=0)
    enemy75 = max((at75[p]["prod"] for p in range(players) if p != slot), default=0)
    if at50[slot]["prod"] <= 0 or final_turn <= 70:
        labels.append("early_death")
    if enemy50 >= at50[slot]["prod"] + 8 or enemy75 >= at75[slot]["prod"] + 10:
        labels.append("underexpanded")
    if at75[slot]["fleet_frac"] > 0.55 or at100[slot]["fleet_frac"] > 0.55:
        labels.append("overfleeted")
    if at125[slot]["prod"] >= at125[winner]["prod"] and at125[slot]["ships"] < at125[winner]["ships"] * 0.65:
        labels.append("low_bank_after_equal_prod")

    gained, lost, _, _ = ownership_changes(replay, slot)
    enemy_loss_owners = [owner for owner, count in lost.items() if owner >= 0 and owner != slot and count > 0]
    if len(enemy_loss_owners) >= 2:
        labels.append("multi_focus")
    if sum(lost.values()) >= 8:
        labels.append("defense_collapse")

    action_counts, _ = action_breakdown(replay, slot)
    late_moves = sum(bucket["moves"] for window, bucket in action_counts.items() if window[0] >= 100)
    late_ships = sum(bucket["ships"] for window, bucket in action_counts.items() if window[0] >= 100)
    late_self = sum(bucket["self"] for window, bucket in action_counts.items() if window[0] >= 100)
    if final_turn > 130 and late_moves <= 4 and at125[slot]["prod"] < at125[winner]["prod"]:
        labels.append("overbanked_while_behind")
    if late_self >= 4 and late_self >= late_moves * 0.35:
        labels.append("too_much_consolidation")
    if late_ships > 0 and late_self >= 3 and at125[slot]["fleet_frac"] > 0.35:
        labels.append("hub_transfer_exposure")

    if not labels:
        labels.append("close_or_unclear")
    return labels


def summarize(path, team):
    replay = json.loads(Path(path).read_text())
    teams = replay.get("info", {}).get("TeamNames") or []
    if team not in teams:
        return None
    slot = teams.index(team)
    players = len(replay["steps"][0])
    rewards = [state.get("reward") for state in replay["steps"][-1]]
    winner_reward = max(reward for reward in rewards if reward is not None)
    winners = [i for i, reward in enumerate(rewards) if reward == winner_reward]
    winner = winners[0]
    won = slot in winners and rewards[slot] == 1
    final_turn = len(replay["steps"]) - 1
    gained, lost, gains_by_turn, losses_by_turn = ownership_changes(replay, slot)
    action_counts, action_totals = action_breakdown(replay, slot)
    pressure, soon_pressure = incoming_pressure(replay, slot)
    labels = classify_issue(replay, slot, winner)

    return {
        "path": Path(path),
        "episode": replay.get("id") or replay.get("info", {}).get("EpisodeId"),
        "seed": replay.get("info", {}).get("seed"),
        "teams": teams,
        "slot": slot,
        "players": players,
        "rewards": rewards,
        "winner": winner,
        "winner_name": teams[winner] if winner < len(teams) else str(winner),
        "won": won,
        "turns": final_turn,
        "gained": gained,
        "lost": lost,
        "gains_by_turn": gains_by_turn,
        "losses_by_turn": losses_by_turn,
        "actions": action_counts,
        "action_totals": action_totals,
        "pressure": pressure,
        "soon_pressure": soon_pressure,
        "labels": labels,
        "replay": replay,
    }


def print_episode(item, detail):
    replay = item["replay"]
    slot = item["slot"]
    players = item["players"]
    winner = item["winner"]
    result = "WIN" if item["won"] else "LOSS"
    team_text = ", ".join(f"{i}:{name}" for i, name in enumerate(item["teams"]))
    print(
        f"{item['episode']} {result} {players}P slot={slot} winner={winner}:{item['winner_name']} "
        f"turns={item['turns']} seed={item['seed']} labels={','.join(item['labels'])}"
    )
    print(f"  teams: {team_text}")
    print(f"  rewards: {item['rewards']}")
    for turn in SAMPLE_TURNS:
        if turn > item["turns"]:
            continue
        stats = player_stats(obs_at(replay, turn), players)
        ours = format_stats(stats[slot])
        win = format_stats(stats[winner])
        leader = max(range(players), key=lambda player: stats[player]["prod"])
        lead = format_stats(stats[leader])
        print(f"  t{turn:03d}: us {ours}; winner {win}; prod_leader {leader}:{lead}")

    lost = ", ".join(f"{owner}:{count}" for owner, count in sorted(item["lost"].items())) or "-"
    gained = ", ".join(f"{owner}:{count}" for owner, count in sorted(item["gained"].items())) or "-"
    pressure = ", ".join(f"{owner}:{ships}" for owner, ships in item["pressure"].most_common()) or "-"
    print(f"  flips: gained [{gained}] lost [{lost}] incoming_pressure [{pressure}]")

    action_line = []
    for window in WINDOWS:
        bucket = item["actions"][window]
        action_line.append(
            f"{window[0]}-{window[1]} m{bucket['moves']}/s{bucket['ships']}"
            f"/N{bucket['neutral']}/E{bucket['enemy']}/S{bucket['self']}/M{bucket['miss']}"
        )
    print("  actions: " + " | ".join(action_line))

    totals = " ".join(
        f"{player}:m{stats['moves']}/s{stats['ships']}"
        for player, stats in sorted(item["action_totals"].items())
    )
    print(f"  action_totals: {totals}")

    if detail:
        print(f"  first_losses: {item['losses_by_turn'][:12]}")
        print(f"  first_gains: {item['gains_by_turn'][:12]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--team", default="MorS")
    parser.add_argument("--losses-only", action="store_true")
    parser.add_argument("--wins-only", action="store_true")
    parser.add_argument("--detail", action="store_true")
    args = parser.parse_args()

    files = []
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            files.extend(sorted(path.glob("episode-*-replay.json")))
        else:
            files.append(path)

    items = [summarize(path, args.team) for path in files]
    items = [item for item in items if item is not None]
    if args.losses_only:
        items = [item for item in items if not item["won"]]
    if args.wins_only:
        items = [item for item in items if item["won"]]
    items.sort(key=lambda item: item["episode"])

    wins = sum(1 for item in items if item["won"])
    losses = len(items) - wins
    label_counts = Counter(label for item in items if not item["won"] for label in item["labels"])
    print(f"episodes={len(items)} wins={wins} losses={losses}")
    if label_counts:
        print("loss_labels: " + ", ".join(f"{label}={count}" for label, count in label_counts.most_common()))
    print()

    for item in items:
        print_episode(item, args.detail)
        print()


if __name__ == "__main__":
    main()
