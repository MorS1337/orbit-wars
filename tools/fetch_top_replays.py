#!/usr/bin/env python3
"""Bulk-download Orbit Wars replays that contain leaderboard-top teachers.

Kaggle exposes episode lists by submission id, not by arbitrary team. The
practical BC workflow is therefore:

1. Take our high-rated submissions.
2. List every public episode they played.
3. Download replays, parse TeamNames, and match slots against a fresh
   leaderboard top-N / score threshold.
4. Write a teacher manifest for downstream BC extraction.

Example:

  python3 tools/fetch_top_replays.py \
      --submissions 53514141 53530544 53540987 53593544 \
      --leaderboard data/orbit-wars.zip \
      --top-rank 100 --out replays/bc_top100 --max-downloads 600
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMP = "orbit-wars"


def run(cmd: list[str], *, capture: bool = True) -> subprocess.CompletedProcess:
    print("  $", " ".join(cmd), flush=True)
    return subprocess.run(cmd, capture_output=capture, text=True)


def normalize_name(value: str | None) -> str:
    return " ".join((value or "").strip().lower().split())


def load_leaderboard(path: Path | None) -> list[dict]:
    if path is None:
        candidates = (
            glob.glob(str(ROOT / "data" / "*orbit-wars*leaderboard*.csv"))
            + glob.glob(str(ROOT / "data" / "*orbit-wars*.zip"))
            + glob.glob(str(ROOT / "*orbit-wars*leaderboard*.csv"))
            + glob.glob(str(ROOT / "*orbit-wars*.zip"))
        )
        if not candidates:
            raise SystemExit("no leaderboard CSV/ZIP found; pass --leaderboard")
        path = max((Path(p) for p in candidates), key=lambda p: p.stat().st_mtime)

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise SystemExit(f"no CSV inside {path}")
            text = zf.read(names[0]).decode("utf-8-sig")
            rows = list(csv.DictReader(io.StringIO(text)))
    else:
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))

    out = []
    for row in rows:
        try:
            rank = int(float(row.get("Rank") or row.get("rank") or "0"))
            score = float(row.get("Score") or row.get("score") or "nan")
        except ValueError:
            continue
        name = row.get("TeamName") or row.get("teamName") or row.get("Team") or row.get("Name") or ""
        if not name:
            continue
        item = dict(row)
        item["_rank"] = rank
        item["_score"] = score
        item["_name"] = name
        item["_norm"] = normalize_name(name)
        out.append(item)
    out.sort(key=lambda r: r["_rank"])
    print(f"leaderboard: {len(out)} teams from {path}")
    return out


def teacher_lookup(
    rows: list[dict],
    *,
    top_rank: int,
    min_score: float | None,
    include: set[str],
    exclude: set[str],
) -> dict[str, dict]:
    teachers = {}
    for row in rows:
        if row["_norm"] in exclude:
            continue
        if include:
            if row["_norm"] in include:
                teachers[row["_norm"]] = row
            continue
        if top_rank and row["_rank"] <= top_rank:
            teachers[row["_norm"]] = row
        if min_score is not None and row["_score"] >= min_score:
            teachers[row["_norm"]] = row
    print(
        f"teacher set: {len(teachers)} teams "
        f"(top_rank={top_rank}, min_score={min_score}, included={len(include)}, excluded={len(exclude)})"
    )
    return teachers


def parse_csv_stdout(text: str) -> list[dict]:
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        lower = line.lower()
        if "," in line and ("id" in lower or "episode" in lower):
            start = i
            break
    if start is None:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def list_episodes(submission: str, cache_dir: Path, *, refresh: bool) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"submission_{submission}_episodes.csv"
    if cache.exists() and not refresh:
        return list(csv.DictReader(cache.open(newline="")))

    res = run(["kaggle", "competitions", "episodes", str(submission), "-v"])
    if res.returncode != 0:
        print(f"warning: episodes failed for {submission}: {(res.stderr or res.stdout)[:500]}", file=sys.stderr)
        return []
    rows = parse_csv_stdout(res.stdout)
    if not rows:
        print(f"warning: no episode rows parsed for {submission}", file=sys.stderr)
        return []
    with cache.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def episode_id(row: dict) -> str | None:
    for key in ("id", "Id", "EpisodeId", "episodeId", "episode_id"):
        val = row.get(key)
        if val:
            return str(val)
    for val in row.values():
        text = str(val or "")
        if text.isdigit() and len(text) >= 6:
            return text
    return None


def download_replay(eid: str, out_dir: Path, *, refresh: bool) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob(f"*{eid}*.json"))
    if existing and not refresh:
        return existing[0]

    res = run(["kaggle", "competitions", "replay", str(eid), "-p", str(out_dir)])
    if res.returncode != 0:
        print(f"warning: replay failed for {eid}: {(res.stderr or res.stdout)[:500]}", file=sys.stderr)
        return None
    found = sorted(out_dir.glob(f"*{eid}*.json"))
    if found:
        return found[0]

    # Some CLI versions save as just '<id>.json'.
    fallback = out_dir / f"{eid}.json"
    return fallback if fallback.exists() else None


def names_of(replay: dict) -> list[str]:
    info = replay.get("info") or {}
    names = info.get("TeamNames")
    if names:
        return [str(x or "") for x in names]
    agents = info.get("Agents") or []
    return [str((agent or {}).get("Name") or "") for agent in agents]


def final_scores(replay: dict, players: int) -> list[int]:
    try:
        obs = replay["steps"][-1][0]["observation"]
    except Exception:
        return [0] * players
    scores = [0] * players
    for planet in obs.get("planets", []):
        owner = int(planet[1])
        if 0 <= owner < players:
            scores[owner] += int(planet[5])
    for fleet in obs.get("fleets", []):
        owner = int(fleet[1])
        if 0 <= owner < players:
            scores[owner] += int(fleet[6])
    return scores


def replay_rows(
    replay_path: Path,
    *,
    source_submission: str,
    teachers: dict[str, dict],
    players_filter: int,
    winner_only: bool,
) -> list[dict]:
    try:
        replay = json.loads(replay_path.read_text())
    except Exception as exc:
        print(f"warning: cannot parse {replay_path}: {exc}", file=sys.stderr)
        return []

    steps = replay.get("steps") or []
    if not steps:
        return []
    players = len(steps[0])
    if players_filter and players != players_filter:
        return []

    names = names_of(replay)
    rewards = replay.get("rewards") or [state.get("reward") for state in steps[-1]]
    statuses = replay.get("statuses") or [state.get("status") for state in steps[-1]]
    scores = final_scores(replay, players)
    max_score = max(scores) if scores else 0
    eid = str(replay.get("id") or (replay.get("info") or {}).get("EpisodeId") or replay_path.stem)

    rows = []
    for slot, name in enumerate(names[:players]):
        lb = teachers.get(normalize_name(name))
        if not lb:
            continue
        reward = rewards[slot] if slot < len(rewards) else ""
        if winner_only and reward != 1:
            continue
        rows.append(
            {
                "episode_id": eid,
                "source_submission": source_submission,
                "replay_path": str(replay_path.relative_to(ROOT) if replay_path.is_relative_to(ROOT) else replay_path),
                "players": players,
                "slot": slot,
                "team_name": name,
                "rank": lb["_rank"],
                "score": lb["_score"],
                "reward": reward,
                "status": statuses[slot] if slot < len(statuses) else "",
                "final_score": scores[slot] if slot < len(scores) else "",
                "strict_winner": int(scores[slot] == max_score and scores.count(max_score) == 1) if scores else 0,
            }
        )
    return rows


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "episode_id",
        "source_submission",
        "replay_path",
        "players",
        "slot",
        "team_name",
        "rank",
        "score",
        "reward",
        "status",
        "final_score",
        "strict_winner",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submissions", nargs="+", required=True, help="our submission ids to mine episodes from")
    ap.add_argument("--leaderboard", type=Path, help="fresh leaderboard CSV/ZIP; defaults to latest in data/")
    ap.add_argument("--top-rank", type=int, default=100, help="teacher team rank cutoff; 0 disables")
    ap.add_argument("--min-score", type=float, default=None, help="also include teams with score >= threshold")
    ap.add_argument("--include-team", action="append", default=[], help="only include this exact leaderboard team as teacher; repeatable")
    ap.add_argument("--exclude-team", action="append", default=[], help="team name to exclude from teachers; repeatable")
    ap.add_argument("--out", default="replays/bc_top", help="directory for replay JSON files")
    ap.add_argument("--manifest", default="data/bc_top_manifest.csv")
    ap.add_argument("--cache-dir", default="data/episode_lists")
    ap.add_argument("--players", type=int, choices=(0, 2, 4), default=0, help="0=any, 2=1v1 only, 4=FFA only")
    ap.add_argument("--winner-only", action="store_true", help="only keep teacher slots that won the episode")
    ap.add_argument("--max-episodes-per-submission", type=int, default=0, help="0=all listed episodes")
    ap.add_argument("--max-downloads", type=int, default=0, help="0=no global cap")
    ap.add_argument("--max-per-team", type=int, default=0, help="0=no cap in manifest")
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--refresh-episodes", action="store_true")
    ap.add_argument("--refresh-replays", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    leaderboard = load_leaderboard(args.leaderboard)
    include = {normalize_name(x) for x in args.include_team}
    exclude = {normalize_name(x) for x in args.exclude_team}
    teachers = teacher_lookup(
        leaderboard,
        top_rank=args.top_rank,
        min_score=args.min_score,
        include=include,
        exclude=exclude,
    )

    out_dir = (ROOT / args.out) if not os.path.isabs(args.out) else Path(args.out)
    cache_dir = (ROOT / args.cache_dir) if not os.path.isabs(args.cache_dir) else Path(args.cache_dir)
    manifest_path = (ROOT / args.manifest) if not os.path.isabs(args.manifest) else Path(args.manifest)

    episode_sources: list[tuple[str, str]] = []
    seen_eids = set()
    for submission in args.submissions:
        rows = list_episodes(submission, cache_dir, refresh=args.refresh_episodes)
        rows = [r for r in rows if "COMPLETED" in " ".join(str(v) for v in r.values()).upper()]
        if args.max_episodes_per_submission:
            rows = rows[: args.max_episodes_per_submission]
        print(f"submission {submission}: {len(rows)} completed episode rows")
        for row in rows:
            eid = episode_id(row)
            if not eid or eid in seen_eids:
                continue
            seen_eids.add(eid)
            episode_sources.append((submission, eid))

    print(f"unique episodes to inspect: {len(episode_sources)}")
    if args.dry_run:
        return

    manifest_rows: list[dict] = []
    per_team = defaultdict(int)
    downloaded = 0
    matched_episodes = 0
    for source_submission, eid in episode_sources:
        if args.max_downloads and downloaded >= args.max_downloads:
            break
        path = download_replay(eid, out_dir, refresh=args.refresh_replays)
        if path is None:
            time.sleep(args.sleep)
            continue
        downloaded += 1

        rows = replay_rows(
            path,
            source_submission=source_submission,
            teachers=teachers,
            players_filter=args.players,
            winner_only=args.winner_only,
        )
        kept = []
        for row in rows:
            key = normalize_name(row["team_name"])
            if args.max_per_team and per_team[key] >= args.max_per_team:
                continue
            per_team[key] += 1
            kept.append(row)
        if kept:
            matched_episodes += 1
            manifest_rows.extend(kept)
        if downloaded % 25 == 0:
            print(
                f"progress: downloaded={downloaded} matched_episodes={matched_episodes} "
                f"teacher_slots={len(manifest_rows)}",
                flush=True,
            )
        time.sleep(args.sleep)

    # Stable order: strongest teacher first, then episode id.
    manifest_rows.sort(key=lambda r: (int(r["rank"]), str(r["episode_id"]), int(r["slot"])))
    write_manifest(manifest_path, manifest_rows)

    print("\n=== done ===")
    print(f"downloaded replays: {downloaded} -> {out_dir}")
    print(f"matched episodes:   {matched_episodes}")
    print(f"teacher slots:      {len(manifest_rows)}")
    print(f"manifest:           {manifest_path}")
    if manifest_rows:
        print("\ntop teachers in manifest:")
        counts = defaultdict(int)
        best = {}
        for row in manifest_rows:
            counts[row["team_name"]] += 1
            best[row["team_name"]] = (row["rank"], row["score"])
        for name, count in sorted(counts.items(), key=lambda kv: (best[kv[0]][0], -kv[1]))[:20]:
            rank, score = best[name]
            print(f"  rank {rank:>4} score {float(score):>7.1f}  {count:>4}x  {name}")


if __name__ == "__main__":
    main()
