#!/usr/bin/env python3
"""Revolution detector for Orbit Wars — run daily until the deadline.

Two early-warning signals that the public base shifted (the thing that sank us
twice: v1 -> v2):
  1. LB drift: our score / the silver+bronze cutoffs vs the last snapshot. If
     the cutoffs jump or our rank slides with no new submission from us, the
     field got a stronger base.
  2. New public kernels: lists the most recent public notebooks for the comp,
     flagging slawek / producer / new strong-base candidates to fork-and-rebase.

Stores a snapshot in analysis/lb_watch.json; diffs against it each run.

  python3 tools/watch_lb.py                 # one check (cron this daily)
  python3 tools/watch_lb.py --team MorS
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "analysis/lb_watch.json"
COMP = "orbit-wars"


def pull_lb():
    d = tempfile.mkdtemp()
    subprocess.run(["kaggle", "competitions", "leaderboard", COMP, "--download", "-p", d],
                   capture_output=True)
    z = glob.glob(os.path.join(d, "*.zip"))
    if z:
        subprocess.run(["unzip", "-o", "-q", z[0], "-d", d])
    csvs = glob.glob(os.path.join(d, "*.csv"))
    rows = list(csv.DictReader(open(sorted(csvs)[-1])))
    rows.sort(key=lambda r: -float(r["Score"]))
    return rows


def recent_kernels(n=15):
    out = subprocess.run(
        ["kaggle", "kernels", "list", "--competition", COMP,
         "--sort-by", "dateCreated", "--page-size", str(n)],
        capture_output=True, text=True).stdout
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="puBERTat")
    args = ap.parse_args()

    rows = pull_lb()
    n = len(rows)
    cut = {pct: float(rows[max(0, int(n * pct) - 1)]["Score"])
           for pct in (0.01, 0.05, 0.10)}
    ours = next((i for i, r in enumerate(rows) if args.team.lower() in r["TeamName"].lower()), None)
    snap = {
        "teams": n,
        "top1": float(rows[0]["Score"]),
        "cut_gold": cut[0.01], "cut_silver": cut[0.05], "cut_bronze": cut[0.10],
        "our_rank": (ours + 1) if ours is not None else None,
        "our_score": float(rows[ours]["Score"]) if ours is not None else None,
    }

    prev = json.loads(SNAP.read_text()) if SNAP.exists() else None
    print(f"=== LB watch ({n} teams) ===")
    print(f"top1 {snap['top1']}  | gold {snap['cut_gold']}  silver {snap['cut_silver']}  bronze {snap['cut_bronze']}")
    print(f"{args.team}: rank {snap['our_rank']}  score {snap['our_score']}")
    if prev:
        ds = snap["cut_silver"] - prev["cut_silver"]
        dt = snap["top1"] - prev["top1"]
        dr = (snap["our_rank"] or 0) - (prev["our_rank"] or 0)
        print(f"\nΔ since last: silver-cut {ds:+.1f}  top1 {dt:+.1f}  our-rank {dr:+d}")
        flags = []
        if ds > 8: flags.append(f"SILVER CUTOFF JUMPED {ds:+.1f} — field got a stronger base")
        if dt > 15: flags.append(f"TOP1 JUMPED {dt:+.1f} — new strong agent at the top")
        if dr > 30: flags.append(f"WE SLID {dr:+d} ranks with no resubmit — REVOLUTION likely, consider rebase")
        if flags:
            print("\n*** ALERT ***")
            for f in flags:
                print("  !!", f)
        else:
            print("  no revolution signal.")
    SNAP.write_text(json.dumps(snap, indent=1))

    print("\n=== most recent public kernels (watch for producer-v3 / new base) ===")
    print(recent_kernels())


if __name__ == "__main__":
    main()
