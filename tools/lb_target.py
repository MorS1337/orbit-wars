#!/usr/bin/env python3
"""
Read the Orbit Wars leaderboard and print what each medal actually requires
(percentile thresholds + Kaggle's team-count tiers) and where YOUR team sits.
Run on your Mac (uses the kaggle CLI download).

  # download then analyze:
  kaggle competitions leaderboard orbit-wars --download -p .
  python3 tools/lb_target.py --csv orbit-wars-publicleaderboard.csv --team "MorS"

  # or let it download:
  python3 tools/lb_target.py --team "MorS"
"""
import argparse
import csv
import glob
import io
import os
import subprocess
import sys
import zipfile


def load_csv(path):
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                print(f"no CSV inside {path}"); sys.exit(1)
            data = zf.read(names[0]).decode("utf-8-sig")
            rows = list(csv.DictReader(io.StringIO(data)))
    else:
        rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    if not rows:
        print("empty CSV"); sys.exit(1)
    return rows


def score_of(row):
    for k in row:
        if k.lower() in ("score", "rating", "mu"):
            try:
                return float(row[k])
            except (TypeError, ValueError):
                return None
    return None


def name_of(row):
    for k in row:
        if "team" in k.lower() and "name" in k.lower():
            return row[k]
    for k in row:
        if k.lower() in ("teamname", "team", "name"):
            return row[k]
    return ""


def kaggle_tiers(n):
    """Kaggle competition medal cutoffs by team count (rank thresholds)."""
    if n < 100:
        return {"gold": max(1, round(0.10 * n)), "silver": max(1, round(0.20 * n)), "bronze": max(1, round(0.40 * n))}
    if n < 250:
        return {"gold": 10, "silver": max(1, round(0.20 * n)), "bronze": max(1, round(0.40 * n))}
    if n < 1000:
        return {"gold": 10 + round(0.002 * n), "silver": 50, "bronze": 100}
    return {"gold": 10 + round(0.002 * n), "silver": round(0.05 * n), "bronze": round(0.10 * n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--team", default="")
    args = ap.parse_args()

    path = args.csv
    if not path:
        subprocess.run(["kaggle", "competitions", "leaderboard", "orbit-wars", "--download", "-p", "."])
        cand = (
            glob.glob("*orbit-wars*eaderboard*.csv")
            + glob.glob("*rbit*ars*.csv")
            + glob.glob("*orbit-wars*.zip")
        )
        if not cand:
            print("could not find downloaded CSV/ZIP; pass --csv"); sys.exit(1)
        path = cand[0]
    rows = load_csv(path)

    scored = [(name_of(r), score_of(r)) for r in rows]
    scored = [(n, s) for n, s in scored if s is not None]
    scored.sort(key=lambda t: -t[1])
    n = len(scored)
    tiers = kaggle_tiers(n)
    print(f"teams = {n}")
    for medal in ("gold", "silver", "bronze"):
        rank = min(tiers[medal], n)
        thr = scored[rank - 1][1]
        print(f"  {medal:<6}: rank <= {rank:<4} -> score >= {thr:.1f}")

    if args.team:
        for i, (nm, s) in enumerate(scored):
            if args.team.lower() in (nm or "").lower():
                pct = 100 * (i + 1) / n
                print(f"\nYOU: '{nm}' rank {i+1}/{n} (top {pct:.1f}%) score {s:.1f}")
                # gap to silver
                srank = min(tiers["silver"], n)
                print(f"  silver needs rank <= {srank} (score >= {scored[srank-1][1]:.1f}); "
                      f"you are {i+1 - srank:+d} ranks / {s - scored[srank-1][1]:+.1f} score away")
                break
        else:
            print(f"\nteam matching '{args.team}' not found on leaderboard")


if __name__ == "__main__":
    main()
