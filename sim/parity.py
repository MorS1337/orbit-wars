#!/usr/bin/env python3
"""Parity check: batched_step vs the real engine oracle (pre-comet window)."""
import glob
import math
import os
import pickle

import torch

from orbit_batch import batched_step

DTYPE = torch.float64
DUMPS = os.path.join(os.path.dirname(__file__), "oracle_dumps")


def load_transitions(max_step=48):
    trs = []
    for f in sorted(glob.glob(os.path.join(DUMPS, "*.pkl"))):
        for tr in pickle.load(open(f, "rb")):
            if tr["before"]["step"] < max_step:        # pre-comet only
                trs.append(tr)
    return trs


def pack(trs, nplayers=2):
    B = len(trs)
    P = max(len(t["before"]["planets"]) for t in trs)
    planets = torch.zeros(B, P, 7, dtype=DTYPE)
    p_alive = torch.zeros(B, P, dtype=torch.bool)
    init_xy = torch.zeros(B, P, 2, dtype=DTYPE)
    av = torch.zeros(B, dtype=DTYPE)
    step = torch.zeros(B, dtype=torch.long)
    # initial planet xy keyed by id
    Fmax = max(len(t["before"]["fleets"]) for t in trs) or 1
    fleets = torch.zeros(B, Fmax, 7, dtype=DTYPE)
    f_alive = torch.zeros(B, Fmax, dtype=torch.bool)
    # per-planet aligned moves: moves[pl][b, slot] = (from_id, angle, ships)
    moves = [torch.zeros(B, P, 3, dtype=DTYPE) for _ in range(nplayers)]
    for b, t in enumerate(trs):
        bef = t["before"]
        av[b] = bef["angular_velocity"]
        step[b] = bef["step"]
        init_by_id = {int(p[0]): p for p in bef["initial_planets"]}
        slot_of_id = {}
        for j, p in enumerate(bef["planets"]):
            planets[b, j] = torch.tensor(p, dtype=DTYPE)
            p_alive[b, j] = True
            slot_of_id[int(p[0])] = j
            ip = init_by_id.get(int(p[0]))
            init_xy[b, j] = torch.tensor([ip[2], ip[3]] if ip else [p[2], p[3]], dtype=DTYPE)
        for j, fl in enumerate(bef["fleets"]):
            fleets[b, j] = torch.tensor(fl, dtype=DTYPE)
            f_alive[b, j] = True
        for pl in range(nplayers):
            for mv in (t["actions"][pl] or []):
                slot = slot_of_id.get(int(mv[0]))
                if slot is not None:
                    moves[pl][b, slot] = torch.tensor(mv, dtype=DTYPE)
    cfg = trs[0]["config"]
    return dict(planets=planets, p_alive=p_alive, fleets=fleets, f_alive=f_alive,
                init_xy=init_xy, av=av, step=step, moves=moves,
                max_speed=cfg["shipSpeed"], P=P)


def compare(trs, out, step_shift=0):
    planets, p_alive, fleets, f_alive = out
    n_pl_ok = n_pl_bad = 0
    n_fl_ok = n_fl_bad = 0
    worst = []
    for b, t in enumerate(trs):
        aft = t["after"]
        # planets by id
        got = {int(planets[b, j, 0]): planets[b, j] for j in range(planets.shape[1]) if p_alive[b, j]}
        exp = {int(p[0]): p for p in aft["planets"]}
        for pid, ep in exp.items():
            gp = got.get(pid)
            if gp is None:
                n_pl_bad += 1; continue
            ok = (int(gp[1]) == int(ep[1]) and abs(float(gp[5]) - ep[5]) < 1e-6
                  and abs(float(gp[2]) - ep[2]) < 1e-6 and abs(float(gp[3]) - ep[3]) < 1e-6)
            if ok:
                n_pl_ok += 1
            else:
                n_pl_bad += 1
                if len(worst) < 6:
                    worst.append(f"  ep_b{b} step{t['before']['step']} planet{pid}: "
                                 f"got owner={int(gp[1])} ships={float(gp[5]):.3f} xy=({float(gp[2]):.3f},{float(gp[3]):.3f}) | "
                                 f"exp owner={int(ep[1])} ships={ep[5]} xy=({ep[2]:.3f},{ep[3]:.3f})")
        # fleets as multiset of (owner, round ships, round x, round y)
        def fkey(o, s, x, y):
            return (int(o), int(round(float(s))), round(float(x), 3), round(float(y), 3))
        gset = sorted(fkey(fleets[b, j, 1], fleets[b, j, 6], float(fleets[b, j, 2]), float(fleets[b, j, 3]))
                      for j in range(fleets.shape[1]) if f_alive[b, j])
        eset = sorted(fkey(fl[1], fl[6], fl[2], fl[3]) for fl in aft["fleets"])
        if gset == eset:
            n_fl_ok += 1
        else:
            n_fl_bad += 1
            if len(worst) < 12:
                worst.append(f"  ep_b{b} step{t['before']['step']} FLEETS got={gset[:4]} exp={eset[:4]} (n got/exp={len(gset)}/{len(eset)})")
    print(f"PLANETS ok={n_pl_ok} bad={n_pl_bad}  | FLEET-SETS ok={n_fl_ok} bad={n_fl_bad}  (n_trans={len(trs)})")
    for w in worst:
        print(w)
    return n_pl_bad == 0 and n_fl_bad == 0


def main():
    trs = load_transitions()
    print(f"loaded {len(trs)} pre-comet transitions")
    data = pack(trs)
    out = batched_step(data["planets"], data["p_alive"], data["fleets"], data["f_alive"],
                       data["init_xy"], data["av"], data["step"], data["moves"],
                       max_speed=data["max_speed"])
    ok = compare(trs, out)
    print("PARITY", "✅ PASS" if ok else "❌ FAIL")


if __name__ == "__main__":
    main()
