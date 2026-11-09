#!/usr/bin/env python3
"""Label producer demos with target-pointer actions and behaviour-clone PolicyTP.

For each demo (raw obs snapshot, seat, producer moves):
  * owners are rotated so the acting seat -> relative 0 (one shared policy).
  * each producer move [src_id, angle, ships] is converted to a target-pointer
    label: target = the planet whose intercept-aim angle best matches the
    producer's angle; frac = ships/garrison bin; launch=1 on the source.
Then BC-train PolicyTP (launch BCE + pointer/frac CE on launched planets).
"""
import argparse
import pickle
import time
from types import SimpleNamespace

import torch
from torch import optim

import rl_interface as RL
from rl_interface import FRACS, N_FRAC
from aim import intercept_angle
from policy_tp import PolicyTP, act, decode_moves

DT = torch.float32


def build_batch(demos, dev):
    """demos: list of (snap, seat, players, moves). Vectorized: numpy array fills
    in the loop (no torch ops), then ONE batched intercept for all moves."""
    import numpy as np
    B = len(demos)
    P = max(len(d[0]["planets"]) for d in demos)
    F = max(max(len(d[0]["fleets"]), 1) for d in demos)
    pl_a = np.zeros((B, P, 7), np.float32)
    pal_a = np.zeros((B, P), bool)
    ini_a = np.zeros((B, P, 2), np.float32)
    fl_a = np.zeros((B, F, 7), np.float32)
    fal_a = np.zeros((B, F), bool)
    av_a = np.zeros(B, np.float32); st_a = np.zeros(B, np.int64)
    # flat move list: (b, src_slot, angle, ships)
    mb, msrc, mang, mship = [], [], [], []
    for b, (snap, seat, npl, moves) in enumerate(demos):
        av_a[b] = snap["av"]; st_a[b] = snap["step"]
        init_by_id = {int(p[0]): p for p in snap["initial_planets"]}
        slot = {}
        for j, p in enumerate(snap["planets"]):
            o = int(p[1]); ro = -1 if o < 0 else ((o - seat) % npl)
            pl_a[b, j] = (p[0], ro, p[2], p[3], p[4], p[5], p[6]); pal_a[b, j] = True
            slot[int(p[0])] = j
            ip = init_by_id.get(int(p[0]), p); ini_a[b, j] = (ip[2], ip[3])
        for j, fl in enumerate(snap["fleets"]):
            o = int(fl[1]); ro = -1 if o < 0 else ((o - seat) % npl)
            fl_a[b, j] = (fl[0], ro, fl[2], fl[3], fl[4], fl[5], fl[6]); fal_a[b, j] = True
        for mv in moves:
            s = slot.get(int(mv[0]))
            if s is not None:
                mb.append(b); msrc.append(s); mang.append(float(mv[1])); mship.append(float(mv[2]))
    planets = torch.from_numpy(pl_a).to(dev); p_alive = torch.from_numpy(pal_a).to(dev)
    init_xy = torch.from_numpy(ini_a).to(dev); fleets = torch.from_numpy(fl_a).to(dev)
    f_alive = torch.from_numpy(fal_a).to(dev)
    av = torch.from_numpy(av_a).to(dev); step = torch.from_numpy(st_a).to(dev)
    launch = torch.zeros(B, P, dtype=torch.bool, device=dev)
    tgt = torch.zeros(B, P, dtype=torch.long, device=dev)
    fi = torch.zeros(B, P, dtype=torch.long, device=dev)
    M = len(mb)
    if M:
        mb = torch.tensor(mb, device=dev); msrc = torch.tensor(msrc, device=dev)
        mang = torch.tensor(mang, dtype=DT, device=dev); mship = torch.tensor(mship, dtype=DT, device=dev)
        src_xy = planets[mb, msrc, 2:4]                              # [M,2]
        cand_init = init_xy[mb]                                      # [M,P,2]
        cand_rad = planets[mb, :, 4]                                # [M,P]
        ia = intercept_angle(src_xy.unsqueeze(1).expand(M, P, 2), cand_init, cand_rad,
                             av[mb].unsqueeze(1).expand(M, P), step[mb].unsqueeze(1).expand(M, P),
                             mship.unsqueeze(1).expand(M, P))         # [M,P]
        dang = torch.atan2(torch.sin(ia - mang.unsqueeze(1)), torch.cos(ia - mang.unsqueeze(1))).abs()
        dang[~p_alive[mb]] = 1e9
        dang[torch.arange(M, device=dev), msrc] = 1e9               # not self
        t = dang.argmin(1)                                          # [M]
        gar = planets[mb, msrc, 5] + 1e-6
        fb = (mship / gar * N_FRAC).round().clamp(min=1).long() - 1
        fb = fb.clamp(0, N_FRAC - 1)
        launch[mb, msrc] = True; tgt[mb, msrc] = t; fi[mb, msrc] = fb
    env = SimpleNamespace(planets=planets, p_alive=p_alive, fleets=fleets, f_alive=f_alive,
                          init_xy=init_xy, av=av, step_idx=step, np=2, device=dev, dtype=DT,
                          max_speed=6.0)
    env.scores = lambda: _scores(env)
    actionable = (planets[:, :, 1].long() == 0) & p_alive & (planets[:, :, 5] > 0)
    return env, actionable, launch, tgt, fi


def _scores(env):
    B = env.planets.shape[0]
    s = torch.zeros(B, env.np, dtype=DT, device=env.device)
    owner = env.planets[:, :, 1].long()
    for pl in range(env.np):
        s[:, pl] += (env.planets[:, :, 5] * ((owner == pl) & env.p_alive)).sum(1)
    return s


def evaluate(policy, dev, T=80, B=128, seed0=999):
    import env as E
    e = E.BatchedOrbitEnv(B, 2, T + 2, device=dev, dtype=DT).reset(seed0=seed0)
    for t in range(T):
        ob = RL.encode_obs(e, 0)
        with torch.no_grad():
            la, tg, fb, *_ = act(policy, ob, 0, e, greedy=True)
        m0 = decode_moves(e, 0, la, tg, fb)
        P = e.planets.shape[1]
        owner = e.planets[:, :, 1].long(); sh = e.planets[:, :, 5]
        mine = (owner == 1) & e.p_alive & (sh > 0) & (torch.rand(B, P, device=dev) < 0.1)
        m1 = decode_moves(e, 1, mine, torch.randint(0, P, (B, P), device=dev), torch.randint(0, N_FRAC, (B, P), device=dev))
        e.step([m0, m1])
    s = e.scores()
    return ((s[:, 0] >= s.max(1).values) & (s[:, 0] > 0)).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="/root/demos.pkl")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--save", default="/root/bc_tp.pt")
    args = ap.parse_args()
    dev = args.device
    demos = pickle.load(open(args.demos, "rb"))
    print(f"loaded {len(demos)} demos; building batch + inferring targets...")
    t0 = time.time()
    # build in chunks to bound memory, concat label tensors + keep obs on cpu
    keys = ["planet_feats", "planet_mask", "fleet_feats", "fleet_mask", "global_feats"]
    OB = {k: [] for k in keys}; AM = []; LA = []; TG = []; FI = []
    CH = 4096
    for i in range(0, len(demos), CH):
        env, am, la, tg, fi = build_batch(demos[i:i + CH], dev)
        ob = RL.encode_obs(env, 0)
        for k in keys:
            OB[k].append(ob[k].cpu())
        AM.append(am.cpu()); LA.append(la.cpu()); TG.append(tg.cpu()); FI.append(fi.cpu())
    # pad planet/fleet dims to global max for concat
    Pm = max(x.shape[1] for x in OB["planet_feats"]); Fm = max(x.shape[1] for x in OB["fleet_feats"])
    def padcat(lst, dim_to, last=None):
        out = []
        for x in lst:
            if x.shape[1] < dim_to:
                pad = list(x.shape); pad[1] = dim_to - x.shape[1]
                x = torch.cat([x, torch.zeros(*pad, dtype=x.dtype)], 1)
            out.append(x)
        return torch.cat(out, 0)
    obs = {}
    for k in keys:
        dim = Pm if "planet" in k else (Fm if "fleet" in k else None)
        obs[k] = padcat(OB[k], dim) if dim else torch.cat(OB[k], 0)
    am = padcat(AM, Pm); la = padcat(LA, Pm); tg = padcat(TG, Pm); fi = padcat(FI, Pm)
    print(f"built {am.shape[0]} samples (P={Pm},F={Fm}) in {time.time()-t0:.0f}s; "
          f"launch rate {la.float().mean():.3f}")
    obs = {k: v.to(dev) for k, v in obs.items()}
    am, la, tg, fi = am.to(dev), la.to(dev), tg.to(dev), fi.to(dev)
    pol = PolicyTP().to(dev)
    opt = optim.Adam(pol.parameters(), lr=5e-4)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    M = am.shape[0]
    for ep in range(args.epochs):
        idx = torch.randperm(M, device=dev); tot = 0.0
        for s in range(0, M, args.bs):
            mb = idx[s:s + args.bs]
            ob_mb = {k: obs[k][mb] for k in keys}
            launch_l, ptr, frac_l, _ = pol(ob_mb)
            amb = am[mb]
            l1 = (bce(launch_l, la[mb].float()) * amb).sum() / amb.sum().clamp(min=1)
            fire = amb & la[mb]
            if fire.any():
                ce = torch.nn.functional.cross_entropy(ptr[fire], tg[mb][fire]) + \
                     torch.nn.functional.cross_entropy(frac_l[fire], fi[mb][fire])
            else:
                ce = torch.zeros((), device=dev)
            loss = l1 + ce
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        wr = evaluate(pol, dev)
        print(f"ep {ep} | loss {tot:.1f} | clone vs_random {wr:.2f}", flush=True)
        torch.save({"model": pol.state_dict()}, args.save)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
