#!/usr/bin/env python3
"""Warmstart PolicyTP by behaviour-cloning a scripted target-pointer teacher.

Teacher (competent, native target-pointer): each owned planet targets the
nearest non-owned planet it can afford (ships >= predicted garrison + 1), sends
~70%, aims via intercept. This skips the degenerate from-scratch RL discovery.
Clone the net to match, then RL-finetune (ppo) starting from competent play.
"""
import argparse
import time
import torch
from torch import optim

import env as E
import rl_interface as RL
from policy_tp import PolicyTP, decode_moves
from rl_interface import FRACS, N_FRAC


def teacher_action(env, seat):
    """Returns (launch[B,P] bool, tgt[B,P] long, fi[B,P] long) — scripted."""
    B, P = env.planets.shape[0], env.planets.shape[1]
    dev = env.device
    owner = env.planets[:, :, 1].long()
    ships = env.planets[:, :, 5]
    prod = env.planets[:, :, 6]
    mine = (owner == seat) & env.p_alive & (ships > 0)
    pos = env.planets[:, :, 2:4]
    # distance source->target
    d2 = ((pos.unsqueeze(2) - pos.unsqueeze(1)) ** 2).sum(-1)              # [B,src,tgt]
    notmine = (owner != seat) & env.p_alive                                # [B,tgt]
    # predicted defenders ~ ships + small growth; affordable if my ships >= need
    need = ships.unsqueeze(1) + 1.0                                         # [B,1,tgt] approx
    affordable = ships.unsqueeze(2) >= (need)                              # [B,src,tgt]
    valid = notmine.unsqueeze(1) & affordable & (d2 > 0)
    big = torch.full_like(d2, 1e18)
    score = torch.where(valid, d2, big)                                    # nearest affordable
    tgt = score.argmin(2)                                                  # [B,src]
    has_tgt = valid.any(2)                                                 # [B,src]
    launch = mine & has_tgt
    # frac ~ 0.75 (bin index 2)
    fi = torch.full((B, P), 2, dtype=torch.long, device=dev)
    return launch, tgt, fi


def gen_demos(B, T, n_rollouts, dev, seed0=0):
    """Run the teacher in self-play; collect (obs, action) for seat 0."""
    obs_list, am_l, la_l, tg_l, fi_l = [], [], [], [], []
    for r in range(n_rollouts):
        env = E.BatchedOrbitEnv(B, 2, T + 2, device=dev, dtype=torch.float32).reset(seed0=seed0 + r * B)
        for t in range(T):
            ob = RL.encode_obs(env, 0)
            la0, tg0, fi0 = teacher_action(env, 0)
            am = (env.planets[:, :, 1].long() == 0) & env.p_alive & (env.planets[:, :, 5] > 0)
            obs_list.append({k: ob[k].detach() for k in
                             ["planet_feats", "planet_mask", "fleet_feats", "fleet_mask", "global_feats"]})
            am_l.append(am.detach()); la_l.append(la0.detach()); tg_l.append(tg0.detach()); fi_l.append(fi0.detach())
            # both seats play teacher
            m0 = decode_moves(env, 0, la0, tg0, fi0)
            la1, tg1, fi1 = teacher_action(env, 1)
            m1 = decode_moves(env, 1, la1, tg1, fi1)
            env.step([m0, m1])
    return obs_list, am_l, la_l, tg_l, fi_l


def evaluate(policy, B, T, dev, seed0, opponent_teacher=False):
    from policy_tp import act
    env = E.BatchedOrbitEnv(B, 2, T + 2, device=dev, dtype=torch.float32).reset(seed0=seed0)
    for t in range(T):
        ob = RL.encode_obs(env, 0)
        with torch.no_grad():
            la, tg, fi, *_ = act(policy, ob, 0, env, greedy=True)
        m0 = decode_moves(env, 0, la, tg, fi)
        if opponent_teacher:
            la1, tg1, fi1 = teacher_action(env, 1); m1 = decode_moves(env, 1, la1, tg1, fi1)
        else:
            P = env.planets.shape[1]
            owner = env.planets[:, :, 1].long(); sh = env.planets[:, :, 5]
            mine = (owner == 1) & env.p_alive & (sh > 0) & (torch.rand(B, P, device=dev) < 0.1)
            tg1 = torch.randint(0, P, (B, P), device=dev); fi1 = torch.randint(0, N_FRAC, (B, P), device=dev)
            m1 = decode_moves(env, 1, mine, tg1, fi1)
        env.step([m0, m1])
    s = env.scores()
    return ((s[:, 0] >= s.max(1).values) & (s[:, 0] > 0)).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=256)
    ap.add_argument("--T", type=int, default=80)
    ap.add_argument("--rollouts", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save", default="bc_tp.pt")
    args = ap.parse_args()
    dev = args.device
    print(f"baseline teacher vs random: {evaluate(PolicyTP().to(dev), 128, args.T, dev, 999):.2f} (untrained net)")
    t0 = time.time()
    obs_l, am_l, la_l, tg_l, fi_l = gen_demos(args.B, args.T, args.rollouts, dev)
    N = len(obs_l)
    print(f"generated {N} demo steps in {time.time()-t0:.0f}s; launch rate {torch.cat(la_l).float().mean():.2f}")
    keys = ["planet_feats", "planet_mask", "fleet_feats", "fleet_mask", "global_feats"]
    obs = {k: torch.cat([o[k] for o in obs_l], 0) for k in keys}
    am = torch.cat(am_l); la = torch.cat(la_l); tg = torch.cat(tg_l); fi = torch.cat(fi_l)
    pol = PolicyTP().to(dev)
    opt = optim.Adam(pol.parameters(), lr=5e-4)
    M = am.shape[0]
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    for ep in range(args.epochs):
        idx = torch.randperm(M, device=dev)
        tot = 0.0
        for s in range(0, M, 2048):
            mb = idx[s:s + 2048]
            ob_mb = {k: obs[k][mb] for k in keys}
            launch_l, ptr, frac_l, _ = pol(ob_mb)
            amb = am[mb]
            P = launch_l.shape[1]
            # launch BCE on actionable planets
            l_launch = (bce(launch_l, la[mb].float()) * amb).sum() / amb.sum().clamp(min=1)
            # pointer + frac CE only where teacher launched
            fire = amb & la[mb]
            if fire.any():
                ce = torch.nn.functional.cross_entropy(ptr[fire], tg[mb][fire]) \
                    + torch.nn.functional.cross_entropy(frac_l[fire], fi[mb][fire])
            else:
                ce = torch.zeros((), device=dev)
            loss = l_launch + ce
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        wr_rand = evaluate(pol, 128, args.T, dev, 999)
        wr_teach = evaluate(pol, 128, args.T, dev, 888, opponent_teacher=True)
        print(f"ep {ep} | bc_loss {tot:.2f} | clone vs_random {wr_rand:.2f} | clone vs_TEACHER {wr_teach:.2f}")
    torch.save({"model": pol.state_dict()}, args.save)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
