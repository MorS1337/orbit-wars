#!/usr/bin/env python3
"""Minimal self-play PPO for Orbit Wars on the batched sim.

Fixed-length episodes, terminal sparse reward, GAE, clipped PPO. Both seats use
the current policy (symmetric self-play -> 2x data). Eval vs a random opponent
each iteration so you can WATCH winrate climb.
"""
import argparse
import time
import torch
from torch import optim

import env as E
import rl_interface as RL
from policy import Policy, act, logprob_value


def detach_ob(ob):
    return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in ob.items()}


def rollout(policy, B, T, nplayers, seed0, device, greedy=False):
    """Self-play rollout. Returns stored transitions + terminal reward per seat."""
    env = E.BatchedOrbitEnv(B, nplayers, episode_steps=T + 2, device=device).reset(seed0=seed0)
    store = []  # list of dict per (step, seat)
    for t in range(T):
        per_seat = []
        moves = []
        for seat in range(nplayers):
            ob = RL.encode_obs(env, seat)
            launch, ai, fi, lp, ent, val, actionable = act(policy, ob, seat, env, greedy=greedy)
            moves.append(RL.decode_moves(env, seat, launch, ai, fi))
            per_seat.append((detach_ob(ob), actionable.detach(), launch.detach(),
                             ai.detach(), fi.detach(), lp.detach(), val.detach()))
        env.step(moves)
        store.append((per_seat, env.scores().detach()))   # scores after this step
    term = RL.rewards(env)               # [B, nplayers]
    return store, term, env


def shaped_rewards(store, term, nplayers, shape_coef=0.5):
    """Dense per-step reward = shape_coef * delta(my_share - best_opp_share) + terminal +-1 at end."""
    T = len(store)
    B = term.shape[0]
    rew = [[None] * nplayers for _ in range(T)]
    prev = [None] * nplayers
    for t in range(T):
        sc = store[t][1]                                  # [B,np] scores
        tot = sc.sum(1, keepdim=True).clamp(min=1)
        share = sc / tot
        for seat in range(nplayers):
            others = torch.cat([share[:, :seat], share[:, seat + 1:]], 1).max(1).values
            margin = share[:, seat] - others
            r = torch.zeros(B) if prev[seat] is None else shape_coef * (margin - prev[seat])
            if t == T - 1:
                r = r + term[:, seat]
            rew[t][seat] = r
            prev[seat] = margin
    return rew


def compute_gae(store, rew, nplayers, gamma=0.99, lam=0.95):
    """Dense per-step rewards `rew[t][seat]`. Returns flat tensors over (T*seat*B)."""
    T = len(store)
    obs_all, act_mask, launch_all, ai_all, fi_all, lp_all, val_all, adv_all, ret_all = \
        [], [], [], [], [], [], [], [], []
    for seat in range(nplayers):
        vals = [store[t][0][seat][6] for t in range(T)]
        B = vals[0].shape[0]
        adv = torch.zeros(B)
        nextval = torch.zeros(B)
        advs = [None] * T
        for t in reversed(range(T)):
            nonterm = 0.0 if t == T - 1 else 1.0
            delta = rew[t][seat] + gamma * nextval * nonterm - vals[t]
            adv = delta + gamma * lam * nonterm * adv
            advs[t] = adv.clone()
            nextval = vals[t]
        for t in range(T):
            ob, am, la, ai, fi, lp, val = store[t][0][seat]
            obs_all.append(ob); act_mask.append(am); launch_all.append(la)
            ai_all.append(ai); fi_all.append(fi); lp_all.append(lp)
            val_all.append(val); adv_all.append(advs[t]); ret_all.append(advs[t] + val)
    return obs_all, act_mask, launch_all, ai_all, fi_all, lp_all, val_all, adv_all, ret_all


def cat_obs(obs_list):
    keys = ["planet_feats", "planet_mask", "fleet_feats", "fleet_mask", "global_feats"]
    return {k: torch.cat([o[k] for o in obs_list], 0) for k in keys}


def evaluate(policy, B, T, nplayers, seed0, device, opponent=None):
    """seat 0 = policy (greedy) vs other seats = opponent policy (greedy) or 10%-random.
    Returns seat-0 winrate."""
    env = E.BatchedOrbitEnv(B, nplayers, episode_steps=T + 2, device=device).reset(seed0=seed0)
    for t in range(T):
        moves = []
        for seat in range(nplayers):
            pol = policy if seat == 0 else opponent
            if pol is not None:
                ob = RL.encode_obs(env, seat)
                with torch.no_grad():
                    launch, ai, fi, *_ = act(pol, ob, seat, env, greedy=True)
                moves.append(RL.decode_moves(env, seat, launch, ai, fi))
            else:
                P = env.planets.shape[1]
                owner = env.planets[:, :, 1].long(); ships = env.planets[:, :, 5]
                mine = (owner == seat) & env.p_alive & (ships > 0) & (torch.rand(B, P, device=device) < 0.1)
                ai = torch.randint(0, RL.N_ANGLE, (B, P), device=device)
                fi = torch.randint(0, RL.N_FRAC, (B, P), device=device)
                moves.append(RL.decode_moves(env, seat, mine, ai, fi))
        env.step(moves)
    s = env.scores()
    win = (s[:, 0] >= s.max(1).values) & (s[:, 0] > 0)
    return win.float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--T", type=int, default=60)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.manual_seed(0)
    dev = args.device
    import copy
    policy = Policy().to(dev)
    frozen = copy.deepcopy(policy)            # snapshot of the untrained policy
    opt = optim.Adam(policy.parameters(), lr=args.lr)
    print(f"params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"baseline vs random: {evaluate(policy, 128, args.T, 2, 999000, dev):.2f}")
    for it in range(args.iters):
        t0 = time.time()
        store, term, _ = rollout(policy, args.B, args.T, 2, seed0=it * args.B, device=dev)
        rew = shaped_rewards(store, term, 2)
        obs_l, am_l, la_l, ai_l, fi_l, lp_l, val_l, adv_l, ret_l = compute_gae(store, rew, 2)
        obs = cat_obs(obs_l)
        am = torch.cat(am_l); la = torch.cat(la_l); ai = torch.cat(ai_l); fi = torch.cat(fi_l)
        old_lp = torch.cat(lp_l); adv = torch.cat(adv_l); ret = torch.cat(ret_l)
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        N = old_lp.shape[0]
        for _ in range(args.epochs):
            idx = torch.randperm(N, device=dev)
            for s in range(0, N, 4096):
                mb = idx[s:s + 4096]
                ob_mb = {k: v[mb] for k, v in obs.items()}
                lp, ent, val = logprob_value(policy, ob_mb, am[mb], la[mb], ai[mb], fi[mb])
                ratio = torch.exp(lp - old_lp[mb])
                a = adv[mb]
                l1 = ratio * a
                l2 = torch.clamp(ratio, 0.8, 1.2) * a
                pol_loss = -torch.min(l1, l2).mean()
                val_loss = 0.5 * (val - ret[mb]).pow(2).mean()
                loss = pol_loss + 0.5 * val_loss - 0.01 * ent.mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
                opt.step()
        wr_frozen = evaluate(policy, 128, args.T, 2, 999000, dev, opponent=frozen)
        wr_rand = evaluate(policy, 128, args.T, 2, 777000, dev, opponent=None)
        print(f"it {it:>3} | vs_frozen {wr_frozen:.2f} | vs_random {wr_rand:.2f} | "
              f"loss {loss.item():.3f} | {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
