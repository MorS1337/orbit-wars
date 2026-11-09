#!/usr/bin/env python3
"""Stabilized self-play PPO: opponent pool + reward normalization + entropy floor.

Asymmetric: seat 0 = learner (collect+train), seat 1 = frozen opponent sampled
from a pool (75% latest checkpoint, 25% older). This stops the policy from
chasing its own tail (the collapse we saw). Shaped dense reward (ship-share
margin) + terminal. Returns normalized by a running std.
"""
import argparse
import copy
import random
import time
import torch
from torch import optim

import env as E
import rl_interface as RL
from policy import Policy, act, logprob_value


def detach_ob(ob):
    return {k: (v.detach() if torch.is_tensor(v) else v) for k, v in ob.items()}


def rollout(learner, opponent, B, T, seed0, device, shape_coef=0.5):
    """seat0=learner (stored), seat1=opponent (frozen). Returns seat0 transitions
    + dense reward list."""
    env = E.BatchedOrbitEnv(B, 2, episode_steps=T + 2, device=device).reset(seed0=seed0)
    store, scores_hist = [], []
    for t in range(T):
        # seat 0 learner
        ob0 = RL.encode_obs(env, 0)
        launch, ai, fi, lp, ent, val, am = act(learner, ob0, 0, env, greedy=False)
        m0 = RL.decode_moves(env, 0, launch, ai, fi)
        # seat 1 opponent (no grad)
        with torch.no_grad():
            ob1 = RL.encode_obs(env, 1)
            l1, a1, f1, *_ = act(opponent, ob1, 1, env, greedy=False)
        m1 = RL.decode_moves(env, 1, l1, a1, f1)
        env.step([m0, m1])
        store.append((detach_ob(ob0), am.detach(), launch.detach(), ai.detach(),
                      fi.detach(), lp.detach(), val.detach()))
        scores_hist.append(env.scores().detach())
    term = RL.rewards(env)[:, 0]                       # [B] seat-0 terminal
    # dense reward: shape_coef * delta(share0 - share1) + terminal at end
    rew = []
    prev = None
    for t in range(T):
        sc = scores_hist[t]
        tot = sc.sum(1).clamp(min=1)
        margin = (sc[:, 0] - sc[:, 1]) / tot
        r = torch.zeros(B, device=device) if prev is None else shape_coef * (margin - prev)
        if t == T - 1:
            r = r + term
        rew.append(r)
        prev = margin
    return store, rew, env


def gae(store, rew, gamma=0.99, lam=0.95):
    T = len(store)
    vals = [store[t][6] for t in range(T)]
    B = vals[0].shape[0]
    dev = vals[0].device
    adv = torch.zeros(B, device=dev); nextval = torch.zeros(B, device=dev); advs = [None] * T
    for t in reversed(range(T)):
        nonterm = 0.0 if t == T - 1 else 1.0
        delta = rew[t] + gamma * nextval * nonterm - vals[t]
        adv = delta + gamma * lam * nonterm * adv
        advs[t] = adv.clone(); nextval = vals[t]
    obs = [store[t][0] for t in range(T)]
    am = torch.cat([store[t][1] for t in range(T)])
    la = torch.cat([store[t][2] for t in range(T)])
    ai = torch.cat([store[t][3] for t in range(T)])
    fi = torch.cat([store[t][4] for t in range(T)])
    lp = torch.cat([store[t][5] for t in range(T)])
    val = torch.cat(vals)
    adv = torch.cat(advs)
    ret = adv + val
    return obs, am, la, ai, fi, lp, adv, ret


def cat_obs(obs_list):
    keys = ["planet_feats", "planet_mask", "fleet_feats", "fleet_mask", "global_feats"]
    return {k: torch.cat([o[k] for o in obs_list], 0) for k in keys}


def evaluate(policy, B, T, seed0, device, opponent=None):
    env = E.BatchedOrbitEnv(B, 2, episode_steps=T + 2, device=device).reset(seed0=seed0)
    for t in range(T):
        moves = []
        for seat in range(2):
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
    return ((s[:, 0] >= s.max(1).values) & (s[:, 0] > 0)).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--T", type=int, default=60)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ent", type=float, default=0.02)
    ap.add_argument("--pool_every", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save", default="")
    args = ap.parse_args()
    torch.manual_seed(0); random.seed(0)
    dev = args.device
    learner = Policy().to(dev)
    pool = [copy.deepcopy(learner)]
    opt = optim.Adam(learner.parameters(), lr=args.lr)
    ret_std = 1.0
    print(f"params {sum(p.numel() for p in learner.parameters()):,} | vs_random {evaluate(learner,128,args.T,999,dev):.2f}")
    for it in range(args.iters):
        t0 = time.time()
        opp = pool[-1] if random.random() < 0.75 else random.choice(pool)
        store, rew, _ = rollout(learner, opp, args.B, args.T, seed0=it * args.B, device=dev)
        obs_l, am, la, ai, fi, old_lp, adv, ret = gae(store, rew)
        # running return-std normalization
        ret_std = 0.95 * ret_std + 0.05 * ret.std().item()
        adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        obs = cat_obs(obs_l)
        N = old_lp.shape[0]
        for _ in range(args.epochs):
            idx = torch.randperm(N, device=dev)
            for s in range(0, N, 4096):
                mb = idx[s:s + 4096]
                ob_mb = {k: v[mb] for k, v in obs.items()}
                lp, ent, val = logprob_value(learner, ob_mb, am[mb], la[mb], ai[mb], fi[mb])
                ratio = torch.exp(lp - old_lp[mb])
                a = adv[mb]
                pol_loss = -torch.min(ratio * a, torch.clamp(ratio, 0.85, 1.15) * a).mean()
                val_loss = (val - ret[mb] / max(ret_std, 1e-3)).pow(2).mean()
                loss = pol_loss + 0.5 * val_loss - args.ent * ent.mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
                opt.step()
        if (it + 1) % args.pool_every == 0:
            pool.append(copy.deepcopy(learner))
            if len(pool) > 12:
                pool.pop(1)
        wr_pool0 = evaluate(learner, 128, args.T, 999, dev, opponent=pool[0])
        wr_rand = evaluate(learner, 128, args.T, 777, dev, opponent=None)
        print(f"it {it:>3} | vs_init {wr_pool0:.2f} | vs_random {wr_rand:.2f} | pool {len(pool)} | "
              f"loss {loss.item():.3f} | {time.time()-t0:.1f}s", flush=True)
        if args.save and (it + 1) % 10 == 0:
            torch.save({"model": learner.state_dict(), "iter": it,
                        "wr_init": wr_pool0, "wr_rand": wr_rand}, args.save)


if __name__ == "__main__":
    main()
