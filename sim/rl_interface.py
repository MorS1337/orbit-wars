"""RL interface for BatchedOrbitEnv: observation encoding, action decoding,
reward. Owner features are RELATIVE to the acting seat so one shared policy can
play any seat (self-play). Action is per-planet: {launch?, angle-bin, frac-bin}.
"""
from __future__ import annotations

import math
import torch

N_ANGLE = 16                      # angle bins (0..2pi)
FRACS = (0.25, 0.5, 0.75, 1.0)    # ship fraction bins
N_FRAC = len(FRACS)
ANGLE_CENTERS = torch.tensor([(k + 0.5) * 2 * math.pi / N_ANGLE for k in range(N_ANGLE)])


def _rel_owner(owner, seat, nplayers):
    """0=neutral, 1=me, 2..=enemies in seat-relative order. owner:[...] long."""
    rel = torch.where(owner < 0, torch.zeros_like(owner),
                      ((owner - seat) % nplayers) + 1)
    return rel.clamp(0, 4)


def encode_obs(env, seat):
    """Returns dict of tensors for the policy (owner-relative to `seat`)."""
    B, P = env.planets.shape[0], env.planets.shape[1]
    dev, dt = env.device, torch.float32
    pl = env.planets
    owner = pl[:, :, 1].long()
    rel = _rel_owner(owner, seat, env.np)               # [B,P]
    oh = torch.zeros(B, P, 5, dtype=dt, device=dev)
    oh.scatter_(2, rel.unsqueeze(-1), 1.0)
    oh = oh * env.p_alive.unsqueeze(-1).to(dt)
    pfeat = torch.stack([
        torch.log1p(pl[:, :, 5].clamp(min=0)).to(dt) / 6.0,   # ships
        pl[:, :, 6].to(dt) / 3.0,                              # prod
        pl[:, :, 2].to(dt) / 100.0,                           # x
        pl[:, :, 3].to(dt) / 100.0,                           # y
        pl[:, :, 4].to(dt) / 5.0,                             # radius
        env.p_alive.to(dt),
    ], dim=-1)
    planet_feats = torch.cat([oh, pfeat], dim=-1)            # [B,P,11]

    F = env.fleets.shape[1]
    fo = env.fleets[:, :, 1].long()
    frel = _rel_owner(fo, seat, env.np)
    foh = torch.zeros(B, F, 5, dtype=dt, device=dev)
    foh.scatter_(2, frel.unsqueeze(-1), 1.0)
    foh = foh * env.f_alive.unsqueeze(-1).to(dt)
    ang = env.fleets[:, :, 4].to(dt)
    ffeat = torch.stack([
        torch.log1p(env.fleets[:, :, 6].clamp(min=0)).to(dt) / 6.0,
        env.fleets[:, :, 2].to(dt) / 100.0,
        env.fleets[:, :, 3].to(dt) / 100.0,
        torch.sin(ang), torch.cos(ang),
        env.f_alive.to(dt),
    ], dim=-1)
    ffeat = ffeat * env.f_alive.unsqueeze(-1).to(dt)        # zero dead/pad slots (no NaN leak)
    fleet_feats = torch.cat([foh, ffeat], dim=-1)           # [B,F,11]

    scores = env.scores()
    my = scores[:, seat]
    tot = scores.sum(1).clamp(min=1)
    glob = torch.stack([
        env.step_idx.to(dt) / 500.0,
        torch.full((B,), env.np / 4.0, dtype=dt, device=dev),
        (my / tot).to(dt),
    ], dim=-1)                                              # [B,3]
    return {
        "planet_feats": planet_feats, "planet_mask": env.p_alive,
        "fleet_feats": fleet_feats, "fleet_mask": env.f_alive,
        "global_feats": glob, "owner": owner, "seat": seat,
    }


def decode_moves(env, seat, launch, angle_idx, frac_idx):
    """Policy outputs -> [B,P,3] moves (from_id, angle, ships) for `seat`.
    launch: bool [B,P]; angle_idx: long [B,P]; frac_idx: long [B,P]."""
    B, P = env.planets.shape[0], env.planets.shape[1]
    dev = env.device
    owner = env.planets[:, :, 1].long()
    ships = env.planets[:, :, 5]
    legal = (owner == seat) & env.p_alive & (ships > 0) & launch
    centers = ANGLE_CENTERS.to(dev, env.planets.dtype)
    ang = centers[angle_idx.clamp(0, N_ANGLE - 1)]
    fr = torch.tensor(FRACS, device=dev, dtype=env.planets.dtype)[frac_idx.clamp(0, N_FRAC - 1)]
    send = torch.where(legal, (ships * fr).floor().clamp(min=1), torch.zeros_like(ships))
    mv = torch.stack([env.planets[:, :, 0], ang, send], dim=-1)   # [B,P,3]
    return mv


def rewards(env):
    """Sparse ±1 per seat at terminal: +1 iff strict-or-tied max score & >0."""
    s = env.scores()                       # [B,np]
    mx = s.max(1, keepdim=True).values
    win = (s == mx) & (mx > 0)
    return torch.where(win, torch.ones_like(s), -torch.ones_like(s))


if __name__ == "__main__":
    import env as E
    e = E.BatchedOrbitEnv(8, num_agents=2, episode_steps=60).reset(seed0=5000)
    for t in range(60):
        moves = []
        for seat in range(e.np):
            ob = encode_obs(e, seat)
            B, P = e.planets.shape[0], e.planets.shape[1]
            launch = torch.rand(B, P) < 0.1
            aidx = torch.randint(0, N_ANGLE, (B, P))
            fidx = torch.randint(0, N_FRAC, (B, P))
            moves.append(decode_moves(e, seat, launch, aidx, fidx))
        e.step(moves, fmax=128)
    print("obs keys:", list(ob.keys()))
    print("planet_feats", ob["planet_feats"].shape, "fleet_feats", ob["fleet_feats"].shape,
          "global", ob["global_feats"].shape)
    print("rewards sample:", rewards(e)[0].tolist(), "| scores", e.scores()[0].tolist())
    print("RL interface smoke OK")
