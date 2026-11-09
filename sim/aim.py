"""Batched intercept aim for the moonshot sim — give the RL policy producer-grade
aiming for free. The policy picks a TARGET planet; this solves the lead angle so
the fleet hits the rotating target. Self-contained (uses the sim's own state).

Target planet rotates on a circle: pos(t) = C + R*[cos(th0+w*t), sin(th0+w*t)]
(static planets have w=0). Fleet from source travels straight at speed s(ships).
Solve arrival time by fixed-point iteration on t = dist(source, pos(t))/s, then
angle = atan2(pos(t*) - source). Mirrors orbit_lite's t_star approach.
"""
from __future__ import annotations

import math
import torch

CENTER = 50.0
ROT_LIMIT = 50.0
LOG1000 = math.log(1000.0)


def fleet_speed(ships, max_speed=6.0):
    return torch.clamp(1.0 + (max_speed - 1.0) * (torch.log(torch.clamp(ships, min=1.0)) / LOG1000) ** 1.5,
                       max=max_speed)


def intercept_angle(src_xy, tgt_init_xy, tgt_radius, av, step, ships, max_speed=6.0, iters=12):
    """All tensors broadcastable to a common shape [...].
      src_xy:      [...,2] source position now
      tgt_init_xy: [...,2] target INITIAL position (for rotation)
      tgt_radius:  [...]  target planet radius
      av:          [...]  angular velocity
      step:        [...]  current step
      ships:       [...]  fleet size (sets speed)
    Returns angle [...] that leads the rotating target.
    """
    dxy = tgt_init_xy - CENTER
    R = torch.sqrt((dxy ** 2).sum(-1))                       # orbit radius
    th0 = torch.atan2(dxy[..., 1], dxy[..., 0])
    rotating = (R + tgt_radius < ROT_LIMIT)
    w = torch.where(rotating, av, torch.zeros_like(av))
    # current target angle already advanced by step
    base = th0 + w * step.to(R.dtype)
    speed = fleet_speed(ships, max_speed)
    sx, sy = src_xy[..., 0], src_xy[..., 1]

    # fixed-point on arrival time t
    # current target pos as t=0 start guess
    tx = CENTER + R * torch.cos(base)
    ty = CENTER + R * torch.sin(base)
    t = torch.sqrt((tx - sx) ** 2 + (ty - sy) ** 2) / speed.clamp(min=1e-6)
    for _ in range(iters):
        ang_t = base + w * t
        tx = CENTER + R * torch.cos(ang_t)
        ty = CENTER + R * torch.sin(ang_t)
        d = torch.sqrt((tx - sx) ** 2 + (ty - sy) ** 2)
        t = d / speed.clamp(min=1e-6)
    return torch.atan2(ty - sy, tx - sx)


if __name__ == "__main__":
    # HIT TEST: aim a fleet at a rotating target in the real sim, check it collides.
    import env as E
    from orbit_batch import batched_step
    torch.manual_seed(0)
    B = 64
    e = E.BatchedOrbitEnv(B, 2, 200, dtype=torch.float64).reset(seed0=7000)
    hits = 0; tries = 0
    for trial in range(40):
        # pick: source = a planet I own with ships; target = nearest enemy/neutral
        owner = e.planets[:, :, 1].long()
        ships = e.planets[:, :, 5]
        mine = (owner == 0) & e.p_alive & (ships >= 20)
        tgt_ok = (owner != 0) & e.p_alive
        moves0 = torch.zeros(B, e.planets.shape[1], 3, dtype=torch.float64)
        moves1 = torch.zeros(B, e.planets.shape[1], 3, dtype=torch.float64)
        for b in range(B):
            mi = mine[b].nonzero().flatten()
            ti = tgt_ok[b].nonzero().flatten()
            if len(mi) == 0 or len(ti) == 0:
                continue
            s = mi[0].item()
            # nearest target
            sp = e.planets[b, s, 2:4]
            tp = e.planets[b, ti, 2:4]
            j = ((tp - sp) ** 2).sum(-1).argmin()
            tj = ti[j].item()
            send = 20.0
            ang = intercept_angle(e.planets[b, s, 2:4], e.init_xy[b, tj], e.planets[b, tj, 4],
                                  e.av[b], e.step_idx[b], torch.tensor(send, dtype=torch.float64))
            moves0[b, s] = torch.tensor([e.planets[b, s, 0], float(ang), send])
            tries += 1
        # record target garrison before, step a few, see if our fleet captured/damaged it
        e.step([moves0, moves1])
        e.step([torch.zeros_like(moves0), torch.zeros_like(moves1)])  # let fleet travel
    print(f"intercept aim built. launched {tries} aimed fleets across {B} envs (smoke; full hit-rate test next)")
