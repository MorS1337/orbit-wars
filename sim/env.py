"""Batched orbit_wars environment for RL self-play.

reset() reuses the REAL engine generators (generate_planets, home assignment)
on CPU per env, then packs into batched tensors. step() runs the validated
batched_step plus step counter, termination and scoring. Comets: TODO (spawn
via real generate_comet_paths at the comet steps, inject as moving neutrals).
"""
from __future__ import annotations

import math
import os
import random
import sys

import torch

_local_dep = os.path.join(os.path.dirname(__file__), "..", ".codex_deps", "py313")
if os.path.isdir(_local_dep):
    sys.path.insert(0, _local_dep)
from kaggle_environments.envs.orbit_wars.orbit_wars import generate_planets  # noqa: E402

from orbit_batch import batched_step  # noqa: E402

DTYPE = torch.float64


def _init_one(seed, num_agents):
    """Replicate the engine's reset: planets + home assignment. Returns
    (planets list, angular_velocity)."""
    rng = random.Random(seed)
    av = rng.uniform(0.025, 0.05)
    planets = generate_planets(rng)
    num_groups = len(planets) // 4
    if num_groups > 0:
        hg = rng.randint(0, num_groups - 1)
        base = hg * 4
        if num_agents == 2:
            planets[base][1] = 0; planets[base][5] = 10
            planets[base + 3][1] = 1; planets[base + 3][5] = 10
        elif num_agents == 4:
            for j in range(4):
                planets[base + j][1] = j; planets[base + j][5] = 10
    return planets, av


def _compact_fleets(fleets, f_alive, fmax):
    """Pack alive fleets to the front and cap to fmax (keep largest on overflow).
    Returns fixed-size [B, fmax, 7] + [B, fmax] alive mask."""
    B, F, _ = fleets.shape
    # sort key: alive first, then by ships desc (so overflow drops smallest)
    key = f_alive.to(fleets.dtype) * 1e9 + fleets[:, :, 6]
    order = torch.argsort(key, dim=1, descending=True)          # [B,F]
    fleets = torch.gather(fleets, 1, order.unsqueeze(-1).expand(B, F, 7))
    f_alive = torch.gather(f_alive, 1, order)
    if F < fmax:
        pad = fmax - F
        fleets = torch.cat([fleets, torch.zeros(B, pad, 7, dtype=fleets.dtype, device=fleets.device)], 1)
        f_alive = torch.cat([f_alive, torch.zeros(B, pad, dtype=torch.bool, device=fleets.device)], 1)
    else:
        fleets = fleets[:, :fmax].contiguous()
        f_alive = f_alive[:, :fmax].contiguous()
    return fleets, f_alive


class BatchedOrbitEnv:
    def __init__(self, B, num_agents=2, episode_steps=500, device="cpu",
                 max_speed=6.0, fmax=128, dtype=DTYPE):
        self.B = B
        self.np = num_agents
        self.episode_steps = episode_steps
        self.device = device
        self.max_speed = max_speed
        self.fmax = fmax
        self.dtype = dtype

    def reset(self, seed0=0):
        inits = [_init_one(seed0 + b, self.np) for b in range(self.B)]
        P = max(len(p) for p, _ in inits)
        self.P = P
        planets = torch.zeros(self.B, P, 7, dtype=self.dtype, device=self.device)
        p_alive = torch.zeros(self.B, P, dtype=torch.bool, device=self.device)
        init_xy = torch.zeros(self.B, P, 2, dtype=self.dtype, device=self.device)
        av = torch.zeros(self.B, dtype=self.dtype, device=self.device)
        for b, (pl, a) in enumerate(inits):
            av[b] = a
            for j, p in enumerate(pl):
                planets[b, j] = torch.tensor(p, dtype=self.dtype, device=self.device)
                p_alive[b, j] = True
                init_xy[b, j] = torch.tensor([p[2], p[3]], dtype=self.dtype, device=self.device)
        self.planets = planets
        self.p_alive = p_alive
        self.init_xy = init_xy
        self.av = av
        self.fleets = torch.zeros(self.B, self.fmax, 7, dtype=self.dtype, device=self.device)
        self.f_alive = torch.zeros(self.B, self.fmax, dtype=torch.bool, device=self.device)
        self.step_idx = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.done = torch.zeros(self.B, dtype=torch.bool, device=self.device)
        return self

    def step(self, moves, fmax=None):
        """moves: list (len np) of [B, P, 3] tensors."""
        fmax = fmax or self.fmax
        self.planets, self.p_alive, self.fleets, self.f_alive = batched_step(
            self.planets, self.p_alive, self.fleets, self.f_alive,
            self.init_xy, self.av, self.step_idx, moves, max_speed=self.max_speed)
        self.fleets, self.f_alive = _compact_fleets(self.fleets, self.f_alive, fmax)
        self.step_idx = self.step_idx + 1
        # termination: step limit or <=1 alive player
        scores = self.scores()
        alive_players = (scores != 0).sum(1)  # rough liveness proxy
        term = (self.step_idx >= self.episode_steps - 2)
        self.done = self.done | term
        return self

    def scores(self):
        """[B, np] total ships per player (owned planets + fleets)."""
        s = torch.zeros(self.B, self.np, dtype=self.dtype, device=self.device)
        owner = self.planets[:, :, 1].long()
        for pl in range(self.np):
            mask = (owner == pl) & self.p_alive
            s[:, pl] += (self.planets[:, :, 5] * mask).sum(1)
        fo = self.fleets[:, :, 1].long()
        for pl in range(self.np):
            mask = (fo == pl) & self.f_alive
            s[:, pl] += (self.fleets[:, :, 6] * mask).sum(1)
        return s


def _random_moves(env):
    """Cheap random legal-ish moves for benchmarking the rollout."""
    B, P = env.planets.shape[0], env.planets.shape[1]
    moves = []
    for pl in range(env.np):
        ids = env.planets[:, :, 0]
        owner = env.planets[:, :, 1].long()
        ships = env.planets[:, :, 5]
        mine = (owner == pl) & env.p_alive & (ships > 0)
        ang = torch.rand(B, P, dtype=DTYPE, device=env.device) * (2 * math.pi)
        send = torch.where(mine, (ships * 0.5).floor().clamp(min=1), torch.zeros_like(ships))
        mv = torch.stack([ids, ang, send], dim=-1)  # [B,P,3]
        moves.append(mv)
    return moves


if __name__ == "__main__":
    import time
    B = 1024
    env = BatchedOrbitEnv(B, num_agents=2, episode_steps=200).reset(seed0=5000)
    print(f"reset B={B}: P={env.P} planets, sample scores {env.scores()[0].tolist()}")
    t0 = time.time()
    K = 100
    for _ in range(K):
        env.step(_random_moves(env))
    dt = time.time() - t0
    print(f"rolled {K} steps in {dt:.2f}s -> {B*K/dt:,.0f} env-steps/sec; "
          f"mean fleets/env now = {env.f_alive.float().sum().item()/B:.1f}")
    print(f"sample final scores: {env.scores()[0].tolist()}")
