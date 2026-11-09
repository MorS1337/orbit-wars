"""Target-pointer actor-critic. Per source planet: {launch, TARGET planet
(pointer over planets), frac}. The lead angle is solved by aim.intercept_angle
at decode time, so the net learns strategy (which target, how many ships), not
raw geometry — the producer-grade aiming comes for free.
"""
from __future__ import annotations

import torch
from torch import nn

from rl_interface import N_FRAC
from aim import intercept_angle


class PolicyTP(nn.Module):
    def __init__(self, d=128, pdim=11, fdim=11, gdim=3):
        super().__init__()
        self.p_embed = nn.Sequential(nn.Linear(pdim, d), nn.GELU(), nn.LayerNorm(d))
        self.f_embed = nn.Sequential(nn.Linear(fdim, d), nn.GELU(), nn.LayerNorm(d))
        self.null_fleet = nn.Parameter(torch.zeros(1, 1, d))
        self.g_embed = nn.Sequential(nn.Linear(gdim, d), nn.GELU())
        self.attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.trunk = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d),
                                   nn.Linear(d, d), nn.GELU())
        self.head_launch = nn.Linear(d, 1)
        self.head_frac = nn.Linear(d, N_FRAC)
        self.q = nn.Linear(d, d)              # pointer query (source)
        self.k = nn.Linear(d, d)              # pointer key (target)
        self.scale = d ** -0.5
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, ob):
        pe = self.p_embed(ob["planet_feats"]); fe = self.f_embed(ob["fleet_feats"])
        ge = self.g_embed(ob["global_feats"]).unsqueeze(1)
        B = fe.shape[0]
        fe = torch.cat([self.null_fleet.expand(B, 1, -1), fe], 1)
        km = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=fe.device), ~ob["fleet_mask"]], 1)
        ctx, _ = self.attn(pe, fe, fe, key_padding_mask=km)
        h = self.trunk(pe + ctx + ge)                       # [B,P,d]
        launch = self.head_launch(h).squeeze(-1)            # [B,P]
        frac = self.head_frac(h)                            # [B,P,NF]
        ptr = torch.einsum("bid,bjd->bij", self.q(h), self.k(h)) * self.scale   # [B,P,P] source->target
        m = ob["planet_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        val = self.value(pooled).squeeze(-1)
        return launch, ptr, frac, val


def act(policy, ob, seat, env, greedy=False):
    launch_l, ptr, frac_l, val = policy(ob)
    owner = env.planets[:, :, 1].long()
    actionable = (owner == seat) & env.p_alive & (env.planets[:, :, 5] > 0)   # [B,P]
    # valid targets: alive & not self
    B, P = owner.shape
    valid_tgt = env.p_alive.unsqueeze(1).expand(B, P, P).clone()              # [B,src,tgt]
    eye = torch.eye(P, dtype=torch.bool, device=owner.device).unsqueeze(0)
    valid_tgt = valid_tgt & ~eye
    ptr = ptr.masked_fill(~valid_tgt, -1e9)
    p_launch = torch.sigmoid(launch_l)
    t_dist = torch.distributions.Categorical(logits=ptr)
    f_dist = torch.distributions.Categorical(logits=frac_l)
    if greedy:
        launch = (p_launch > 0.5) & actionable
        tgt = ptr.argmax(-1)
        fi = frac_l.argmax(-1)
    else:
        launch = (torch.rand_like(p_launch) < p_launch) & actionable
        tgt = t_dist.sample()
        fi = f_dist.sample()
    lp_l = torch.where(launch, torch.log(p_launch.clamp(1e-6)), torch.log((1 - p_launch).clamp(1e-6)))
    lp = lp_l + launch.to(val.dtype) * (t_dist.log_prob(tgt) + f_dist.log_prob(fi))
    lp = (lp * actionable.to(val.dtype)).sum(1)
    ent_l = -(p_launch * torch.log(p_launch.clamp(1e-6)) + (1 - p_launch) * torch.log((1 - p_launch).clamp(1e-6)))
    ent = ((ent_l + t_dist.entropy() + f_dist.entropy()) * actionable.to(val.dtype)).sum(1)
    return launch, tgt, fi, lp, ent, val, actionable


def logprob_value(policy, ob, actionable, launch, tgt, fi):
    launch_l, ptr, frac_l, val = policy(ob)
    B, P = launch_l.shape
    valid_tgt = ob["planet_mask"].unsqueeze(1).expand(B, P, P).clone()
    eye = torch.eye(P, dtype=torch.bool, device=launch_l.device).unsqueeze(0)
    ptr = ptr.masked_fill(~(valid_tgt & ~eye), -1e9)
    p_launch = torch.sigmoid(launch_l)
    t_dist = torch.distributions.Categorical(logits=ptr)
    f_dist = torch.distributions.Categorical(logits=frac_l)
    lp_l = torch.where(launch, torch.log(p_launch.clamp(1e-6)), torch.log((1 - p_launch).clamp(1e-6)))
    lp = lp_l + launch.to(val.dtype) * (t_dist.log_prob(tgt) + f_dist.log_prob(fi))
    lp = (lp * actionable.to(val.dtype)).sum(1)
    ent_l = -(p_launch * torch.log(p_launch.clamp(1e-6)) + (1 - p_launch) * torch.log((1 - p_launch).clamp(1e-6)))
    ent = ((ent_l + t_dist.entropy() + f_dist.entropy()) * actionable.to(val.dtype)).sum(1)
    return lp, ent, val


def decode_moves(env, seat, launch, tgt, fi):
    """target-pointer + intercept -> [B,P,3] moves (from_id, angle, ships)."""
    from rl_interface import FRACS
    B, P = env.planets.shape[0], env.planets.shape[1]
    dev = env.device
    owner = env.planets[:, :, 1].long()
    ships = env.planets[:, :, 5]
    legal = (owner == seat) & env.p_alive & (ships > 0) & launch
    src_xy = env.planets[:, :, 2:4]                                  # [B,P,2]
    # gather target init_xy / radius per (b, src) using tgt index
    bidx = torch.arange(B, device=dev).unsqueeze(1).expand(B, P)
    tgt_init = env.init_xy[bidx, tgt]                                # [B,P,2]
    tgt_rad = env.planets[bidx, tgt, 4]                             # [B,P]
    fr = torch.tensor(FRACS, device=dev, dtype=env.planets.dtype)[fi.clamp(0, len(FRACS) - 1)]
    send = torch.where(legal, (ships * fr).floor().clamp(min=1), torch.zeros_like(ships))
    ang = intercept_angle(src_xy, tgt_init, tgt_rad,
                          env.av.unsqueeze(1).expand(B, P), env.step_idx.unsqueeze(1).expand(B, P),
                          send, max_speed=env.max_speed)
    return torch.stack([env.planets[:, :, 0], ang, send], dim=-1)


if __name__ == "__main__":
    import env as E, rl_interface as RL
    torch.manual_seed(0)
    e = E.BatchedOrbitEnv(8, 2, 60, dtype=torch.float32).reset(seed0=5000)
    pol = PolicyTP()
    for t in range(60):
        mv = []
        for seat in range(2):
            ob = RL.encode_obs(e, seat)
            launch, tgt, fi, lp, ent, val, am = act(pol, ob, seat, e)
            mv.append(decode_moves(e, seat, launch, tgt, fi))
        e.step(mv)
    print("target-pointer policy smoke OK | final scores", e.scores()[0].tolist(),
          "| params", sum(p.numel() for p in pol.parameters()))
