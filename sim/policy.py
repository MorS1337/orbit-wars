"""Small per-planet actor-critic policy for Orbit Wars self-play RL.

Input: planet tokens [B,P,11], fleet tokens [B,F,11], global [B,3].
Per-planet heads: launch (1), angle (16 bins), frac (4 bins). Value: scalar.
Kept small (Kiwis reached gold with ~500K params); ~250K here.
"""
from __future__ import annotations

import torch
from torch import nn

from rl_interface import N_ANGLE, N_FRAC


class Policy(nn.Module):
    def __init__(self, d=128, pdim=11, fdim=11, gdim=3):
        super().__init__()
        self.p_embed = nn.Sequential(nn.Linear(pdim, d), nn.GELU(), nn.LayerNorm(d))
        self.f_embed = nn.Sequential(nn.Linear(fdim, d), nn.GELU(), nn.LayerNorm(d))
        self.null_fleet = nn.Parameter(torch.zeros(1, 1, d))   # always-valid key (avoids all-masked softmax NaN)
        self.g_embed = nn.Sequential(nn.Linear(gdim, d), nn.GELU())
        self.attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.trunk = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d),
                                   nn.Linear(d, d), nn.GELU())
        self.head_launch = nn.Linear(d, 1)
        self.head_angle = nn.Linear(d, N_ANGLE)
        self.head_frac = nn.Linear(d, N_FRAC)
        self.value = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, ob):
        pe = self.p_embed(ob["planet_feats"])          # [B,P,d]
        fe = self.f_embed(ob["fleet_feats"])           # [B,F,d]
        ge = self.g_embed(ob["global_feats"]).unsqueeze(1)  # [B,1,d]
        # prepend an always-valid null fleet token so no query has all keys masked
        B = fe.shape[0]
        fe = torch.cat([self.null_fleet.expand(B, 1, -1), fe], dim=1)   # [B,1+F,d]
        fmask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=fe.device),
                           ~ob["fleet_mask"]], dim=1)   # [B,1+F], col0 valid
        ctx, _ = self.attn(pe, fe, fe, key_padding_mask=fmask)
        h = self.trunk(pe + ctx + ge)                  # [B,P,d]
        launch = self.head_launch(h).squeeze(-1)       # [B,P]
        angle = self.head_angle(h)                     # [B,P,16]
        frac = self.head_frac(h)                       # [B,P,4]
        # value: mean over alive planets + global
        m = ob["planet_mask"].unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        val = self.value(pooled).squeeze(-1)           # [B]
        return launch, angle, frac, val


def act(policy, ob, seat, env, greedy=False):
    """Sample actions; return (launch_bool, angle_idx, frac_idx, logprob[B], entropy[B], value[B])."""
    launch_l, angle_l, frac_l, val = policy(ob)
    owner = env.planets[:, :, 1].long()
    actionable = (owner == seat) & env.p_alive & (env.planets[:, :, 5] > 0)   # [B,P]
    p_launch = torch.sigmoid(launch_l)
    a_dist = torch.distributions.Categorical(logits=angle_l)
    f_dist = torch.distributions.Categorical(logits=frac_l)
    if greedy:
        launch = (p_launch > 0.5) & actionable
        a_idx = angle_l.argmax(-1)
        f_idx = frac_l.argmax(-1)
    else:
        launch = (torch.rand_like(p_launch) < p_launch) & actionable
        a_idx = a_dist.sample()
        f_idx = f_dist.sample()
    # per-planet logprob: launch bernoulli + (if launched) angle+frac
    lp_launch = torch.where(launch, torch.log(p_launch.clamp(1e-6)),
                            torch.log((1 - p_launch).clamp(1e-6)))
    lp = lp_launch + launch.to(val.dtype) * (a_dist.log_prob(a_idx) + f_dist.log_prob(f_idx))
    lp = (lp * actionable.to(val.dtype)).sum(1)        # joint over actionable planets
    ent_l = -(p_launch * torch.log(p_launch.clamp(1e-6)) + (1 - p_launch) * torch.log((1 - p_launch).clamp(1e-6)))
    ent = ((ent_l + a_dist.entropy() + f_dist.entropy()) * actionable.to(val.dtype)).sum(1)
    return launch, a_idx, f_idx, lp, ent, val, actionable


def logprob_value(policy, ob, actionable, launch, a_idx, f_idx):
    """Recompute logprob/entropy/value for stored actions (PPO update)."""
    launch_l, angle_l, frac_l, val = policy(ob)
    p_launch = torch.sigmoid(launch_l)
    a_dist = torch.distributions.Categorical(logits=angle_l)
    f_dist = torch.distributions.Categorical(logits=frac_l)
    lp_launch = torch.where(launch, torch.log(p_launch.clamp(1e-6)),
                            torch.log((1 - p_launch).clamp(1e-6)))
    lp = lp_launch + launch.to(val.dtype) * (a_dist.log_prob(a_idx) + f_dist.log_prob(f_idx))
    lp = (lp * actionable.to(val.dtype)).sum(1)
    ent_l = -(p_launch * torch.log(p_launch.clamp(1e-6)) + (1 - p_launch) * torch.log((1 - p_launch).clamp(1e-6)))
    ent = ((ent_l + a_dist.entropy() + f_dist.entropy()) * actionable.to(val.dtype)).sum(1)
    return lp, ent, val
