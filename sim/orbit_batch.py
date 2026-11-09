"""Batched, vectorized orbit_wars step — torch tensors, float64 for bit-parity.

Port of kaggle_environments orbit_wars `interpreter` per-tick dynamics, minus
map/comet generation (those reuse the real RNG generators at reset time). This
module is the hot loop that must run thousands of envs in parallel on GPU.

State tensors (B = batch of envs):
  planets [B, P, 7]  (id, owner, x, y, radius, ships, prod); owner -1 = neutral
  p_alive [B, P] bool
  fleets  [B, F, 7]  (id, owner, x, y, angle, from_id, ships)
  f_alive [B, F] bool
  init_xy [B, P, 2]  initial planet positions (for rotation)
  av      [B]        angular velocity
  step    [B]        current step (rotation uses this)

Comets are not handled here yet (validated on the pre-comet window first).
"""
from __future__ import annotations

import math
import torch

BOARD = 100.0
CENTER = 50.0
SUN_R = 10.0
ROT_LIMIT = 50.0
LOG1000 = math.log(1000.0)


def _swept_pair_hit(A, B, P0, P1, r):
    """Vectorized swept_pair_hit. A,B,P0,P1: [...,2]; r: [...]. Returns bool [...]
    True iff a point moving A->B and a point moving P0->P1 come within r."""
    d0 = A - P0                       # [...,2]
    dv = (B - A) - (P1 - P0)          # [...,2]
    a = (dv * dv).sum(-1)             # [...]
    b = 2.0 * (d0 * dv).sum(-1)
    c = (d0 * d0).sum(-1) - r * r
    # branch a < 1e-12: static relative motion -> hit iff c <= 0
    static = a < 1e-12
    disc = b * b - 4.0 * a * c
    safe_a = torch.where(a < 1e-12, torch.ones_like(a), a)
    sq = torch.sqrt(torch.clamp(disc, min=0.0))
    t1 = (-b - sq) / (2.0 * safe_a)
    t2 = (-b + sq) / (2.0 * safe_a)
    moving_hit = (disc >= 0.0) & (t2 >= 0.0) & (t1 <= 1.0)
    return torch.where(static, c <= 0.0, moving_hit)


def _point_to_seg_dist(p, v, w):
    """Distance from point p to segment v-w. p,v,w: [...,2] -> [...]."""
    l2 = ((v - w) ** 2).sum(-1)
    t = ((p - v) * (w - v)).sum(-1) / torch.clamp(l2, min=1e-12)
    t = torch.clamp(t, 0.0, 1.0)
    proj = v + t.unsqueeze(-1) * (w - v)
    d = torch.sqrt(((p - proj) ** 2).sum(-1))
    return torch.where(l2 == 0.0, torch.sqrt(((p - v) ** 2).sum(-1)), d)


def batched_step(planets, p_alive, fleets, f_alive, init_xy, av, step,
                 moves, max_speed=6.0):
    """One tick. `moves` is a list (len=num_players) of [B, Amax, 3] tensors
    (from_planet_id, angle, ships); ships<=0 entries are ignored. Returns new
    (planets, p_alive, fleets, f_alive). Pure functional (no in-place on inputs).
    """
    device = planets.device
    B, P, _ = planets.shape
    dtype = planets.dtype
    planets = planets.clone()

    # ---- 0. Fleet launch (vectorized: <=1 launch per planet per player) ----
    # moves[pl] is [B,P,3] aligned to planet slots: (from_id, angle, ships).
    # Each owned planet may spawn at most one fleet this tick.
    owner_col = planets[:, :, 1].long()
    px = planets[:, :, 2]; py = planets[:, :, 3]; pr = planets[:, :, 4]
    fl_list = [fleets]
    al_list = [f_alive]
    for pid_player, mv in enumerate(moves):
        ang = mv[:, :, 1].to(dtype)
        shp = mv[:, :, 2].to(dtype)
        do = (shp > 0) & p_alive & (owner_col == pid_player) & (planets[:, :, 5] >= shp)
        # deduct ships from launching planets
        planets[:, :, 5] = planets[:, :, 5] - torch.where(do, shp, torch.zeros_like(shp))
        sx = px + torch.cos(ang) * (pr + 0.1)
        sy = py + torch.sin(ang) * (pr + 0.1)
        fl = torch.stack([
            torch.zeros_like(sx),                                  # id (unused)
            torch.full_like(sx, float(pid_player)),                # owner
            sx, sy, ang, planets[:, :, 0], shp,                    # from_id = planet id
        ], dim=-1)                                                 # [B,P,7]
        fl_list.append(fl)
        al_list.append(do)
    fleets = torch.cat(fl_list, dim=1)
    f_alive = torch.cat(al_list, dim=1)
    F = fleets.shape[1]

    # ---- 1. Production -----------------------------------------------------
    owned = (planets[:, :, 1].long() != -1) & p_alive
    planets[:, :, 5] = planets[:, :, 5] + torch.where(owned, planets[:, :, 6], torch.zeros_like(planets[:, :, 6]))

    # ---- 2. Planet rotation (old -> new) ----------------------------------
    old_xy = planets[:, :, 2:4].clone()
    dxy = init_xy - CENTER
    r = torch.sqrt((dxy ** 2).sum(-1))                   # [B,P]
    rotating = (r + planets[:, :, 4] < ROT_LIMIT) & p_alive
    init_angle = torch.atan2(dxy[..., 1], dxy[..., 0])
    cur_angle = init_angle + av.unsqueeze(1) * step.unsqueeze(1).to(dtype)
    rot_xy = torch.stack([CENTER + r * torch.cos(cur_angle),
                          CENTER + r * torch.sin(cur_angle)], dim=-1)
    new_xy = torch.where(rotating.unsqueeze(-1), rot_xy, old_xy)

    # ---- 3. Fleet movement + swept collision ------------------------------
    ships = fleets[:, :, 6]
    # clamp min=1.0: real fleets always have >=1 ship (launch requires ships>0,int);
    # padding fleets have ships=0 which would make log negative -> NaN under ^1.5.
    sp = 1.0 + (max_speed - 1.0) * (torch.log(torch.clamp(ships, min=1.0)) / LOG1000) ** 1.5
    sp = torch.clamp(sp, max=max_speed)
    ang = fleets[:, :, 4]
    fl_old = fleets[:, :, 2:4]
    fl_new = fl_old + torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1) * sp.unsqueeze(-1)

    # swept hit [B, F, P]
    A_ = fl_old.unsqueeze(2)            # [B,F,1,2]
    Bn = fl_new.unsqueeze(2)
    P0 = old_xy.unsqueeze(1)            # [B,1,P,2]
    P1 = new_xy.unsqueeze(1)
    rad = planets[:, :, 4].unsqueeze(1)  # [B,1,P]
    hit = _swept_pair_hit(A_, Bn, P0, P1, rad)           # [B,F,P]
    hit = hit & p_alive.unsqueeze(1) & f_alive.unsqueeze(2)
    any_hit = hit.any(2)                                  # [B,F]
    first_planet = torch.argmax(hit.to(torch.int8), dim=2)  # [B,F] first hit idx

    # out of bounds / sun (only matter if no planet hit)
    oob = (fl_new[..., 0] < 0) | (fl_new[..., 0] > BOARD) | (fl_new[..., 1] < 0) | (fl_new[..., 1] > BOARD)
    sun = _point_to_seg_dist(torch.tensor([CENTER, CENTER], dtype=dtype, device=device),
                             fl_old, fl_new) < SUN_R
    removed = any_hit | oob | sun
    # update fleet positions; survivors keep moving
    fleets = fleets.clone()
    fleets[:, :, 2:4] = fl_new
    surv_alive = f_alive & (~removed)

    # ---- 4. Apply planet movement -----------------------------------------
    planets[:, :, 2:4] = new_xy

    # ---- 5. Combat resolution ---------------------------------------------
    np_players = max(2, int(planets[:, :, 1].max().item()) + 1) if planets.numel() else 2
    np_players = max(np_players, len(moves))
    # ships-per-(env,planet,owner) from fleets that hit
    combat = hit & f_alive.unsqueeze(2)                  # [B,F,P] only assign to first? no: a fleet hits one planet
    # restrict each fleet to its FIRST hit planet only
    fp_onehot = torch.zeros_like(hit)
    bf = any_hit.nonzero(as_tuple=False)
    if bf.numel():
        fp_onehot[bf[:, 0], bf[:, 1], first_planet[bf[:, 0], bf[:, 1]]] = True
    assign = fp_onehot                                    # [B,F,P]
    owners = fleets[:, :, 1].long().clamp(min=0)          # [B,F]
    fl_ships = fleets[:, :, 6]
    # scatter into [B,P,np]
    psum = torch.zeros(B, P, np_players, dtype=dtype, device=device)
    bfp = assign.nonzero(as_tuple=False)                  # [(b,f,p)]
    if bfp.numel():
        bb, ff, pp = bfp[:, 0], bfp[:, 1], bfp[:, 2]
        psum.index_put_((bb, pp, owners[bb, ff]), fl_ships[bb, ff], accumulate=True)
    # top two owners
    top2, top2i = torch.topk(psum, k=min(2, np_players), dim=2)   # [B,P,2]
    top_ships = top2[:, :, 0]
    top_owner = top2i[:, :, 0]
    second_ships = top2[:, :, 1] if top2.shape[2] > 1 else torch.zeros_like(top_ships)
    has_combat = psum.sum(2) > 0
    survivor_ships = top_ships - second_ships
    tie = top_ships == second_ships
    survivor_ships = torch.where(tie, torch.zeros_like(survivor_ships), survivor_ships)
    survivor_owner = torch.where(survivor_ships > 0, top_owner, torch.full_like(top_owner, -1))

    pl_owner = planets[:, :, 1].long()
    pl_ships = planets[:, :, 5]
    apply = has_combat & (survivor_ships > 0)
    same = apply & (pl_owner == survivor_owner)
    diff = apply & (pl_owner != survivor_owner)
    # same owner: reinforce
    pl_ships = torch.where(same, pl_ships + survivor_ships, pl_ships)
    # diff owner: subtract, maybe flip
    new_ships_diff = pl_ships - survivor_ships
    flip = diff & (new_ships_diff < 0)
    pl_ships = torch.where(diff, torch.where(new_ships_diff < 0, -new_ships_diff, new_ships_diff), pl_ships)
    pl_owner = torch.where(flip, survivor_owner, pl_owner)
    planets[:, :, 5] = pl_ships
    planets[:, :, 1] = pl_owner.to(dtype)

    return planets, p_alive, fleets, surv_alive
