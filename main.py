
from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass

# Make the sibling ``orbit_lite`` package importable wherever this file runs:
# loaded in place, dropped at a submission-archive root, or exec'd by
# kaggle_environments with no ``__file__`` (fall back to the working dir).
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch import Tensor

from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import (
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _plan_regroup,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves


@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs.  """

    
    # the projection window, the movement build length, AND the target ETA cap 
    horizon: int = 19
    # --- shortlists ------------------------------------------------------
    max_sources_per_lane: int = 16
    max_offensive_targets: int = 15         # enemy/neutral proximity targets
    max_defensive_targets: int = 5          
    # --- scoring / greedy ------------------------------------------------
    max_waves_per_turn: int = 8
    roi_threshold: float = 2.1              # fire if score > this
    min_ships_to_launch: float = 9.0
    # --- ETA-aware reinforcement risk (capture sizing) -------------------
    # Inflate the capture floor by ``reinforce_size_beta * rho(eta) * C_k`` where
    # C_k is enemy supply reachable to the target during the fleet's flight, so the
    # agent *declines* captures the enemy will reinforce mid-flight instead of
    # sinking its whole garrison into a doomed attack (the flow scorer projects
    # opponents do-nothing, so it can't see reactive reinforcement). ``beta = 0``
    # disables it (bare floor).
    reinforce_size_beta: float = 3.5
    reinforce_eta_free: float = 8.000
    reinforce_eta_scale: float = 4.215
    # --- multi-size commit tiers (exp41 lever) ---------------------------
    # Evaluate several commit fractions of the safe garrison drain per
    # (source, target) instead of a single all-in size, then let the greedy
    # scorer pick the best tier. Smaller fractions free ships for more waves /
    # parallel fronts (the edge that shines in crowded 4P maps).
    size_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0)
    # --- regroup  ------------------------------
    enable_regroup: bool = True
    max_regroup_time: float = 6.528
    regroup_pressure_delta_min: float = 0.244
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 7
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3


def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    """MovementConfig: fleet tracking on, horizon = config.horizon."""
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    """Cheap reachable-enemy-mass proxy per planet — ``[P]``.

    Consumed only as the **regroup gradient** (rank owned planets by how stressed
    they are, move ships up the gradient). For each planet ``t``, sums a
    distance-decayed share of every enemy source's **current** garrison that could
    straight-line reach ``t`` within ``horizon`` turns, using the step-0 centre
    distance ``cross_dist[0]``. The decay ``(1 - d/(speed·H))₊`` weights nearer
    enemies more, giving a graded frontline signal in ship-mass units.

    Approximations: ignores target orbital drift over the horizon, production
    accrued in flight, the per-owner split, and in-flight enemy fleets. Pure
    arithmetic on cached tensors
    """
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)                                   # [src, tgt] current centre dist
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))                          # [P]
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)    # [src, 1]
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))  # [P]
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye              # [src, tgt]
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)                       # nearer enemy -> heavier
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)                                            # [P] summed over sources


def _tier_candidates(
    *,
    movement: PlanetMovement,
    source_idx: Tensor,
    source_exists: Tensor,
    target_idx: Tensor,
    target_exists: Tensor,
    target_is_mine: Tensor,
    drain: Tensor,
    floor: Tensor,
    eta_cap: Tensor,
    size_mult: float,
    S: int,
    T: int,
    pid: int,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    player_count: int,
    device,
    dtype,
):
    """Score one commit-fraction tier of the (source, target) grid.

    Identical to the v33 single-size body, but the fleet size is
    ``drain * size_mult`` (floored, >=1). The capture floor — which carries the
    producer-v2 reinforcement margin — is size-independent and passed in, so each
    tier still respects reactive-reinforcement sizing.
    """
    sizes = (drain.view(S, 1) * float(size_mult)).floor().clamp(min=1.0).expand(S, T)
    K = int(floor.shape[-1])

    active = reachable_mask(
        movement, source_idx=source_idx, target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
    ).squeeze(-1)
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),
        target_idx.unsqueeze(0),
        sizes,
        active=active,
    )
    angle = aim["angle"]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))

    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable & clears_floor & (sizes >= 1.0) & src_neq_tgt
        & source_exists.view(S, 1) & target_exists.view(1, T)
    )

    L = 1
    C = S * T
    cand_src = source_idx.view(S, 1).expand(S, T).reshape(C, L)
    cand_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(C)
    cand_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C)
    cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cand_angle = angle.reshape(C, L)
    cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    cand_active = valid.reshape(C, L)
    cand_valid = valid.reshape(C)
    cand_is_def = target_is_mine[cand_tgt_short]

    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    score = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launches, player_id=pid,
    )
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))
    return (cand_src, cand_send, cand_angle, cand_eta, cand_active,
            cand_tgt_slot, cand_tgt_short, cand_is_def, score)


def plan_lite_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
):
    """Multi-size, single-source attack planner + regroup.

    Builds exactly one candidate per ``(source, target)`` shortlist pair — fleet
    size = the source's max garrison launch (``safe_drain``) — scores them with the
    exact competitive flow diff, and greedily fires the best wave per target up to
    ``max_waves_per_turn``. Returns the combined ``LaunchEntries`` (attack waves ++
    regroup).
    """
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)

    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))

    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)

    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]                       # [T]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)                # [S]
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(
        garrison_status, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )                                                                            # [S]

    # Uniform reach cap = K_eta (= horizon).
    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)          # [T]

    # Reachable-enemy-mass proxy ([P]) — computed ONCE and reused for BOTH the
    # reinforcement-risk floor margin (below) and the regroup gradient (further
    # down). Its decay distance-scale is the attack reach K_eta.
    beta = float(config.reinforce_size_beta)
    enemy_mass = (
        cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)  # [P]
        if beta > 0.0 or bool(config.enable_regroup) else None
    )

    # ETA-aware reinforcement risk: inflate the capture floor by ``beta * rho(k) *
    # reachable-enemy-mass(target)``. The per-arrival-turn growth comes from the
    # rho(k) timing ramp. Gated by beta > 0.
    reinforcement = None
    if beta > 0.0:
        enemy_mass_t = enemy_mass[target_idx.clamp(0, P - 1)]                     # [T]
        k_arange = torch.arange(1, K_eta + 1, device=device, dtype=dtype)
        rho = reinforcement_timing_factor(
            k_arange, eta_free=float(config.reinforce_eta_free),
            eta_scale=float(config.reinforce_eta_scale),
        )                                                                        # [K_eta]
        reinforcement = beta * rho.view(1, K_eta) * enemy_mass_t.view(T, 1)       # [T, K_eta]
    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid,
        reinforcement=reinforcement,
    )                                                                            # [T, K]
    K = int(floor.shape[-1])

    # --- multi-size: evaluate each commit-fraction tier and pool candidates -----
    # Each tier reuses the same reinforcement-aware ``floor`` (size-independent);
    # the greedy selector then picks the best (source, target, size) across tiers.
    tier_parts = [
        _tier_candidates(
            movement=movement,
            source_idx=source_idx,
            source_exists=source_exists,
            target_idx=target_idx,
            target_exists=target_exists,
            target_is_mine=target_is_mine,
            drain=drain,
            floor=floor,
            eta_cap=eta_cap,
            size_mult=float(mult),
            S=S,
            T=T,
            pid=pid,
            garrison_status=garrison_status,
            prod=prod,
            alive_by_step=alive_by_step,
            player_count=player_count,
            device=device,
            dtype=dtype,
        )
        for mult in config.size_multipliers
    ]
    cand_src = torch.cat([p[0] for p in tier_parts], dim=0)
    cand_send = torch.cat([p[1] for p in tier_parts], dim=0)
    cand_angle = torch.cat([p[2] for p in tier_parts], dim=0)
    cand_eta = torch.cat([p[3] for p in tier_parts], dim=0)
    cand_active = torch.cat([p[4] for p in tier_parts], dim=0)
    cand_tgt_slot = torch.cat([p[5] for p in tier_parts], dim=0)
    cand_tgt_short = torch.cat([p[6] for p in tier_parts], dim=0)
    cand_is_def = torch.cat([p[7] for p in tier_parts], dim=0)
    score = torch.cat([p[8] for p in tier_parts], dim=0)

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
    )

    if not bool(config.enable_regroup):
        return wave_entries

    # Reuse the enemy-mass proxy already computed above (one [P, P] reduction
    # serves both the reinforcement floor and this regroup gradient).
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    """Full per-turn pipeline: build movement → plan single-size waves + regroup → emit.

    ``memory`` must expose a mutable ``movement`` attribute (the rolling cache).
    """
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)

    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=int(player_count)),
        cached_movement=getattr(memory, "movement", None),
    )
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]

    entries = plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=config, player_count=int(player_count),
    )
    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement, launches=launches, owner_id=int(obs.player_id),
        obs_tensors=obs_tensors,
    )
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)


# 4P FFA preset — only the knobs that differ from the 2P default. 
CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    # The SPSA tuning above is 2P-only; restore producer-v2's stock values for
    # the tuned knobs so 4P plays exactly stock v2 (the 2P-aggressive config
    # regressed 4P to 27% vs v2). Then apply v2's original 4P overrides.
    max_offensive_targets=11,
    max_waves_per_turn=8,
    roi_threshold=2.535,
    min_ships_to_launch=7.322,
    reinforce_size_beta=3.337,
    reinforce_eta_free=4.711,
    reinforce_eta_scale=3.877,
    regroup_pressure_delta_min=0.102,
    horizon=16,
    max_sources_per_lane=7,
    max_defensive_targets=1,
    max_regroup_time=5.206,
    max_regroup_targets_per_source=8,
    # multi-size is a 2P-only edge: direct duels show it HELPS 2P (64% vs v33)
    # but REGRESSES 4P (17% vs v33). Restore single-size for 4P so this config
    # plays exactly like v33's 4P (which already ties exp41 in 4P).
    size_multipliers=(1.0,),
)


def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else ProducerLiteConfig()


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        config = _config_for(mem.cached_player_count)
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def agent(obs):
    """Single-observation entry point for local play and Kaggle."""
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)

