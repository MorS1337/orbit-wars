# Tools

Small, deliberately boring evaluation harness. The point is not to produce one impressive number,
but to make a number that survives a different seed block and a different opponent pool.

## Real-engine gates

```bash
# Bias-cancelled head-to-head: each seed is played from both slots.
python tools/realduel.py --a main.py --b random --players 2 --seeds 8 --jobs 4

# Mixed 4P pool: candidate vs rotating opponents, reports strict score/rank.
# 4P needs at least 3 opponents; repeat a built-in name or pass different files.
python tools/evaluate_pool.py --agent main.py --opponents random random random --players 4 --episodes 8

# Resumable field gate: 2P round-robin / 4P mixed pool, CSV checkpoints.
python tools/field_gate.py --help
```

`tools/realduel.py` uses `kaggle_environments` directly. `tools/evaluate.py` and
`tools/evaluate_pool.py` share the same loader and rotate seats.

## Replay and leaderboard mining

```bash
python tools/fetch_top_replays.py --help
python tools/analyze_replays.py --help
python tools/lb_target.py --help
python tools/watch_lb.py --help
```

## Historical tuning scripts

`tools/spsa_4p_field.py`, `tools/spsa_v2_4p_wide.py`, and `tools/spsa_v39.py` are the actual
SPSA tuning scripts used during the campaign. They reference historical agent paths from the
private working tree (`agents/v39_ms2p`, `agents/v2_mirror`, ...), so they are included as
methodology, not as turnkey scripts for this release. The final agent is `main.py` at the
repository root.

## Rule of thumb

A change is not promoted from one lucky duel. We required:

- paired seeds and both slot assignments;
- two disjoint seed blocks;
- at least two different opponent pools;
- a positive effect larger than the block-to-block spread;
- a smoke test of the packaged artifact in isolation.
