# Batched simulator

This directory contains a PyTorch port of the Orbit Wars tick dynamics used for local
experiments. It is intentionally `float64`: the goal was bit-parity with the real engine on the
pre-comet window, not speed at the cost of correctness.

## Files

| File | Purpose |
|---|---|
| `orbit_batch.py` | Batched `interpreter()` port: production, fleet movement, swept collisions, rotation, combat. |
| `env.py` | Batched environment wrapper: reset from the real engine generators, step, termination. |
| `oracle.py` | Dumps ground-truth transitions from the real `kaggle_environments` engine. |
| `parity.py` | Compares `batched_step` against the oracle dumps. |
| `aim.py` | Batched intercept aiming used by the RL/self-play line. |
| `policy.py`, `policy_tp.py` | Small per-planet actor-critic policies from the RL line. |
| `ppo.py`, `ppo_pool.py`, `rl_interface.py`, `warmstart.py` | PPO / self-play plumbing. |
| `distill_clone.py`, `distill_gen.py` | Behaviour-cloning experiments. |

## Parity check

```bash
cd sim
python parity.py
```

`oracle_dumps/` contains ten real-engine episodes (`ep_1000.pkl` ... `ep_1009.pkl`) recorded
before the comet window. The parity harness is the reason we trusted the simulator enough to run
thousands of games on it.

## What the simulator is not

- It is not a replacement for the real gate. The final promotion decisions were made on
  `kaggle_environments`.
- Comets are not fully modelled in the hot loop; parity is checked on the pre-comet window.
- The RL/BC files are research artifacts. They are kept because the negative results are part of
  the methodology, not because they are used by the final agent.
