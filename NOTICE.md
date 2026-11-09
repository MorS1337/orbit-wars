# Attribution / NOTICE

This repository is a competition artifact for **Kaggle Orbit Wars** (2026).

## Game and environment

- Orbit Wars game rules, engine, and `kaggle-environments` are the property of their respective
  authors and the competition organizers.
- The frozen rules reference used by this project is in [`docs/GAME_RULES.md`](docs/GAME_RULES.md).

## Public baselines and prior art

The final agent is a derivative/refactor of the open-source Kaggle baseline **The Producer** by
Slawek Biel:

- https://www.kaggle.com/code/slawekbiel/the-producer-agent

We also studied and borrowed architectural ideas from these public notebooks:

- Vickimar, *Orbit Wars heuristic (LB 1110+)*:
  https://www.kaggle.com/code/vickimar/orbit-wars-heuristic-lb-1110
- Roman Tamrazov, *Orbit Star Wars (LB max 1224)*:
  https://www.kaggle.com/code/romantamrazov/orbit-star-wars-lb-max-1224
- Aiden Song, *Search + learned value function*:
  https://www.kaggle.com/code/aidensong123/lb-highest-1000-search-learned-value-function
- Konbu, *Rule-base + ML shot validator*:
  https://www.kaggle.com/code/konbu17/orbit-wars-rule-base-ml-shot-validator-hybrid

The authors of those notebooks retain all rights to their code and ideas. Their public solutions
were used as starting points and reference material, not as a claim of original authorship by
this team.

## Images

- `assets/orbit-wars-header.png` — official Orbit Wars competition header art, taken from the
  Kaggle competition page and used here for identification only. Copyright belongs to Kaggle /
  the competition organizers.

## Original work in this repository

The following components were written/refactored by the team for this competition:

- the final `v45_turtle` configuration and its 2P/4P presets;
- the ETA-aware reinforcement-risk capture floor and its integration into candidate sizing;
- multi-size commit tiers and the greedy selection/role-mutex logic around them;
- the pressure-gradient regroup logic and its integration with `safe_drain`;
- the batched `sim/` simulator, parity harness, and oracle dumps;
- the field-gate / bias-cancelled duel evaluation methodology;
- the replay-mining, SPSA, and packaging pipelines;
- the adaptive 2P experiment in `experimental/v46_adaptive`.

If you reuse this repository, keep this notice and the links to the original public baselines.

## No license for third-party code

This repository does not relicense third-party code. The presence of derived baseline components
here is for reproducibility of a Kaggle competition result. For any commercial or redistributed
use, check the original notebooks and Kaggle terms first.
