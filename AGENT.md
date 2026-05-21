# AGENT.md

Guidance for AI agents working in this repository.

## What this project does

Converts a curated set of **13 OXE (Open X-Embodiment) Franka-Panda subsets**
from their raw RLDS/TFRecord form into **LeRobot v2** datasets, with a unified,
physically-consistent action representation (7D EEF delta: `[Δxyz, Δrpy, gripper]`
in meters + radians), plus tooling to verify the conversion is correct.

## Layout

```
oxe_lerobot/            # the conversion package (the actual deliverable)
  configs.py            # per-dataset metadata: fps, image keys, action_kind, notes
  transforms.py         # per-dataset standardize_fn + finalize-to-delta logic
  rlds_reader.py        # reads local TFRecord shards without a full tfds builder
  lerobot_writer.py     # writes a LeRobot v2 dataset (parquet + videos + meta) on disk
  convert.py            # CLI wiring it together: `python -m oxe_lerobot.convert ...`
scripts/
  download_oxe_samples.sh  # pull one raw shard per dataset
  dump_schemas.py          # inspect raw RLDS feature schemas
  verify_conversion.py     # 4-check correctness report → data/verification/report.html
  oxe_manifest.json        # dataset → source paths/metadata
data/                   # gitignored — raw shards + converted outputs + reports
octo/ , openvla/        # vendored reference repos (NOT part of our code; do not commit/edit)
OXE_Franka抓取数据集分析.xlsx  # source-of-truth notes on each dataset's action/state schema
```

## How the pipeline works

1. `rlds_reader.iter_episodes` decodes raw TFRecord shards under `data/raw/<rlds_name>/0.1.0`.
2. `transforms.standardize_step` maps each raw step into a canonical dict:
   `image_primary`, `image_wrist`/`image_secondary`, `state`, `action`, `language_instruction`.
3. `transforms.finalize_actions_to_delta` turns per-step canonical actions into the
   final 7D delta (look-ahead diff; quaternion-relative rotation; unit normalization).
4. `lerobot_writer.LeRobotV2Writer` writes parquet + meta; action semantics, command→state
   scale, and degenerate-frame filtering stats go into `info.json`.

### Action conversion is the subtle part
Each dataset declares an `action_kind` in `configs.py`. Several datasets ship a
*controller command* (OSC / CALVIN-normalized) whose magnitude differs from physical
motion, so we **derive the action from the state EE pose delta** instead (the
`*_state_delta` kinds). Read the per-dataset `notes` and the `action_kind` docstring at
the top of `configs.py` before touching any conversion logic — the choices are
deliberate and documented.

## Common commands

```bash
# Convert one dataset (smoke test with a few episodes)
python -m oxe_lerobot.convert --dataset viola --max-episodes 5

# Convert everything
python -m oxe_lerobot.convert --all

# Verify correctness → data/verification/report.html
python scripts/verify_conversion.py
```

## Conventions for agents

- **Verify after conversion changes.** Any change to `transforms.py` / `configs.py` /
  `convert.py` should be re-checked with `scripts/verify_conversion.py` — it catches
  axis swaps, sign flips, wrong source fields, and image view mix-ups.
- **Don't commit `data/`** (gitignored — large) or the `octo/`/`openvla/` reference repos.
- **Units are unified across all 13 datasets:** meters + radians, 7D delta. Preserve this
  invariant; if a dataset's raw action is in other units/semantics, normalize it.
- **The Excel doc is the source of truth** for each dataset's raw schema. When in doubt
  about a field's meaning, check it (and the `notes` in `configs.py`) rather than guessing.
- Match the existing commit-message style (imperative, one-line summary of the change).
