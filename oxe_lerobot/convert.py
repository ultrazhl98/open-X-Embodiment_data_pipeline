"""End-to-end converter: raw OXE TFRecord shards → LeRobot v2 dataset.

Usage:
  python -m oxe_lerobot.convert --dataset viola --max-episodes 5
  python -m oxe_lerobot.convert --all
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .configs import DATASETS, get_config
from .rlds_reader import iter_episodes
from .transforms import (
    standardize_step,
    finalize_actions_to_delta,
    STATE_DELTA_KINDS,
    _is_degenerate_pose,
)
from .lerobot_writer import LeRobotV2Writer


def _extract_raw_command_6d(cfg, raw_step: dict) -> np.ndarray | None:
    """Pull the original 6D command [dx, dy, dz, drx, dry, drz] from a raw RLDS
    step for a STATE_DELTA-kind dataset. Returns None if the dataset's original
    action is not a 6D delta (e.g. viola needs special handling).

    The scale metadata we record is:
        scale[i] = mean(|raw_command[:, i]|) / mean(|state_delta[:, i]|)
    so the user can later reconstruct: `raw_command ≈ state_delta * scale`.
    """
    name = cfg.name
    if name == "viola":
        a = raw_step["action"]
        wv = np.asarray(a["world_vector"],  dtype=np.float32).reshape(-1)
        rd = np.asarray(a["rotation_delta"], dtype=np.float32).reshape(-1)
        return np.concatenate([wv, rd]).astype(np.float32)  # 6D normalized command
    # All five other STATE_DELTA datasets: raw action is 7D [Δxyz, Δrpy, grip]
    if name in {"stanford_hydra", "austin_buds", "austin_sailor",
                "austin_sirius", "utaustin_mutex"}:
        a = np.asarray(raw_step["action"], dtype=np.float32).reshape(-1)
        return a[:6].astype(np.float32)
    return None


def _compute_command_to_state_scale(
    raw_cmds_6d: np.ndarray, state_deltas_6d: np.ndarray, eps: float = 1e-9
) -> np.ndarray:
    """Per-axis ratio mean|raw_command| / mean|state_delta| over episode 0.
    Returns shape (6,)."""
    num = np.mean(np.abs(raw_cmds_6d), axis=0)
    den = np.mean(np.abs(state_deltas_6d), axis=0) + eps
    return (num / den).astype(np.float64)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def convert_one(
    rlds_name: str,
    raw_root: Path,
    out_root: Path,
    max_episodes: int | None = None,
) -> None:
    cfg = get_config(rlds_name)
    raw_dir = raw_root / cfg.rlds_name / "0.1.0"
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw dir: {raw_dir}")

    # Decide which image keys to materialize for this dataset.
    image_keys: list[str] = ["primary"]
    if cfg.image_wrist_key:
        image_keys.append("wrist")
    if cfg.image_secondary_key:
        image_keys.append("secondary")

    out_dir = out_root / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # First pass: probe state_dim from the first step of the first episode.
    state_dim = None
    img_shapes: dict[str, tuple] = {}
    first_ep = next(iter_episodes(str(raw_dir), max_episodes=1))
    first_canon = standardize_step(cfg, first_ep["steps"][0], None)
    state_dim = int(first_canon["state"].shape[0])
    img_shapes["primary"] = first_canon["image_primary"].shape
    if "wrist" in image_keys:
        img_shapes["wrist"] = first_canon["image_wrist"].shape
    if "secondary" in image_keys:
        img_shapes["secondary"] = first_canon["image_secondary"].shape

    print(f"[{cfg.name}] state_dim={state_dim} image_keys={image_keys} shapes={img_shapes}")

    with LeRobotV2Writer(
        root=str(out_dir),
        repo_id=f"oxe_franka/{cfg.name}",
        fps=cfg.fps,
        robot_type="franka_panda",
        image_keys=tuple(image_keys),
        state_dim=state_dim,
        action_dim=7,
    ) as writer:
        # Track scale (from ep 0) and degenerate-frame filtering across all
        # episodes; metadata is finalized into info.json after the loop.
        scale_payload: dict | None = None
        filter_summary = {
            "total_frames_filtered": 0,
            "episodes_with_filtering": [],  # capped to first 10 for readability
        }

        ep_iter = iter_episodes(str(raw_dir), max_episodes=max_episodes)
        for ep_idx, ep in enumerate(tqdm(ep_iter, desc=cfg.name)):
            steps = ep["steps"]
            canonical = []
            prev = None
            for raw in steps:
                c = standardize_step(cfg, raw, prev)
                canonical.append(c)
                prev = c

            # Data-integrity filter for state-derived datasets: drop frames
            # whose packed EE pose is identically zero. These come from RLDS
            # logging gaps (e.g. austin_sirius/buds have all-zero state_ee
            # matrices on a handful of frames) and would otherwise inject
            # large fake jumps into the training signal.
            degen_indices: list[int] = []
            steps_kept = steps
            if cfg.action_kind in STATE_DELTA_KINDS:
                degen_indices = [
                    t for t, c in enumerate(canonical)
                    if _is_degenerate_pose(np.asarray(c["action"], dtype=np.float64))
                ]
                if degen_indices:
                    drop = set(degen_indices)
                    canonical = [c for i, c in enumerate(canonical) if i not in drop]
                    steps_kept = [s for i, s in enumerate(steps) if i not in drop]
                    filter_summary["total_frames_filtered"] += len(degen_indices)
                    if len(filter_summary["episodes_with_filtering"]) < 10:
                        filter_summary["episodes_with_filtering"].append({
                            "episode_index": ep_idx,
                            "original_length": len(steps),
                            "kept_length": len(canonical),
                            "filtered_indices": degen_indices,
                        })

            canonical = finalize_actions_to_delta(canonical, cfg)

            # First episode of a STATE_DELTA dataset: compute the raw-command →
            # state-delta scale from the kept frames so it's consistent with
            # the data actually written to parquet.
            if ep_idx == 0 and cfg.action_kind in STATE_DELTA_KINDS:
                state_deltas = np.stack([c["action"][:6] for c in canonical])
                raw_cmds = []
                for raw in steps_kept:
                    cmd = _extract_raw_command_6d(cfg, raw)
                    if cmd is not None:
                        raw_cmds.append(cmd)
                if len(raw_cmds) == len(steps_kept):
                    raw_cmds_arr = np.stack(raw_cmds)
                    n = min(len(raw_cmds_arr), len(state_deltas))
                    scale = _compute_command_to_state_scale(
                        raw_cmds_arr[:n - 1], state_deltas[:n - 1]
                    )
                    scale_payload = {
                        "scale": scale,
                        "source": f"estimated from episode 0 ({n - 1} kept frames)",
                    }

            frames = []
            for c in canonical:
                imgs = {"primary": c["image_primary"]}
                if "wrist" in image_keys:
                    imgs["wrist"] = c["image_wrist"]
                if "secondary" in image_keys:
                    imgs["secondary"] = c["image_secondary"]
                frames.append({
                    "action": c["action"],
                    "state":  c["state"],
                    "images": imgs,
                    "language_instruction": c["language_instruction"],
                })
            writer.write_episode(ep_idx, frames)

        # After all episodes: write action_semantics + scale + filter stats
        # into info.json as a single action_metadata block.
        if cfg.action_kind in STATE_DELTA_KINDS:
            action_meta = {
                "definition": (
                    "action[t] = state[t+1] - state[t] (look-ahead, last frame "
                    "zero); rotation via quaternion-relative math; gripper "
                    "passes through raw command (or state for furniture_bench)."
                ),
            }
            if scale_payload is not None:
                scale = scale_payload["scale"]
                action_meta["raw_command_to_state_delta_scale"] = {
                    "xyz": [float(x) for x in scale[:3]],
                    "rpy": [float(x) for x in scale[3:6]],
                    "interpretation": (
                        "raw_command ≈ state_delta * scale (per axis). Useful "
                        "at deployment time to recover OSC command magnitude "
                        "from a model that predicts physical delta. A scale of "
                        "exactly 0 means the corresponding raw-command axis "
                        "was identically zero across episode 0."
                    ),
                }
                action_meta["source"] = scale_payload["source"]
            else:
                action_meta["scale_unavailable_reason"] = (
                    "raw command field is not a directly-comparable 6D pose delta "
                    "for this dataset (e.g. furniture_bench uses quat velocity)."
                )
            action_meta["frame_filtering"] = {
                "criterion": (
                    "frames whose source EE pose (state_ee or state[8:24] homogeneous "
                    "matrix) is identically zero are dropped — these are RLDS logging "
                    "gaps and would otherwise inject large fake jumps into training."
                ),
                "total_frames_filtered": filter_summary["total_frames_filtered"],
                "episodes_with_filtering": filter_summary["episodes_with_filtering"],
                "note": (
                    "details capped to first 10 affected episodes; "
                    "indices are positions in the ORIGINAL raw RLDS episode "
                    "(before filtering); the LeRobot output omits these frames "
                    "entirely so frame_index is contiguous over the kept frames."
                ),
            }
            writer.set_action_metadata({
                "action_semantics": "physical_delta",
                "action_metadata": action_meta,
            })

    print(f"[{cfg.name}] DONE → {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=None,
                   help="rlds_name or short name (see configs.DATASETS). Omit with --all.")
    p.add_argument("--all", action="store_true", help="convert every dataset listed in configs.")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="cap episodes per dataset (useful for smoke tests).")
    p.add_argument("--raw-root", type=str, default=str(PROJECT_ROOT / "data" / "raw"))
    p.add_argument("--out-root", type=str, default=str(PROJECT_ROOT / "data" / "lerobot"))
    args = p.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)

    if args.all:
        names = list(DATASETS.keys())
    elif args.dataset:
        names = [args.dataset]
    else:
        p.error("must pass --dataset or --all")
        return

    for n in names:
        try:
            convert_one(n, raw_root, out_root, max_episodes=args.max_episodes)
        except Exception as e:
            print(f"[FAIL] {n}: {e!r}", file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
