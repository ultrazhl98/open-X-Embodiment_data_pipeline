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
)
from .lerobot_writer import LeRobotV2Writer


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
        ep_iter = iter_episodes(str(raw_dir), max_episodes=max_episodes)
        for ep_idx, ep in enumerate(tqdm(ep_iter, desc=cfg.name)):
            steps = ep["steps"]
            canonical = []
            prev = None
            for raw in steps:
                c = standardize_step(cfg, raw, prev)
                canonical.append(c)
                prev = c
            canonical = finalize_actions_to_delta(canonical, cfg)

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
