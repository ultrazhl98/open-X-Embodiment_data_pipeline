"""Compare action-derivation *conventions* on the same OXE dataset.

Motivation
----------
Our pipeline derives the EEF delta from the robot *state* (state[t+1]-state[t],
rotation via quaternion-relative -> rotvec), deliberately distrusting the raw
controller command for the "dirty" datasets (taco / nyu / ucsd_pick, ...).

OpenVLA instead *trusts the raw action field* and only re-lays-out + fixes the
gripper. ABot consumes already-converted LeRobot data, so it inherits whatever
convention the upstream (IPEC-COMMUNITY) converter baked in.

This script quantifies how far apart those two conventions land on the SAME
trajectory, so you can see whether "same dataset" really means "same action".

Two comparisons:
  (1) ours (parquet) vs raw-command baseline (the trust-the-field convention),
      reconstructed from the raw RLDS. Default.
  (2) ours (parquet) vs an external LeRobot dataset (e.g. the IPEC version that
      ABot actually eats), aligned by (episode_index, frame_index).
      Enable with --ref-lerobot PATH. Dataset-agnostic.

Usage
-----
  python scripts/compare_action_conventions.py --dataset taco_play
  python scripts/compare_action_conventions.py --dataset taco_play \
         --ref-lerobot /path/to/IPEC/taco_play_lerobot
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oxe_lerobot.configs import get_config  # noqa: E402
from oxe_lerobot.rlds_reader import iter_episodes  # noqa: E402
from oxe_lerobot.transforms import (  # noqa: E402
    _ensure_1d,
    _euler_xyz_to_rotvec,
)


# ----------------------------- rotation utils ----------------------------- #

def rotvec_to_quat_wxyz(rv: np.ndarray) -> np.ndarray:
    """axis-angle rotation vector (rad) -> unit quaternion (w, x, y, z)."""
    rv = np.asarray(rv, dtype=np.float64)
    theta = np.linalg.norm(rv)
    if theta < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = rv / theta
    half = theta / 2.0
    return np.concatenate([[np.cos(half)], axis * np.sin(half)])


def geodesic_angle_deg(rv_a: np.ndarray, rv_b: np.ndarray) -> float:
    """Shortest rotation angle (deg) between two rotvec deltas."""
    qa = rotvec_to_quat_wxyz(rv_a)
    qb = rotvec_to_quat_wxyz(rv_b)
    dot = abs(float(np.dot(qa, qb)))
    dot = min(1.0, max(-1.0, dot))
    return float(np.degrees(2.0 * np.arccos(dot)))


# ----------------- raw-command baseline (the "trust-the-field" convention) ----------------- #
# Each fn maps a list of raw RLDS steps -> (N, 7) array [dx,dy,dz, rotvec(3), gripper],
# using ONLY the dataset's raw action field, the way OpenVLA / a naive pipeline would.
# rotation is expressed as a rotvec so it is directly comparable to our output.

def _raw_taco(raw_steps):
    out = []
    for s in raw_steps:
        a = _ensure_1d(s["action"]["rel_actions_world"]).astype(np.float64)  # 7D CALVIN cmd
        out.append(np.concatenate([a[:3], _euler_xyz_to_rotvec(a[3:6]), a[6:7]]))
    return np.asarray(out, dtype=np.float64)


def _raw_nyu(raw_steps):
    out = []
    for s in raw_steps:
        a = _ensure_1d(s["action"]).astype(np.float64)  # 15D controller cmd
        out.append(np.concatenate([a[7:10], _euler_xyz_to_rotvec(a[10:13]), a[13:14]]))
    return np.asarray(out, dtype=np.float64)


def _raw_ucsd_pick(raw_steps):
    out = []
    for s in raw_steps:
        a = _ensure_1d(s["action"]).astype(np.float64)  # 4D: 3 gripper-vel + 1 torque
        out.append(np.array([a[0], a[1], a[2], 0.0, 0.0, 0.0, a[3]]))  # rotation zeroed (naive)
    return np.asarray(out, dtype=np.float64)


RAW_COMMAND_FNS = {
    "taco_play": _raw_taco,
    "nyu_franka_play": _raw_nyu,
    "ucsd_pick_and_place": _raw_ucsd_pick,
}


# ----------------------------- parquet loading ----------------------------- #

def load_parquet_actions(lerobot_dir: Path):
    """Return list of (episode_index, action[N,7], frame_index[N]) ordered by episode."""
    files = sorted(glob.glob(str(lerobot_dir / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {lerobot_dir}/data")
    eps = []
    for f in files:
        t = pq.read_table(f).to_pydict()
        act = np.asarray(t["action"], dtype=np.float64)
        fidx = np.asarray(t["frame_index"], dtype=np.int64) if "frame_index" in t else np.arange(len(act))
        eidx = int(t["episode_index"][0]) if "episode_index" in t else len(eps)
        eps.append((eidx, act, fidx))
    eps.sort(key=lambda x: x[0])
    return eps


# ----------------------------- reporting ----------------------------- #

def report_pair(name_a, name_b, A, B):
    """A, B: (M,7) arrays already aligned (drop last zero frame upstream)."""
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    pos_a, pos_b = A[:, :3], B[:, :3]
    rot_a, rot_b = A[:, 3:6], B[:, 3:6]
    grip_a, grip_b = A[:, 6], B[:, 6]

    print(f"\n=== {name_a}  vs  {name_b}   ({n} steps) ===")

    # position
    print("[position]")
    for i, ax in enumerate("xyz"):
        r = np.corrcoef(pos_a[:, i], pos_b[:, i])[0, 1] if pos_a[:, i].std() > 0 and pos_b[:, i].std() > 0 else float("nan")
        print(f"  d{ax}: corr={r:+.3f}   MAE={np.mean(np.abs(pos_a[:,i]-pos_b[:,i])):.4f}")
    na = np.linalg.norm(pos_a, axis=1)
    nb = np.linalg.norm(pos_b, axis=1)
    mask = na > 1e-6
    if mask.any():
        ratio = np.median(nb[mask] / np.clip(na[mask], 1e-9, None))
        print(f"  ||{name_b}|| / ||{name_a}|| (median scale ratio): {ratio:.2f}x")

    # rotation
    ang = np.array([geodesic_angle_deg(rot_a[i], rot_b[i]) for i in range(n)])
    print("[rotation]  geodesic angle between the two delta-rotations")
    print(f"  mean={ang.mean():.2f}deg  median={np.median(ang):.2f}deg  max={ang.max():.2f}deg")

    # gripper
    same = np.mean((grip_a > 0.5) == (grip_b > 0.5))
    print(f"[gripper]  binary agreement: {100*same:.1f}%   MAE={np.mean(np.abs(grip_a-grip_b)):.4f}")


def cmp_vs_raw_command(dataset, lerobot_dir):
    cfg = get_config(dataset)
    if dataset not in RAW_COMMAND_FNS:
        print(f"[skip] no raw-command baseline defined for '{dataset}' "
              f"(its raw action has no directly-usable command field; "
              f"add one to RAW_COMMAND_FNS to compare).")
        return
    raw_dir = str(ROOT / "data" / "raw" / cfg.rlds_name / "0.1.0")
    raw_eps = [ep["steps"] for ep in iter_episodes(raw_dir)]
    ours = load_parquet_actions(lerobot_dir)

    if len(raw_eps) != len(ours):
        print(f"[warn] episode count differs: raw={len(raw_eps)} parquet={len(ours)}; "
              f"comparing by order up to min().")

    allA, allB = [], []
    for (eidx, our_act, _), raw_steps in zip(ours, raw_eps):
        cmd = RAW_COMMAND_FNS[dataset](raw_steps)
        m = min(len(our_act), len(cmd)) - 1  # drop last (our last frame delta = 0)
        if m <= 0:
            continue
        allA.append(our_act[:m])
        allB.append(cmd[:m])
    A = np.concatenate(allA); B = np.concatenate(allB)
    report_pair("ours(state-derived)", "raw-command(trust-field)", A, B)
    print("\nInterpretation: large position scale-ratio and/or rotation angle means the "
          "raw command is NOT the same physical delta as ours -> trusting the field "
          "(OpenVLA / IPEC-if-it-trusted-it) would feed a different action for this dataset.")


def cmp_vs_ref_lerobot(lerobot_dir, ref_dir):
    """Dataset-agnostic: align by (episode_index, frame_index) and diff action columns."""
    ours = {e: (a, f) for e, a, f in load_parquet_actions(lerobot_dir)}
    ref = {e: (a, f) for e, a, f in load_parquet_actions(ref_dir)}
    common = sorted(set(ours) & set(ref))
    if not common:
        print("[error] no overlapping episode_index between the two datasets.")
        return
    allA, allB = [], []
    for e in common:
        a, fa = ours[e]; b, fb = ref[e]
        # align by frame_index intersection
        fmap = {int(fr): i for i, fr in enumerate(fb)}
        for i, fr in enumerate(fa):
            j = fmap.get(int(fr))
            if j is not None and a.shape[1] == b.shape[1] == 7:
                allA.append(a[i]); allB.append(b[j])
    if not allA:
        print("[error] no aligned frames (schema/frame_index mismatch).")
        return
    A = np.asarray(allA); B = np.asarray(allB)
    report_pair("ours", "ref-lerobot(IPEC)", A, B)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="taco_play")
    ap.add_argument("--lerobot-root", default=str(ROOT / "data" / "lerobot"))
    ap.add_argument("--ref-lerobot", default=None,
                    help="Path to an external LeRobot dataset (e.g. IPEC version) to diff against.")
    args = ap.parse_args()

    lerobot_dir = Path(args.lerobot_root) / args.dataset
    print(f"dataset       : {args.dataset}")
    print(f"ours (parquet): {lerobot_dir}")

    if args.ref_lerobot:
        cmp_vs_ref_lerobot(lerobot_dir, Path(args.ref_lerobot))
    else:
        cmp_vs_raw_command(args.dataset, lerobot_dir)


if __name__ == "__main__":
    main()
