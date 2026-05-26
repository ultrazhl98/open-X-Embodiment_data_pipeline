"""Verify OXE → LeRobot v2 conversion correctness.

Runs 4 checks per dataset and bundles results into a single HTML report:

  ① Numerical equivalence: re-derive the canonical action straight from the raw
     RLDS step (no abstraction layer) and diff against what's in the parquet.
     This catches "we wired up the wrong field / wrong order" bugs.

  ② Action vs state-EE velocity correlation: for datasets whose state contains
     EE position, the parquet action[:3] should be linearly related to
     diff(state.xyz). Pearson r per axis is reported.  Catches axis swap,
     sign flip, completely wrong source field.

  ③ Action time-series (7 channels) and 3D EE trajectory plots.  Visual sanity
     check for sign/scale.

  ④ Side-by-side primary | wrist GIF.  Hand-eye verification that
     image_primary really is the workspace camera and image_wrist really is
     the in-hand camera (not swapped), and that the action ranges aren't
     wildly inconsistent with what the robot is doing.

Output: data/verification/report.html (self-contained, base64-embedded media).
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oxe_lerobot.configs import get_config  # noqa: E402
from oxe_lerobot.rlds_reader import iter_episodes  # noqa: E402
from oxe_lerobot.transforms import (  # noqa: E402
    ACTION_FNS,
    STATE_DELTA_KINDS,
    _is_degenerate_pose,
    _euler_xyz_to_rotvec,
    _quat_wxyz_to_rotvec,
    euler_xyz_to_quat_wxyz,
    finalize_actions_to_delta,
    quat_relative_wxyz,
    standardize_step,
)
from oxe_lerobot.openvla_transforms import (  # noqa: E402
    openvla_actions_for_trajectory,
    unit_gripper_for_trajectory,
    OPENVLA_ACTION_NOTES,
)


VERIFY = [
    # already vetted in earlier pass
    "viola",
    "iamlab_cmu_pickup_insert",
    "furniture_bench",
    # newly added in this pass
    "stanford_hydra",
    "austin_buds",
    "austin_sailor",
    "austin_sirius",
    "utaustin_mutex",
    "taco_play",
    "nyu_franka_play",
    "ucsd_kitchen",
    "ucsd_pick_and_place",
    "cmu_franka_exploration",
]


# ---------------------------- IO helpers ---------------------------- #

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _gif_b64(frames: list[np.ndarray], fps: int = 5) -> str:
    buf = io.BytesIO()
    duration_ms = int(1000 / fps)  # imageio GIF plugin now wants per-frame duration
    imageio.mimsave(buf, frames, format="GIF", duration=duration_ms, loop=0)
    return base64.b64encode(buf.getvalue()).decode()


def _read_lerobot_ep(short_name: str, ep_index: int = 0, fmt: str = "ours") -> dict:
    base = "lerobot_openvla" if fmt == "openvla" else "lerobot"
    root = ROOT / "data" / base / short_name
    pq_path = root / f"data/chunk-000/episode_{ep_index:06d}.parquet"
    t = pq.read_table(pq_path)
    out = {
        "action": np.array(t["action"].to_pylist(), dtype=np.float32),
        "state":  np.array(t["observation.state"].to_pylist(), dtype=np.float32),
        "primary_path": root / t["observation.images.primary"][0].as_py(),
        "wrist_path":   None,
    }
    if "observation.images.wrist" in t.column_names:
        out["wrist_path"] = root / t["observation.images.wrist"][0].as_py()
    return out


def _read_rlds_first_episode(cfg, fmt: str = "ours") -> list[dict]:
    raw_dir = str(ROOT / "data" / "raw" / cfg.rlds_name / "0.1.0")
    for ep in iter_episodes(raw_dir, max_episodes=1):
        # OpenVLA keeps every frame (no degenerate filtering), so the raw steps
        # must stay unfiltered to align 1:1 with its parquet.
        if fmt == "openvla":
            return ep["steps"]
        return _filter_raw_steps_for_state_delta(cfg, ep["steps"])
    raise RuntimeError(f"no episode in {raw_dir}")


def _filter_raw_steps_for_state_delta(cfg, raw_steps: list[dict]) -> list[dict]:
    """Mirror convert.py: for STATE_DELTA_KINDS datasets, drop raw RLDS steps
    whose packed EE pose is degenerate (all-zero homogeneous matrix). This
    keeps verification aligned with the parquet, which already excludes those
    frames."""
    if cfg.action_kind not in STATE_DELTA_KINDS:
        return raw_steps
    fn = ACTION_FNS[cfg.action_kind]
    return [s for s in raw_steps
            if not _is_degenerate_pose(fn(s, None))]


# ---------------------------- ① numerical equivalence ---------------------------- #

def independent_expected_action(cfg, raw_steps: list[dict], fmt: str = "ours") -> np.ndarray:
    """Recompute the canonical action **directly from raw fields**, bypassing the
    standardize_step/finalize_actions_to_delta abstraction.  If this matches
    the parquet action, the abstraction is doing what we say it is."""

    if fmt == "openvla":
        # OpenVLA path: the expected action IS the per-step transform output.
        return openvla_actions_for_trajectory(cfg, raw_steps)

    # 'ours': motion channels from the state-derived delta, gripper overwritten
    # with the [0,1] (+1=open) normalization (mirrors convert.py).
    out = np.asarray(_expected_motion_action_ours(cfg, raw_steps), dtype=np.float32).copy()
    grip = unit_gripper_for_trajectory(cfg, raw_steps)
    n = min(len(out), len(grip))
    out[:n, 6] = grip[:n]
    return out


def _expected_motion_action_ours(cfg, raw_steps: list[dict]) -> np.ndarray:
    """The 'ours' action with its ORIGINAL (pre-normalization) gripper; the
    caller overwrites channel 6 with the unit gripper."""
    if cfg.action_kind == "abs_quat_7d":
        # iamlab: raw 8D [xyz, quat_wxyz, gripper] → 7D delta with **look-ahead**
        # convention (action[t] = command[t+1] - command[t]). Last frame: zero.
        raws = np.stack([np.asarray(s["action"], dtype=np.float64) for s in raw_steps])
        n = len(raws)
        out = np.zeros((n, 7), dtype=np.float32)
        for t in range(n - 1):
            out[t, :3] = raws[t + 1, :3] - raws[t, :3]
            q_rel = quat_relative_wxyz(raws[t, 3:7], raws[t + 1, 3:7])
            out[t, 3:6] = _quat_wxyz_to_rotvec(q_rel)
            out[t, 6]   = raws[t, 7]
        out[n - 1, :6] = 0.0
        out[n - 1, 6]  = raws[n - 1, 7]
        return out

    if cfg.action_kind in STATE_DELTA_KINDS:
        # *_state_delta datasets: re-run the per-step pack fn to get packed
        # [pos(3), quat_wxyz(4), grip(1)], then diff successive packed states.
        # This mirrors what convert.py does: standardize_step + finalize.
        fn = ACTION_FNS[cfg.action_kind]
        packed = np.stack([fn(s, None) for s in raw_steps])  # (n, 8)
        n = len(packed)
        out = np.zeros((n, 7), dtype=np.float32)
        for t in range(n - 1):
            if _is_degenerate_pose(packed[t]) or _is_degenerate_pose(packed[t + 1]):
                out[t, :6] = 0.0
                out[t, 6]  = packed[t, 7]
                continue
            out[t, :3] = packed[t + 1, :3] - packed[t, :3]
            q_rel = quat_relative_wxyz(packed[t, 3:7], packed[t + 1, 3:7])
            out[t, 3:6] = _quat_wxyz_to_rotvec(q_rel)
            out[t, 6]   = packed[t, 7]
        out[n - 1, :6] = 0.0
        out[n - 1, 6]  = packed[n - 1, 7]
        return out

    if cfg.action_kind == "abs_euler_7d_to_delta":
        # ucsd_kitchen: 8D abs [xyz(mm), euler(deg), grip, terminate] → 7D delta in SI.
        raws = np.stack([np.asarray(s["action"], dtype=np.float64) for s in raw_steps])
        n = len(raws)
        out = np.zeros((n, 7), dtype=np.float32)
        deg2rad = np.pi / 180.0
        for t in range(n - 1):
            out[t, :3] = (raws[t + 1, :3] - raws[t, :3]) * 1e-3
            q_curr = euler_xyz_to_quat_wxyz(raws[t, 3:6]     * deg2rad)
            q_next = euler_xyz_to_quat_wxyz(raws[t + 1, 3:6] * deg2rad)
            q_rel  = quat_relative_wxyz(q_curr, q_next)
            out[t, 3:6] = _quat_wxyz_to_rotvec(q_rel)
            out[t, 6]   = raws[t, 6]
        out[n - 1, :6] = 0.0
        out[n - 1, 6]  = raws[n - 1, 6]
        return out

    if cfg.action_kind == "abs_euler_7d":
        # cmu_franka_exploration: per-step action is already a 7D delta whose
        # rot channels are euler; finalize converts those euler[3:6] → rotvec.
        fn = ACTION_FNS[cfg.action_kind]
        out = np.stack([fn(s, None) for s in raw_steps]).astype(np.float64)
        for t in range(len(out)):
            out[t, 3:6] = _euler_xyz_to_rotvec(out[t, 3:6])
        return out.astype(np.float32)

    # Already-delta kinds — just run the per-step function.
    fn = ACTION_FNS[cfg.action_kind]
    return np.stack([fn(s, None) for s in raw_steps])


def numerical_equivalence(cfg, raw_steps, parquet_action, fmt: str = "ours") -> dict:
    expected = independent_expected_action(cfg, raw_steps, fmt)
    n = min(len(expected), len(parquet_action))
    diff = np.abs(expected[:n] - parquet_action[:n])
    return {
        "n_frames":       int(n),
        "max_abs_diff":   float(diff.max()),
        "mean_abs_diff":  float(diff.mean()),
        "per_channel_max": [float(diff[:, c].max()) for c in range(7)],
        "allclose_1e6":   bool(np.allclose(expected[:n], parquet_action[:n], atol=1e-6)),
        "allclose_1e3":   bool(np.allclose(expected[:n], parquet_action[:n], atol=1e-3)),
    }


# ---------------------------- ② correlation w/ state velocity ---------------------------- #

def _col_major_mat16_to_pos_R(mat16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-major (Fortran/MuJoCo) 4×4 flattened to 16 → (pos (3,), R (3,3))."""
    M = np.asarray(mat16, dtype=np.float64).reshape(4, 4).T
    return M[:3, 3].copy(), M[:3, :3].copy()


def extract_ee_xyz(cfg, lerobot_state: np.ndarray, raw_steps: list[dict]) -> np.ndarray | None:
    """Return (N, 3) end-effector position trajectory, or None if not available."""
    name = cfg.name
    if name == "viola":
        return lerobot_state[:, :3]                           # synth: [pos, euler, grip]
    if name == "stanford_hydra":
        return lerobot_state[:, :3]                           # state: [pos, quat_wxyz, euler, ...]
    if name in ("austin_buds", "utaustin_mutex"):
        # state: [7 joint, 1 grip, 16 ee_mat_col_major]
        return np.stack([_col_major_mat16_to_pos_R(lerobot_state[t, 8:24])[0]
                         for t in range(len(lerobot_state))])
    if name in ("austin_sailor", "austin_sirius"):
        # default state is just 8D (joint+grip); EE matrix is at raw obs.state_ee
        return np.stack([_col_major_mat16_to_pos_R(
            np.asarray(s["observation"]["state_ee"], dtype=np.float64))[0]
                         for s in raw_steps])
    if name == "furniture_bench":
        return lerobot_state[:, :3]                           # [pos, quat, ...]
    if name == "iamlab_cmu_pickup_insert":
        return np.stack([np.asarray(s["action"], dtype=np.float32)[:3] for s in raw_steps])
    if name == "taco_play":
        return lerobot_state[:, :3]                           # robot_obs: [pos, euler, grip, ...]
    if name == "nyu_franka_play":
        return lerobot_state[:, 7:10]                         # state: [7 joint, 3 xyz, 3 rpy]
    if name == "ucsd_kitchen":
        # state is 21D joints only; raw action stores absolute pose in mm.
        return np.stack([np.asarray(s["action"], dtype=np.float64)[:3] * 1e-3
                         for s in raw_steps])                  # mm → m
    if name == "ucsd_pick_and_place":
        return lerobot_state[:, :3]                           # state: [3 gripper_pos, 3 ori, 1 fingers]
    return None                                               # cmu_franka_exploration: no state, no abs pose


def _mask_clean_frames(ee_xyz: np.ndarray) -> np.ndarray:
    """Drop frames where the EE position is exactly (0, 0, 0) — these come from
    degenerate / missing state-matrix entries (e.g. austin_buds frame 408 has
    an all-zero 4x4 homogeneous matrix). Returns a boolean mask of length N."""
    return np.linalg.norm(ee_xyz, axis=1) > 1e-6


def correlation_per_axis(action: np.ndarray, ee_xyz: np.ndarray) -> list[float]:
    """Pearson r between action[:3] and the per-frame state-EE displacement.

    Robust to:
      - degenerate state frames (EE pos ≈ 0, 0, 0) — those are dropped
      - per-axis constant signals (returns NaN with 'bad' style)
    """
    vel = np.diff(ee_xyz, axis=0)            # (N-1, 3)
    a = action[:-1, :3]                      # align lengths
    n = min(len(vel), len(a))
    vel, a = vel[:n], a[:n]
    # Mask out frames adjacent to degenerate state entries
    clean = _mask_clean_frames(ee_xyz[:-1]) & _mask_clean_frames(ee_xyz[1:])
    clean = clean[:n]
    vel, a = vel[clean], a[clean]
    rs = []
    for axis in range(3):
        v, x = vel[:, axis], a[:, axis]
        if len(v) < 3 or v.std() < 1e-9 or x.std() < 1e-9:
            rs.append(float("nan"))
        else:
            rs.append(float(np.corrcoef(v, x)[0, 1]))
    return rs


def plot_action_vs_velocity(action: np.ndarray, ee_xyz: np.ndarray, title: str) -> str:
    vel = np.diff(ee_xyz, axis=0)
    a = action[:-1, :3]
    n = min(len(vel), len(a))
    vel, a = vel[:n], a[:n]
    clean = _mask_clean_frames(ee_xyz[:-1]) & _mask_clean_frames(ee_xyz[1:])
    clean = clean[:n]
    vel, a = vel[clean], a[clean]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis in range(3):
        v, x = vel[:, axis], a[:, axis]
        r = (np.corrcoef(v, x)[0, 1]
             if len(v) > 2 and v.std() > 1e-9 and x.std() > 1e-9 else float("nan"))
        axes[axis].scatter(v, x, s=4, alpha=0.5)
        axes[axis].set_title(f"axis {'xyz'[axis]}  r = {r:.3f}")
        axes[axis].set_xlabel("state Δ (next - current)")
        axes[axis].set_ylabel("action [:3]")
        axes[axis].axhline(0, color="k", lw=0.5)
        axes[axis].axvline(0, color="k", lw=0.5)
        axes[axis].grid(alpha=0.3)
    fig.suptitle(title)
    return _fig_to_b64(fig)


# ---------------------------- ⑥ rotation round-trip ---------------------------- #

def _is_valid_R(R: np.ndarray) -> bool:
    """A 3×3 is a usable rotation if it's finite, non-degenerate, and has det≈1.
    Filters out the all-zero homogeneous-matrix frames (austin_buds/sirius) and
    any NaN orientation before they pollute the geodesic-error stats."""
    if not np.isfinite(R).all():
        return False
    if np.linalg.norm(R) < 1e-6:
        return False
    return abs(float(np.linalg.det(R)) - 1.0) < 1e-2


def rotation_roundtrip(action: np.ndarray, R_per_frame: np.ndarray) -> dict | None:
    """Independent numerical check of the axis-angle rotation channel.

    `action[t, 3:6]` is the world-frame relative rotation taking the EE
    orientation R_t → R_{t+1} (pipeline convention: q_rel = q_{t+1}·q_t⁻¹, i.e.
    R_{t+1} = R_rel · R_t — left/world multiply). We reconstruct R_{t+1} from
    R_t + action using **scipy** — an implementation of the rotvec→matrix math
    that is independent of the pipeline's hand-rolled `_quat_wxyz_to_rotvec` —
    and measure the geodesic angle between predicted and actual next orientation.

    Because R_per_frame comes straight from the *state* (quat/euler/4×4 matrix)
    while the rotvec went through the action path, a near-zero error proves the
    rotvec math, the frame convention (axis order, multiply side, sign), and the
    state↔action orientation sources are all mutually consistent. A non-trivial
    error pinpoints exactly which of those is wrong.

    Returns None if no valid orientation is available (e.g. cmu_franka_exploration).
    """
    from scipy.spatial.transform import Rotation as Rsp

    if R_per_frame is None:
        return None
    n = min(len(action), len(R_per_frame))
    errs_deg = np.full(n, np.nan, dtype=np.float64)   # full length for plotting
    for t in range(n - 1):
        R_t, R_next = R_per_frame[t], R_per_frame[t + 1]
        if not (_is_valid_R(R_t) and _is_valid_R(R_next)):
            continue
        R_rel = Rsp.from_rotvec(np.asarray(action[t, 3:6], dtype=np.float64)).as_matrix()
        R_pred = R_rel @ R_t                          # world-frame left multiply
        ang = Rsp.from_matrix(R_pred @ R_next.T).magnitude()  # geodesic angle (rad)
        errs_deg[t] = np.degrees(ang)

    valid = errs_deg[np.isfinite(errs_deg)]
    if len(valid) == 0:
        return None
    return {
        "n_valid":   int(len(valid)),
        "mean_deg":  float(valid.mean()),
        "median_deg": float(np.median(valid)),
        "p99_deg":   float(np.percentile(valid, 99)),
        "max_deg":   float(valid.max()),
        "errs_deg":  errs_deg,                         # per-frame, NaN where skipped
    }


def plot_rotation_error(errs_deg: np.ndarray, title: str) -> str:
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(errs_deg, linewidth=1, color="#9467bd")
    ax.set_ylabel("geodesic\nerror (deg)")
    ax.set_xlabel("frame")
    ax.grid(alpha=0.3)
    fig.suptitle(title)
    return _fig_to_b64(fig)


# ---------------------------- ③ time-series + 3D trajectory ---------------------------- #

def plot_action_timeseries(action: np.ndarray, title: str) -> str:
    fig, axes = plt.subplots(7, 1, figsize=(9, 8), sharex=True)
    names = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
    for i, n in enumerate(names):
        axes[i].plot(action[:, i], linewidth=1)
        axes[i].set_ylabel(n)
        axes[i].grid(alpha=0.3)
    axes[-1].set_xlabel("frame")
    fig.suptitle(title)
    return _fig_to_b64(fig)


def plot_trajectory_3d(ee_xyz: np.ndarray, title: str,
                       integrated: np.ndarray | None = None) -> str:
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(ee_xyz[:, 0], ee_xyz[:, 1], ee_xyz[:, 2], "-", label="state EE pos", linewidth=1.2)
    if integrated is not None:
        ax.plot(integrated[:, 0], integrated[:, 1], integrated[:, 2],
                "--", label="∑action[:3] + state[0]", linewidth=1.2)
    ax.scatter(*ee_xyz[0], c="g", s=60, label="start")
    ax.scatter(*ee_xyz[-1], c="r", s=60, label="end")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    return _fig_to_b64(fig)


# ---------------------------- animated 3D trajectory + EE frame ---------------------------- #

def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) → 3×3 rotation matrix."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - w * z),       2 * (x * z + w * y)],
        [2 * (x * y + w * z),         1 - 2 * (x * x + z * z),   2 * (y * z - w * x)],
        [2 * (x * z - w * y),         2 * (y * z + w * x),       1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def euler_xyz_to_rotmat(rpy: np.ndarray) -> np.ndarray:
    """XYZ Tait–Bryan (roll, pitch, yaw) → 3×3 rotation matrix.
    Convention matches `homogeneous16_to_pos_euler`: R = Rz · Ry · Rx."""
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw),   np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def extract_rotations(cfg, lerobot_state: np.ndarray, raw_steps: list[dict]) -> np.ndarray | None:
    """Return (N, 3, 3) rotation matrices describing the EE orientation per frame.
    Returns None if the dataset doesn't expose EE orientation."""
    name = cfg.name
    n = len(raw_steps)
    R = np.zeros((n, 3, 3), dtype=np.float64)

    if name == "viola":
        for t in range(n):
            R[t] = euler_xyz_to_rotmat(lerobot_state[t, 3:6])
        return R

    if name == "stanford_hydra":
        # raw state[3:7] is stored (x, y, z, w); reorder to wxyz to match the
        # action path (action_stanford_hydra_state_delta does the same).
        for t in range(n):
            q = lerobot_state[t, 3:7]
            R[t] = quat_wxyz_to_rotmat(q[[3, 0, 1, 2]])
        return R

    if name in ("austin_buds", "utaustin_mutex"):
        for t in range(n):
            _, R[t] = _col_major_mat16_to_pos_R(lerobot_state[t, 8:24])
        return R

    if name in ("austin_sailor", "austin_sirius"):
        for t in range(n):
            mat = np.asarray(raw_steps[t]["observation"]["state_ee"], dtype=np.float64)
            _, R[t] = _col_major_mat16_to_pos_R(mat)
        return R

    if name == "furniture_bench":
        for t in range(n):
            q = np.asarray(raw_steps[t]["observation"]["state"], dtype=np.float64)[3:7]
            R[t] = quat_wxyz_to_rotmat(q)
        return R

    if name == "iamlab_cmu_pickup_insert":
        for t in range(n):
            q = np.asarray(raw_steps[t]["action"], dtype=np.float64)[3:7]
            R[t] = quat_wxyz_to_rotmat(q)
        return R

    if name == "taco_play":
        # robot_obs[3:6] is euler (radians) — values span ±π so it's rad, not deg
        for t in range(n):
            R[t] = euler_xyz_to_rotmat(lerobot_state[t, 3:6])
        return R

    if name == "nyu_franka_play":
        for t in range(n):
            R[t] = euler_xyz_to_rotmat(lerobot_state[t, 10:13])
        return R

    if name == "ucsd_kitchen":
        # raw action euler is in degrees → convert to rad
        for t in range(n):
            rpy_deg = np.asarray(raw_steps[t]["action"], dtype=np.float64)[3:6]
            R[t] = euler_xyz_to_rotmat(rpy_deg * np.pi / 180.0)
        return R

    if name == "ucsd_pick_and_place":
        for t in range(n):
            R[t] = euler_xyz_to_rotmat(lerobot_state[t, 3:6])
        return R

    return None  # cmu_franka_exploration: no EE pose available


def _downsample_indices(n: int, target: int = 40) -> list[int]:
    if n <= target:
        return list(range(n))
    return [int(i * (n - 1) / (target - 1)) for i in range(target)]


def _rotvec_arrow_scale(action_rotvec: np.ndarray | None, ref_len: float,
                        pct: float = 90.0) -> float:
    """Scale factor mapping a *reference* per-step |rotvec| (rad) to `ref_len`
    (scene units). We normalize against the `pct`-th percentile rather than the
    max so the typical-frame arrow is long enough to be legible: most datasets
    have a heavily right-skewed rotation distribution (p50 ≪ max), and dividing
    by max squashes nearly every frame to ~10% length. Peak frames are allowed
    to exceed `ref_len` and get clipped at draw time (see `_scaled_rotvec`)."""
    if action_rotvec is None:
        return 0.0
    mags = np.linalg.norm(action_rotvec, axis=1)
    ref = float(np.percentile(mags, pct))
    if ref <= 1e-9:                       # near-degenerate p90 → fall back to max
        ref = float(mags.max())
    return ref_len / ref if ref > 1e-9 else 0.0


def _scaled_rotvec(rv: np.ndarray, rot_scale: float, max_visual: float) -> np.ndarray:
    """Scale a raw rotvec to scene units, clipping its length to `max_visual`
    (keeps peak-frame arrows from shooting off-screen while preserving direction)."""
    vec = np.asarray(rv, dtype=np.float64) * rot_scale
    L = float(np.linalg.norm(vec))
    if L > max_visual > 0.0:
        vec = vec * (max_visual / L)
    return vec


def animate_trajectory_with_frame(ee_xyz: np.ndarray,
                                  R_per_frame: np.ndarray,
                                  title: str,
                                  indices: list[int] | None = None,
                                  target_frames: int = 60,
                                  fps: int = 10,
                                  action_rotvec: np.ndarray | None = None) -> str:
    """Animated GIF: growing 3D trajectory + EE coordinate frame (x=R, y=G, z=B),
    plus the per-step action axis-angle (rotvec) as a magenta arrow from the EE
    (direction = world-frame rotation axis, length ∝ rotation angle this step).
    Kept alongside the interactive Plotly version so the GIF can be placed next to
    the video GIF and play in sync at the same fps for visual comparison."""
    n = len(ee_xyz)
    idx = indices if indices is not None else _downsample_indices(n, target_frames)

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    mins = ee_xyz.min(axis=0)
    maxs = ee_xyz.max(axis=0)
    center = (mins + maxs) / 2
    half_range = max(((maxs - mins).max() / 2) * 1.25, 0.05)
    axis_len = half_range * 0.18
    rot_ref_len = half_range * 0.45        # p90-frame arrow length (≫ EE triad, legible)
    rot_max_visual = half_range * 0.65     # clip peak frames so they stay on-screen
    rot_scale = _rotvec_arrow_scale(action_rotvec, rot_ref_len)

    def update(k):
        ax.clear()
        i = idx[k]
        ax.plot(ee_xyz[: i + 1, 0], ee_xyz[: i + 1, 1], ee_xyz[: i + 1, 2],
                color="#1f77b4", linewidth=1.4)
        ax.scatter(*ee_xyz[0],  c="g", s=40)
        ax.scatter(*ee_xyz[i],  c="r", s=50)
        R = R_per_frame[i]
        origin = ee_xyz[i]
        for col, color in zip(range(3), ("#d62728", "#2ca02c", "#1f77b4")):
            vec = R[:, col] * axis_len
            ax.quiver(origin[0], origin[1], origin[2],
                      vec[0], vec[1], vec[2],
                      color=color, linewidth=2.5, arrow_length_ratio=0.25)
        if rot_scale > 0.0 and i < len(action_rotvec):
            rv = _scaled_rotvec(action_rotvec[i], rot_scale, rot_max_visual)
            if np.linalg.norm(rv) > 1e-9:
                ax.quiver(origin[0], origin[1], origin[2],
                          rv[0], rv[1], rv[2],
                          color="#e377c2", linewidth=3.0, arrow_length_ratio=0.3)
        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[1] - half_range, center[1] + half_range)
        ax.set_zlim(center[2] - half_range, center[2] + half_range)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"{title}\nframe {i}/{n - 1}  (EE: red=x green=y blue=z; "
                     f"magenta=action rotvec)")

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=False)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        anim.save(tmp_path, writer=PillowWriter(fps=fps), savefig_kwargs={"dpi": 70})
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        plt.close(fig)
        os.unlink(tmp_path)
    return base64.b64encode(data).decode()


def build_plotly_traj_anim(ee_xyz: np.ndarray,
                           R_per_frame: np.ndarray,
                           title: str,
                           indices: list[int] | None = None,
                           target_frames: int = 60,
                           fps: int = 8,
                           action_rotvec: np.ndarray | None = None) -> dict:
    """Build a Plotly figure spec ({data, layout, frames}) for an interactive 3D
    trajectory animation. The user can drag to rotate, scroll to zoom, and use
    the play button / slider to scrub through frames.

    Traces:
      0: full trajectory (static gray background, helps orient while rotating)
      1: active trajectory up to current frame (animated, blue)
      2: current EE position marker (animated, red)
      3-5: EE coordinate axes x/y/z (animated, red/green/blue)
      6: action rotvec arrow line (animated, magenta)
      7: action rotvec arrow head cone (animated, magenta)
      8: start marker (static, green)
    """
    n = len(ee_xyz)
    idx = indices if indices is not None else _downsample_indices(n, target_frames)

    # Include world origin in view range so the world-axis triad is always visible.
    mins = np.minimum(ee_xyz.min(axis=0), 0.0)
    maxs = np.maximum(ee_xyz.max(axis=0), 0.0)
    center = (mins + maxs) / 2
    half_range = max(((maxs - mins).max() / 2) * 1.25, 0.05)
    axis_len = float(half_range * 0.18)
    world_axis_len = float(half_range * 0.30)
    rot_ref_len = float(half_range * 0.45)     # p90-frame arrow length (≫ EE triad)
    rot_max_visual = float(half_range * 0.65)  # clip peak frames so they stay on-screen
    rot_scale = _rotvec_arrow_scale(action_rotvec, rot_ref_len)
    rot_cone_ref = float(half_range * 0.12)

    def axis_segment(origin: np.ndarray, R: np.ndarray, col: int) -> dict:
        end = origin + R[:, col] * axis_len
        return {"x": [float(origin[0]), float(end[0])],
                "y": [float(origin[1]), float(end[1])],
                "z": [float(origin[2]), float(end[2])]}

    def _rv_end_dir(i: int) -> tuple[np.ndarray, np.ndarray]:
        """Clipped scene-unit rotvec vector for frame i (zeros if unavailable)."""
        if rot_scale > 0.0 and i < len(action_rotvec):
            return _scaled_rotvec(action_rotvec[i], rot_scale, rot_max_visual), action_rotvec[i]
        return np.zeros(3), np.zeros(3)

    def rotvec_line(origin: np.ndarray, i: int) -> dict:
        vec, _ = _rv_end_dir(i)
        end = origin + vec
        return {"x": [float(origin[0]), float(end[0])],
                "y": [float(origin[1]), float(end[1])],
                "z": [float(origin[2]), float(end[2])]}

    def rotvec_cone(origin: np.ndarray, i: int) -> dict:
        vec, rv = _rv_end_dir(i)
        end = origin + vec
        return {"x": [float(end[0])], "y": [float(end[1])], "z": [float(end[2])],
                "u": [float(rv[0])], "v": [float(rv[1])], "w": [float(rv[2])]}

    frames = []
    for k, i in enumerate(idx):
        origin = ee_xyz[i]
        R = R_per_frame[i]
        frames.append({
            "name": str(k),
            "data": [
                {"x": ee_xyz[: i + 1, 0].tolist(),
                 "y": ee_xyz[: i + 1, 1].tolist(),
                 "z": ee_xyz[: i + 1, 2].tolist()},
                {"x": [float(origin[0])],
                 "y": [float(origin[1])],
                 "z": [float(origin[2])]},
                axis_segment(origin, R, 0),
                axis_segment(origin, R, 1),
                axis_segment(origin, R, 2),
                rotvec_line(origin, i),
                rotvec_cone(origin, i),
            ],
            "traces": [1, 2, 3, 4, 5, 6, 7],
            "layout": {"title": {"text": f"{title}<br><sub>frame {i}/{n - 1}</sub>"}},
        })

    i0 = idx[0]
    origin0 = ee_xyz[i0]
    R0 = R_per_frame[i0]

    data = [
        {"type": "scatter3d", "mode": "lines",
         "x": ee_xyz[:, 0].tolist(),
         "y": ee_xyz[:, 1].tolist(),
         "z": ee_xyz[:, 2].tolist(),
         "line": {"color": "rgba(120,120,120,0.30)", "width": 2},
         "name": "full trajectory", "hoverinfo": "skip"},
        {"type": "scatter3d", "mode": "lines",
         "x": ee_xyz[: i0 + 1, 0].tolist(),
         "y": ee_xyz[: i0 + 1, 1].tolist(),
         "z": ee_xyz[: i0 + 1, 2].tolist(),
         "line": {"color": "#1f77b4", "width": 4},
         "name": "trajectory"},
        {"type": "scatter3d", "mode": "markers",
         "x": [float(origin0[0])], "y": [float(origin0[1])], "z": [float(origin0[2])],
         "marker": {"color": "red", "size": 5},
         "name": "current"},
        {"type": "scatter3d", "mode": "lines",
         **axis_segment(origin0, R0, 0),
         "line": {"color": "#d62728", "width": 6},
         "name": "EE x", "hoverinfo": "skip"},
        {"type": "scatter3d", "mode": "lines",
         **axis_segment(origin0, R0, 1),
         "line": {"color": "#2ca02c", "width": 6},
         "name": "EE y", "hoverinfo": "skip"},
        {"type": "scatter3d", "mode": "lines",
         **axis_segment(origin0, R0, 2),
         "line": {"color": "#1f77b4", "width": 6},
         "name": "EE z", "hoverinfo": "skip"},
        {"type": "scatter3d", "mode": "lines",
         **rotvec_line(origin0, i0),
         "line": {"color": "#e377c2", "width": 7},
         "name": "action rotvec", "hoverinfo": "skip"},
        {"type": "cone",
         **rotvec_cone(origin0, i0),
         "sizemode": "absolute", "sizeref": rot_cone_ref,
         "anchor": "tip", "showscale": False,
         "colorscale": [[0, "#e377c2"], [1, "#e377c2"]],
         "hoverinfo": "skip", "showlegend": False},
        {"type": "scatter3d", "mode": "markers",
         "x": [float(ee_xyz[0, 0])], "y": [float(ee_xyz[0, 1])], "z": [float(ee_xyz[0, 2])],
         "marker": {"color": "green", "size": 5},
         "name": "start"},
    ]

    # World-frame axis triad at origin (red=X, green=Y, blue=Z) — static, helps the
    # viewer identify which direction each scene axis points to when rotating.
    for col, (color, label) in enumerate([
        ("#d62728", "X"), ("#2ca02c", "Y"), ("#1f77b4", "Z"),
    ]):
        tip = [0.0, 0.0, 0.0]
        tip[col] = world_axis_len
        lbl = [0.0, 0.0, 0.0]
        lbl[col] = world_axis_len * 1.18
        direction = [0.0, 0.0, 0.0]
        direction[col] = 1.0
        data.extend([
            {"type": "scatter3d", "mode": "lines",
             "x": [0.0, tip[0]], "y": [0.0, tip[1]], "z": [0.0, tip[2]],
             "line": {"color": color, "width": 5},
             "name": f"world {label}", "hoverinfo": "skip", "showlegend": False},
            {"type": "cone",
             "x": [tip[0]], "y": [tip[1]], "z": [tip[2]],
             "u": [direction[0]], "v": [direction[1]], "w": [direction[2]],
             "sizemode": "absolute", "sizeref": world_axis_len * 0.30,
             "anchor": "tip", "showscale": False,
             "colorscale": [[0, color], [1, color]],
             "hoverinfo": "skip", "showlegend": False},
            {"type": "scatter3d", "mode": "text",
             "x": [lbl[0]], "y": [lbl[1]], "z": [lbl[2]],
             "text": [label], "textfont": {"color": color, "size": 14},
             "hoverinfo": "skip", "showlegend": False},
        ])

    frame_ms = int(1000 / fps)
    play_args = [None, {"frame": {"duration": frame_ms, "redraw": True},
                        "fromcurrent": True, "transition": {"duration": 0}}]
    pause_args = [[None], {"frame": {"duration": 0, "redraw": False},
                           "mode": "immediate", "transition": {"duration": 0}}]
    slider_steps = [{"label": str(k), "method": "animate",
                     "args": [[str(k)], {"frame": {"duration": 0, "redraw": True},
                                          "mode": "immediate",
                                          "transition": {"duration": 0}}]}
                    for k in range(len(idx))]

    layout = {
        "title": {"text": f"{title}<br><sub>frame {i0}/{n - 1}</sub>"},
        "scene": {
            "xaxis": {"range": [float(center[0] - half_range), float(center[0] + half_range)], "title": "x"},
            "yaxis": {"range": [float(center[1] - half_range), float(center[1] + half_range)], "title": "y"},
            "zaxis": {"range": [float(center[2] - half_range), float(center[2] + half_range)], "title": "z"},
            "aspectmode": "cube",
        },
        "margin": {"l": 0, "r": 0, "t": 60, "b": 0},
        "showlegend": False,
        "height": 500,
        "updatemenus": [{
            "type": "buttons", "direction": "left",
            "x": 0.05, "y": -0.05, "xanchor": "left", "yanchor": "top",
            "pad": {"t": 0, "r": 8},
            "buttons": [
                {"label": "▶ Play",  "method": "animate", "args": play_args},
                {"label": "⏸ Pause", "method": "animate", "args": pause_args},
            ],
        }],
        "sliders": [{
            "active": 0, "x": 0.18, "y": -0.05, "len": 0.78,
            "xanchor": "left", "yanchor": "top",
            "currentvalue": {"prefix": "frame index: ", "font": {"size": 12}},
            "steps": slider_steps,
        }],
    }

    return {"data": data, "layout": layout, "frames": frames}


def integrated_xyz(action: np.ndarray, start_xyz: np.ndarray) -> np.ndarray:
    """Cumulative sum of action[:, :3] starting from start_xyz."""
    out = np.zeros((len(action) + 1, 3))
    out[0] = start_xyz
    for t in range(len(action)):
        out[t + 1] = out[t] + action[t, :3]
    return out


# ---------------------------- ④ GIF ---------------------------- #

def _frames_from_mp4_at(path: Path, indices: list[int]) -> list[np.ndarray]:
    """Read specific frame indices from an mp4. Indices must be sorted ascending."""
    r = imageio.get_reader(str(path))
    out, want = [], iter(sorted(set(indices)))
    next_idx = next(want, None)
    for i, fr in enumerate(r):
        while next_idx is not None and i == next_idx:
            out.append(fr)
            next_idx = next(want, None)
        if next_idx is None:
            break
    r.close()
    # If we ran out of frames in the mp4 but still have requested indices, repeat the last
    if len(out) < len(indices):
        out.extend([out[-1]] * (len(indices) - len(out)))
    return out


def side_by_side_gif(primary: Path, wrist: Path | None,
                     indices: list[int], fps: int = 8) -> str:
    p = _frames_from_mp4_at(primary, indices)
    if wrist is None or not Path(wrist).exists():
        return _gif_b64(p, fps=fps)
    w = _frames_from_mp4_at(wrist, indices)
    n = min(len(p), len(w))
    out = []
    for i in range(n):
        a, b = p[i], w[i]
        h = max(a.shape[0], b.shape[0])
        def pad(im, h):
            if im.shape[0] == h:
                return im
            top = (h - im.shape[0]) // 2
            bot = h - im.shape[0] - top
            return np.pad(im, ((top, bot), (0, 0), (0, 0)), constant_values=0)
        out.append(np.concatenate([pad(a, h), pad(b, h)], axis=1))
    return _gif_b64(out, fps=fps)


# ---------------------------- driver ---------------------------- #

def verify_one(short_name: str, fmt: str = "ours") -> dict:
    cfg = get_config(short_name)
    print(f"  loading parquet…", flush=True)
    lerobot = _read_lerobot_ep(cfg.name, ep_index=0, fmt=fmt)
    print(f"  re-reading first RLDS episode…", flush=True)
    raw_steps = _read_rlds_first_episode(cfg, fmt=fmt)

    # ① numerical equivalence
    eq = numerical_equivalence(cfg, raw_steps, lerobot["action"], fmt)

    # EE orientation per frame (from state) — reused by ⑥ round-trip and the animation.
    R_per_frame = extract_rotations(cfg, lerobot["state"], raw_steps)

    # ⑥ rotation round-trip: reconstruct R_{t+1} from R_t + action via scipy.
    # Only meaningful for 'ours' (action[3:6] is a world-frame relative axis-angle).
    # OpenVLA stores raw euler / absolute-euler in that slot, so the round-trip
    # does not apply — skip it.
    rot_rt = None
    rot_rt_b64 = None
    if fmt != "openvla":
        rot_rt = rotation_roundtrip(lerobot["action"], R_per_frame)
        if rot_rt is not None:
            rot_rt_b64 = plot_rotation_error(
                rot_rt["errs_deg"],
                f"{cfg.name}: rotation round-trip geodesic error (deg)",
            )

    # ② correlation w/ EE velocity
    ee_xyz = extract_ee_xyz(cfg, lerobot["state"], raw_steps)
    corr = scatter_b64 = traj_b64 = traj_anim_b64 = None
    traj_anim_plotly = None
    if ee_xyz is not None and len(ee_xyz) >= 3:
        corr = correlation_per_axis(lerobot["action"], ee_xyz)
        scatter_b64 = plot_action_vs_velocity(
            lerobot["action"], ee_xyz, f"{cfg.name}: action[:3] vs state-EE Δ"
        )
        # ∑action overlay only makes sense when action[:3] is a position delta.
        # OpenVLA leaves some datasets absolute (ucsd_kitchen, iamlab) or in
        # non-SI command units (taco_play), where integrating is meaningless.
        integ = integrated_xyz(lerobot["action"], ee_xyz[0]) if fmt != "openvla" else None
        dashed = "dashed=∑action+x₀" if fmt != "openvla" else "no ∑action (openvla raw units)"
        traj_b64 = plot_trajectory_3d(
            ee_xyz, f"{cfg.name}: EE trajectory (solid=state, {dashed})", integ
        )
        # animated 3D trajectory + EE coordinate frame — two flavors:
        #   GIF: plays in sync with the video GIF (same indices + fps) for hand-eye
        #        verification of the reconstructed EE pose against the video.
        #   Plotly: same data but interactive (drag to rotate, slider to scrub).
        common_indices = _downsample_indices(eq["n_frames"], target=50)
        shared_fps = 8
        if R_per_frame is not None:
            print("  rendering 3D trajectory animation (GIF + interactive)…", flush=True)
            # The magenta arrow interprets action[3:6] as a world-frame axis-angle —
            # true only for 'ours'. OpenVLA's rotation channel is raw euler, so we
            # suppress the arrow (None) to avoid drawing a misleading vector.
            action_rotvec = lerobot["action"][:, 3:6] if fmt != "openvla" else None
            traj_anim_b64 = animate_trajectory_with_frame(
                ee_xyz, R_per_frame,
                title=f"{cfg.name}: animated EE pose",
                indices=common_indices, fps=shared_fps,
                action_rotvec=action_rotvec,
            )
            traj_anim_plotly = build_plotly_traj_anim(
                ee_xyz, R_per_frame,
                title=f"{cfg.name}: animated EE pose",
                indices=common_indices, fps=shared_fps,
                action_rotvec=action_rotvec,
            )

    # ③ time series
    ts_b64 = plot_action_timeseries(lerobot["action"], f"{cfg.name}: action over time")

    # ④ GIF (same frame indices + fps as the trajectory animation so they sync visually)
    print(f"  rendering GIF…", flush=True)
    if ee_xyz is None or len(ee_xyz) < 3:
        common_indices = _downsample_indices(eq["n_frames"], target=50)
        shared_fps = 8
    gif_b64 = side_by_side_gif(lerobot["primary_path"], lerobot["wrist_path"],
                               indices=common_indices, fps=shared_fps)

    return {
        "name":           cfg.name,
        "fmt":            fmt,
        "action_kind":    ("openvla_raw" if fmt == "openvla" else cfg.action_kind),
        "notes":          (OPENVLA_ACTION_NOTES.get(cfg.name, "") if fmt == "openvla" else cfg.notes),
        "n_frames":       eq["n_frames"],
        "equivalence":    eq,
        "rot_roundtrip":  rot_rt,
        "rot_rt_b64":     rot_rt_b64,
        "correlation":    corr,
        "scatter_b64":       scatter_b64,
        "traj_b64":          traj_b64,
        "traj_anim_b64":     traj_anim_b64,
        "traj_anim_plotly":  traj_anim_plotly,
        "ts_b64":            ts_b64,
        "gif_b64":           gif_b64,
    }


CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1200px; margin: auto; padding: 24px; color: #222; }
h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
h2 { margin-top: 48px; color: #06c; }
h3 { margin-top: 24px; }
table { border-collapse: collapse; margin: 8px 0 16px; }
td, th { border: 1px solid #ccc; padding: 6px 14px; text-align: left; }
th { background: #f4f4f4; }
.ok  { color: #0a0; font-weight: 600; }
.meh { color: #c80; font-weight: 600; }
.bad { color: #c00; font-weight: 600; }
img { max-width: 100%; height: auto; box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.muted { color: #888; font-style: italic; }
hr { margin: 48px 0; border: 0; border-top: 1px solid #ddd; }
code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
.side-by-side { display: flex; flex-wrap: wrap; gap: 16px; align-items: flex-start; margin-top: 8px; }
.side-by-side > figure { flex: 1 1 480px; margin: 0; min-width: 0; }
.side-by-side figcaption { font-size: 13px; color: #555; margin-top: 4px; }
"""


def _eq_cls(diff_max: float, allclose_1e6: bool) -> str:
    if allclose_1e6: return "ok"
    if diff_max < 1e-3: return "meh"
    return "bad"


def _r_cls(r: float) -> str:
    if r != r: return "bad"            # NaN
    a = abs(r)
    if a >= 0.95: return "ok"
    if a >= 0.7:  return "meh"
    return "bad"


def _rot_cls(max_deg: float) -> str:
    # state-derived rotvec round-trips to ~1e-3°; allow generous headroom before
    # flagging. >2° means a real convention/axis/sign error, not float noise.
    if max_deg < 0.1: return "ok"
    if max_deg < 2.0: return "meh"
    return "bad"


def render_html(results: list[dict], out_path: Path, fmt: str = "ours") -> None:
    fmt_label = {
        "ours": "我们的格式（state 派生 7D 物理 delta，轴角旋转，SI 单位）",
        "openvla": "OpenVLA 格式（逐帧复刻 OpenVLA 原始 action 变换，欧拉角旋转，原始单位）",
    }.get(fmt, fmt)
    openvla_note = ("<p class='muted'>注意：本报告为 <b>OpenVLA 格式</b>。⑥ 轴角 round-trip 与 ∑action 轨迹叠加、"
                    "品红轴角箭头均不适用（OpenVLA 旋转通道是原始欧拉角、部分数据集为绝对位姿/非 SI 单位），已自动跳过。"
                    "② 相关性对「delta 型」数据集仍有意义；对绝对位姿数据集（ucsd_kitchen / iamlab）相关性偏低属预期。</p>"
                    if fmt == "openvla" else "")
    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             "<title>OXE → LeRobot v2 验证报告</title>",
             # Plotly.js for interactive 3D trajectory views (drag to rotate, scroll to zoom)
             "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js' charset='utf-8'></script>",
             f"<style>{CSS}</style></head><body>",
             "<h1>OXE → LeRobot v2 转换验证报告</h1>",
             f"<p><b>动作格式：</b>{fmt_label}</p>",
             openvla_note,
             "<p>每个数据集跑 4 项检查：① 数值一致性 ② action 与 EE 速度相关性 ③ 时序图 ④ 视频回放。</p>",
             "<p class='muted'>提示：④ 中上方是与视频同步播放的 GIF（便于对比），下方的交互式 3D 视图支持鼠标拖拽旋转、滚轮缩放，可暂停在任意帧从不同角度查看。</p>"]

    for r in results:
        parts.append(f"<h2>{r['name']} &nbsp;<code>{r['action_kind']}</code></h2>")
        if r["notes"]:
            parts.append(f"<p class='muted'>{r['notes']}</p>")
        parts.append(f"<p>Episode 0,  <b>{r['n_frames']}</b> 帧</p>")

        # ① equivalence
        eq = r["equivalence"]
        cls = _eq_cls(eq["max_abs_diff"], eq["allclose_1e6"])
        parts += [
            "<h3>① RLDS↔Parquet 数值一致性</h3>",
            "<table>",
            f"<tr><th>max abs diff</th><td class='{cls}'>{eq['max_abs_diff']:.3e}</td></tr>",
            f"<tr><th>mean abs diff</th><td>{eq['mean_abs_diff']:.3e}</td></tr>",
            f"<tr><th>per-channel max</th><td>"
            + ", ".join(f"{v:.2e}" for v in eq["per_channel_max"])
            + "</td></tr>",
            f"<tr><th>allclose @ 1e-6</th><td class='{'ok' if eq['allclose_1e6'] else 'bad'}'>{eq['allclose_1e6']}</td></tr>",
            f"<tr><th>allclose @ 1e-3</th><td class='{'ok' if eq['allclose_1e3'] else 'bad'}'>{eq['allclose_1e3']}</td></tr>",
            "</table>",
        ]

        # ⑥ rotation round-trip (independent scipy reconstruction)
        parts.append("<h3>⑥ 轴角旋转 round-trip（scipy 独立重建 R<sub>t+1</sub>）</h3>")
        rt = r.get("rot_roundtrip")
        if rt:
            cls = _rot_cls(rt["max_deg"])
            parts += [
                "<p class='muted'>用 scipy 把 <code>action[t,3:6]</code> 当作 world 系相对旋转，"
                "由 state 的 R<sub>t</sub> 重建 R<sub>t+1</sub>，与真实 R<sub>t+1</sub> 比测地角误差。"
                "越接近 0 越好（state 派生的旋转应 ~1e-3°）；若达到几度则说明轴角数学/frame 约定/轴序有误。</p>",
                "<table>",
                f"<tr><th>有效帧</th><td>{rt['n_valid']}</td></tr>",
                f"<tr><th>mean</th><td>{rt['mean_deg']:.3e}°</td></tr>",
                f"<tr><th>median</th><td>{rt['median_deg']:.3e}°</td></tr>",
                f"<tr><th>p99</th><td>{rt['p99_deg']:.3e}°</td></tr>",
                f"<tr><th>max</th><td class='{cls}'>{rt['max_deg']:.3e}°</td></tr>",
                "</table>",
            ]
            if r.get("rot_rt_b64"):
                parts.append(f"<img src='data:image/png;base64,{r['rot_rt_b64']}'>")
        else:
            parts.append("<p class='muted'>该数据集 state 不含 EE 姿态，无法做旋转 round-trip，跳过此项。</p>")

        # ② correlation
        parts.append("<h3>② Action vs State-EE 速度相关性</h3>")
        if r["correlation"]:
            parts.append("<table><tr><th>axis</th><th>Pearson r</th></tr>")
            for i, rv in enumerate(r["correlation"]):
                cls = _r_cls(rv)
                parts.append(f"<tr><td>{'xyz'[i]}</td><td class='{cls}'>{rv:+.3f}</td></tr>")
            parts.append("</table>")
            parts.append(f"<img src='data:image/png;base64,{r['scatter_b64']}'>")
        else:
            parts.append("<p class='muted'>该数据集 state 中没有 EE 位置，跳过此项。</p>")

        # 3D trajectory (static — full overview)
        if r["traj_b64"]:
            parts.append("<h3>③ 末端轨迹 3D 视图（全程）</h3>")
            parts.append(f"<img src='data:image/png;base64,{r['traj_b64']}'>")

        # ④ video + synced trajectory GIF (top row), interactive Plotly (below).
        parts.append("<h3>④ 视频回放  ↔  末端轨迹 + 姿态动画（同步播放 + 可拖拽旋转）</h3>")
        parts.append("<p class='muted'>上方左右两个 GIF 使用相同帧索引和 fps 同步播放：左是 primary + wrist 相机视频，右是 EE 位置（蓝线）+ 末端坐标系（红=x，绿=y，蓝=z），以及当前帧 action 的轴角增量（品红箭头：方向=world 系下的旋转轴，长度∝该步旋转角，按本 episode 最大角归一化），用来对比「视频里 gripper 的位置朝向」和「重构出的 EE 位姿/动作」是否一致。下方是同一段数据的交互式 3D 视图——鼠标拖拽旋转，滚轮缩放，▶ 播放或拖动滑块查看任意帧。</p>")
        parts.append("<div class='side-by-side'>")
        parts.append("<figure>"
                     f"<img src='data:image/gif;base64,{r['gif_b64']}'>"
                     "<figcaption>视频（primary | wrist）</figcaption>"
                     "</figure>")
        if r.get("traj_anim_b64"):
            parts.append("<figure>"
                         f"<img src='data:image/gif;base64,{r['traj_anim_b64']}'>"
                         "<figcaption>EE 位置 + 姿态坐标系 + action 轴角增量（品红，GIF，与视频同步）</figcaption>"
                         "</figure>")
        parts.append("</div>")
        if r.get("traj_anim_plotly"):
            div_id = f"traj-anim-{r['name']}"
            json_id = f"traj-anim-json-{r['name']}"
            fig_json = json.dumps(r["traj_anim_plotly"])
            parts.append(
                "<figure style='margin:12px 0 0;'>"
                f"<div id='{div_id}' style='width:100%;height:560px;'></div>"
                # Embed the figure spec as JSON in a script tag — sidesteps escaping
                # issues that come up with inline JS strings containing quotes.
                f"<script type='application/json' id='{json_id}'>{fig_json}</script>"
                "<script>(function(){"
                f"var fig=JSON.parse(document.getElementById('{json_id}').textContent);"
                f"Plotly.newPlot('{div_id}',fig.data,fig.layout,{{responsive:true,displaylogo:false}})"
                f".then(function(){{Plotly.addFrames('{div_id}',fig.frames);}});"
                "})();</script>"
                "<figcaption>EE 位置 + 姿态坐标系 + action 轴角增量（品红；交互式：拖拽旋转，滑块定格）</figcaption>"
                "</figure>"
            )

        # time series
        parts.append("<h3>⑤ Action 7 通道时序</h3>")
        parts.append(f"<img src='data:image/png;base64,{r['ts_b64']}'>")

        parts.append("<hr>")

    parts.append("</body></html>")
    out_path.write_text("".join(parts), encoding="utf-8")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Verify OXE → LeRobot v2 conversion.")
    ap.add_argument("--format", choices=["ours", "openvla"], default="ours",
                    help="which conversion to verify (reads data/lerobot or "
                         "data/lerobot_openvla accordingly).")
    ap.add_argument("--out", default=None, help="output HTML filename (under data/verification).")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="subset of dataset short names; default = all in VERIFY.")
    args = ap.parse_args()

    out_dir = ROOT / "data" / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    names = args.datasets or VERIFY
    results = []
    for name in names:
        print(f"=== {name} ({args.format}) ===")
        results.append(verify_one(name, fmt=args.format))

    default_out = "report_openvla.html" if args.format == "openvla" else "report_rotvec.html"
    out = out_dir / (args.out or default_out)
    render_html(results, out, fmt=args.format)
    print(f"\nWrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
