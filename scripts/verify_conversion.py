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
    _quat_wxyz_to_euler_xyz,
    euler_xyz_to_quat_wxyz,
    finalize_actions_to_delta,
    quat_relative_wxyz,
    standardize_step,
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


def _read_lerobot_ep(short_name: str, ep_index: int = 0) -> dict:
    root = ROOT / "data" / "lerobot" / short_name
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


def _read_rlds_first_episode(cfg) -> list[dict]:
    raw_dir = str(ROOT / "data" / "raw" / cfg.rlds_name / "0.1.0")
    for ep in iter_episodes(raw_dir, max_episodes=1):
        return ep["steps"]
    raise RuntimeError(f"no episode in {raw_dir}")


# ---------------------------- ① numerical equivalence ---------------------------- #

def independent_expected_action(cfg, raw_steps: list[dict]) -> np.ndarray:
    """Recompute the canonical action **directly from raw fields**, bypassing the
    standardize_step/finalize_actions_to_delta abstraction.  If this matches
    the parquet action, the abstraction is doing what we say it is."""

    if cfg.action_kind == "abs_quat_7d":
        # iamlab: raw 8D [xyz, quat_wxyz, gripper] → 7D delta with **look-ahead**
        # convention (action[t] = command[t+1] - command[t]). Last frame: zero.
        raws = np.stack([np.asarray(s["action"], dtype=np.float64) for s in raw_steps])
        n = len(raws)
        out = np.zeros((n, 7), dtype=np.float32)
        for t in range(n - 1):
            out[t, :3] = raws[t + 1, :3] - raws[t, :3]
            q_rel = quat_relative_wxyz(raws[t, 3:7], raws[t + 1, 3:7])
            out[t, 3:6] = _quat_wxyz_to_euler_xyz(q_rel)
            out[t, 6]   = raws[t, 7]
        out[n - 1, :6] = 0.0
        out[n - 1, 6]  = raws[n - 1, 7]
        return out

    if cfg.action_kind == "furniture_state_delta":
        # furniture_bench: state-derived deltas.
        states = np.stack(
            [np.asarray(s["observation"]["state"], dtype=np.float64) for s in raw_steps]
        )
        n = len(states)
        out = np.zeros((n, 7), dtype=np.float32)
        for t in range(n - 1):
            out[t, :3] = states[t + 1, :3] - states[t, :3]
            q_rel = quat_relative_wxyz(states[t, 3:7], states[t + 1, 3:7])
            out[t, 3:6] = _quat_wxyz_to_euler_xyz(q_rel)
            out[t, 6]   = states[t, 34]
        out[n - 1, :6] = 0.0
        out[n - 1, 6]  = states[n - 1, 34]
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
            out[t, 3:6] = _quat_wxyz_to_euler_xyz(q_rel)
            out[t, 6]   = raws[t, 6]
        out[n - 1, :6] = 0.0
        out[n - 1, 6]  = raws[n - 1, 6]
        return out

    # Already-delta kinds — just run the per-step function.
    fn = ACTION_FNS[cfg.action_kind]
    return np.stack([fn(s, None) for s in raw_steps])


def numerical_equivalence(cfg, raw_steps, parquet_action) -> dict:
    expected = independent_expected_action(cfg, raw_steps)
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
        # state[3:7] is (w, x, y, z) (largest component at index 0; verified)
        for t in range(n):
            R[t] = quat_wxyz_to_rotmat(lerobot_state[t, 3:7])
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


def animate_trajectory_with_frame(ee_xyz: np.ndarray,
                                  R_per_frame: np.ndarray,
                                  title: str,
                                  indices: list[int] | None = None,
                                  target_frames: int = 60,
                                  fps: int = 10) -> str:
    """Animated GIF: growing 3D trajectory + EE coordinate frame (x=R, y=G, z=B)."""
    n = len(ee_xyz)
    idx = indices if indices is not None else _downsample_indices(n, target_frames)

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    mins = ee_xyz.min(axis=0)
    maxs = ee_xyz.max(axis=0)
    center = (mins + maxs) / 2
    half_range = max(((maxs - mins).max() / 2) * 1.25, 0.05)
    axis_len = half_range * 0.18

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
        ax.set_xlim(center[0] - half_range, center[0] + half_range)
        ax.set_ylim(center[1] - half_range, center[1] + half_range)
        ax.set_zlim(center[2] - half_range, center[2] + half_range)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"{title}\nframe {i}/{n - 1}  (red=EE x, green=EE y, blue=EE z)")

    anim = FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=False)
    # matplotlib's anim.save needs a file path, not a BytesIO — use a temp file.
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

def verify_one(short_name: str) -> dict:
    cfg = get_config(short_name)
    print(f"  loading parquet…", flush=True)
    lerobot = _read_lerobot_ep(cfg.name, ep_index=0)
    print(f"  re-reading first RLDS episode…", flush=True)
    raw_steps = _read_rlds_first_episode(cfg)

    # ① numerical equivalence
    eq = numerical_equivalence(cfg, raw_steps, lerobot["action"])

    # ② correlation w/ EE velocity
    ee_xyz = extract_ee_xyz(cfg, lerobot["state"], raw_steps)
    corr = scatter_b64 = traj_b64 = traj_anim_b64 = None
    if ee_xyz is not None and len(ee_xyz) >= 3:
        corr = correlation_per_axis(lerobot["action"], ee_xyz)
        scatter_b64 = plot_action_vs_velocity(
            lerobot["action"], ee_xyz, f"{cfg.name}: action[:3] vs state-EE Δ"
        )
        integ = integrated_xyz(lerobot["action"], ee_xyz[0])
        traj_b64 = plot_trajectory_3d(
            ee_xyz, f"{cfg.name}: EE trajectory (solid=state, dashed=∑action+x₀)", integ
        )
        # animated 3D trajectory + EE coordinate frame (for visual comparison with the video)
        R_per_frame = extract_rotations(cfg, lerobot["state"], raw_steps)
        # Pick a common frame index list so the video GIF and the trajectory animation
        # play in sync — useful for visual comparison.
        common_indices = _downsample_indices(eq["n_frames"], target=50)
        shared_fps = 8
        if R_per_frame is not None:
            print("  rendering 3D trajectory animation…", flush=True)
            traj_anim_b64 = animate_trajectory_with_frame(
                ee_xyz, R_per_frame,
                title=f"{cfg.name}: animated EE pose",
                indices=common_indices, fps=shared_fps,
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
        "action_kind":    cfg.action_kind,
        "notes":          cfg.notes,
        "n_frames":       eq["n_frames"],
        "equivalence":    eq,
        "correlation":    corr,
        "scatter_b64":    scatter_b64,
        "traj_b64":       traj_b64,
        "traj_anim_b64":  traj_anim_b64,
        "ts_b64":         ts_b64,
        "gif_b64":        gif_b64,
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


def render_html(results: list[dict], out_path: Path) -> None:
    parts = ["<!doctype html><html><head><meta charset='utf-8'>",
             "<title>OXE → LeRobot v2 验证报告</title>",
             f"<style>{CSS}</style></head><body>",
             "<h1>OXE → LeRobot v2 转换验证报告</h1>",
             "<p>每个数据集跑 4 项检查：① 数值一致性 ② action 与 EE 速度相关性 ③ 时序图 ④ 视频回放。</p>"]

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

        # Side-by-side: video playback + animated trajectory with EE frame.
        # Both use the same frame indices + fps, so they play in sync.
        parts.append("<h3>④ 视频回放  ↔  末端轨迹 + 姿态动画（同步播放）</h3>")
        parts.append("<p class='muted'>左边是 primary + wrist 相机视频，右边是 EE 位置（蓝线）+ 末端坐标系（红=x，绿=y，蓝=z）。两段动画使用相同的帧索引和 fps，可以同时观察「视频里 gripper 的位置朝向」和「我们重构出的 EE 位姿」是否一致。</p>")
        parts.append("<div class='side-by-side'>")
        parts.append("<figure>"
                     f"<img src='data:image/gif;base64,{r['gif_b64']}'>"
                     "<figcaption>视频（primary | wrist）</figcaption>"
                     "</figure>")
        if r.get("traj_anim_b64"):
            parts.append("<figure>"
                         f"<img src='data:image/gif;base64,{r['traj_anim_b64']}'>"
                         "<figcaption>EE 位置 + 姿态坐标系</figcaption>"
                         "</figure>")
        parts.append("</div>")

        # time series
        parts.append("<h3>⑤ Action 7 通道时序</h3>")
        parts.append(f"<img src='data:image/png;base64,{r['ts_b64']}'>")

        parts.append("<hr>")

    parts.append("</body></html>")
    out_path.write_text("".join(parts), encoding="utf-8")


def main():
    out_dir = ROOT / "data" / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name in VERIFY:
        print(f"=== {name} ===")
        results.append(verify_one(name))

    out = out_dir / "report.html"
    render_html(results, out)
    print(f"\nWrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
