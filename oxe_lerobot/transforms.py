"""Per-dataset standardize_fn — pure NumPy versions of OpenVLA/Octo transforms.

Each `standardize_<kind>(step_dict, prev_state) -> dict` returns the canonical
schema described in __init__.py. `step_dict` is a plain Python dict (already
decoded out of TFDS / TFRecord), `prev_state` is the previous canonical step
(used only for absolute→delta conversion).
"""
from __future__ import annotations

import numpy as np


# ---------------------------- rotation helpers ----------------------------

def quat_to_euler_xyz(quat: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) → euler XYZ (roll, pitch, yaw). Pure NumPy."""
    q = np.asarray(quat, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quat must be (..., 4), got {q.shape}")
    # Many OXE datasets store (x,y,z,w). We accept both; assume (w,x,y,z) here
    # — call site converts if needed.
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    # roll (x-axis rotation)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)
    # yaw (z-axis rotation)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return np.stack([roll, pitch, yaw], axis=-1).astype(np.float32)


def homogeneous16_to_pos_euler(mat16: np.ndarray, column_major: bool = True) -> np.ndarray:
    """4x4 matrix flattened to 16 → (3 pos, 3 euler).

    Defaults to column-major (Fortran order) layout — this is the RobotSuite /
    MuJoCo convention used by VIOLA / Austin BUDS / SAILOR / etc. where the
    translation lives in the LAST ROW of the reshaped 4x4 instead of the last
    column. Pass `column_major=False` for already-standard row-major matrices.
    """
    m = np.asarray(mat16, dtype=np.float64).reshape(4, 4)
    if column_major:
        m = m.T
    pos = m[:3, 3]
    R = m[:3, :3]
    # XYZ Tait–Bryan
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([pos[0], pos[1], pos[2], roll, pitch, yaw], dtype=np.float32)


# ---------------------------- action standardizers ----------------------------

def _ensure_1d(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def action_delta_7d(step, prev_state) -> np.ndarray:
    a = _ensure_1d(step["action"])
    assert a.shape == (7,), f"expected 7D action, got {a.shape}"
    return a


def action_delta_7d_pack(step, prev_state) -> np.ndarray:
    # viola: action is a dict
    a = step["action"]
    return np.concatenate(
        [_ensure_1d(a["world_vector"]),
         _ensure_1d(a["rotation_delta"]),
         _ensure_1d(a["gripper_closedness_action"])]
    ).astype(np.float32)


def action_taco_rel_world(step, prev_state) -> np.ndarray:
    a = step["action"]
    return _ensure_1d(a["rel_actions_world"])  # already 7D


def action_joint_vel_plus_delta(step, prev_state) -> np.ndarray:
    # nyu_franka_play: 15D = [7 joint_vel, 3 Δxyz, 3 Δrpy, 1 gripper, 1 terminate]
    a = _ensure_1d(step["action"])
    assert a.shape == (15,), f"expected 15D, got {a.shape}"
    return np.concatenate([a[7:13], a[13:14]]).astype(np.float32)


def action_delta_furniture_8d(step, prev_state) -> np.ndarray:
    # furniture_bench: 8D [3 lin_vel, 4 quat_vel (w,x,y,z), 1 gripper] → 7D euler
    a = _ensure_1d(step["action"])
    assert a.shape == (8,), f"expected 8D, got {a.shape}"
    pos = a[:3]
    quat = a[3:7]
    grip = a[7:8]
    euler = quat_to_euler_xyz(quat)
    return np.concatenate([pos, euler, grip]).astype(np.float32)


def action_abs_euler_7d(step, prev_state) -> np.ndarray:
    # ucsd_kitchen / cmu_franka_exploration: 8D absolute [xyz, euler, grip, terminate]
    # → drop terminate, return 7D (treated as either absolute or already-delta per dataset notes)
    a = _ensure_1d(step["action"])
    assert a.shape == (8,), f"expected 8D, got {a.shape}"
    return a[:7].astype(np.float32)


def action_abs_quat_7d(step, prev_state) -> np.ndarray:
    # iamlab_cmu_pickup_insert: 8D absolute [xyz, quat(x,y,z,w), gripper]
    # Convert to current absolute (xyz, euler, gripper); caller will diff to prev.
    a = _ensure_1d(step["action"])
    assert a.shape == (8,), f"expected 8D, got {a.shape}"
    pos = a[:3]
    # NOTE: iamlab quat order is (x, y, z, w) per inspection of TFDS spec.
    qx, qy, qz, qw = a[3], a[4], a[5], a[6]
    euler = quat_to_euler_xyz(np.array([qw, qx, qy, qz]))
    grip = a[7:8]
    return np.concatenate([pos, euler, grip]).astype(np.float32)


def action_delta_xyz_only_4d(step, prev_state) -> np.ndarray:
    # ucsd_pick_and_place: 4D [3 lin_vel, 1 gripper] — pad rot with zeros
    a = _ensure_1d(step["action"])
    assert a.shape == (4,), f"expected 4D, got {a.shape}"
    return np.array([a[0], a[1], a[2], 0.0, 0.0, 0.0, a[3]], dtype=np.float32)


ACTION_FNS = {
    "delta_7d":              action_delta_7d,
    "delta_7d_pack":         action_delta_7d_pack,
    "taco_rel_world":        action_taco_rel_world,
    "joint_vel_plus_delta":  action_joint_vel_plus_delta,
    "delta_furniture_8d":    action_delta_furniture_8d,
    "abs_euler_7d":          action_abs_euler_7d,
    "abs_quat_7d":           action_abs_quat_7d,
    "delta_xyz_only_4d":     action_delta_xyz_only_4d,
}


# ---------------------------- state extractor ----------------------------

def extract_state(cfg, obs: dict) -> np.ndarray:
    """Return a 1D proprio vector. Falls back to a synthesized one when no
    canonical 'state' field exists (e.g. viola: ee_states + gripper_states)."""
    if cfg.state_key is not None and cfg.state_key in obs:
        return _ensure_1d(obs[cfg.state_key])
    # viola synthesis
    if "ee_states" in obs and "gripper_states" in obs:
        pos_euler = homogeneous16_to_pos_euler(obs["ee_states"])
        grip = _ensure_1d(obs["gripper_states"])
        return np.concatenate([pos_euler, grip]).astype(np.float32)  # 7D
    # cmu_franka_exploration has no state at all
    return np.zeros((1,), dtype=np.float32)


# ---------------------------- language extractor ----------------------------

def extract_language(cfg, step: dict) -> str:
    if cfg.language_in_observation:
        obs = step.get("observation", {})
        val = obs.get(cfg.language_key, "")
    else:
        val = step.get(cfg.language_key, "")
    if isinstance(val, bytes):
        val = val.decode("utf-8", errors="replace")
    return str(val)


# ---------------------------- top-level standardize ----------------------------

def standardize_step(cfg, raw_step: dict, prev_canonical: dict | None) -> dict:
    """Map one raw RLDS step (Python dict) to the canonical schema."""
    obs = raw_step["observation"]
    image_primary = np.asarray(obs[cfg.image_primary_key], dtype=np.uint8)
    image_wrist = None
    if cfg.image_wrist_key and cfg.image_wrist_key in obs:
        image_wrist = np.asarray(obs[cfg.image_wrist_key], dtype=np.uint8)
    image_secondary = None
    if cfg.image_secondary_key and cfg.image_secondary_key in obs:
        image_secondary = np.asarray(obs[cfg.image_secondary_key], dtype=np.uint8)

    state = extract_state(cfg, obs)
    action_raw = ACTION_FNS[cfg.action_kind](raw_step, prev_canonical)
    language = extract_language(cfg, raw_step)

    return {
        "image_primary": image_primary,
        "image_wrist": image_wrist,
        "image_secondary": image_secondary,
        "state": state,
        "action": action_raw.astype(np.float32),
        "language_instruction": language,
    }


def finalize_actions_to_delta(canonical_steps: list[dict], cfg) -> list[dict]:
    """Some datasets emit ABSOLUTE poses per step (iamlab); convert to delta in
    a second pass once the full episode is known. No-op for already-delta kinds."""
    if cfg.action_kind != "abs_quat_7d":
        return canonical_steps
    out = []
    prev = None
    for s in canonical_steps:
        a = s["action"].copy()
        if prev is None:
            d = np.zeros(7, dtype=np.float32)
        else:
            d = np.zeros(7, dtype=np.float32)
            d[:6] = a[:6] - prev[:6]
        d[6] = a[6]  # gripper stays absolute
        s = {**s, "action": d.astype(np.float32)}
        out.append(s)
        prev = a
    return out
