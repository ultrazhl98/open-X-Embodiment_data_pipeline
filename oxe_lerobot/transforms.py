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


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product of two unit quaternions in (w, x, y, z) order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    """Conjugate (== inverse for unit quat). Input/output in (w, x, y, z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_relative_wxyz(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """Rotation that takes q_from to q_to:  q_rel = q_to · q_from⁻¹.
    Both inputs and output are (w, x, y, z) unit quaternions."""
    return quat_mul_wxyz(q_to, quat_conj_wxyz(q_from))


def euler_xyz_to_quat_wxyz(rpy: np.ndarray) -> np.ndarray:
    """XYZ Tait-Bryan euler (in radians) → (w, x, y, z) unit quaternion.
    Matches the convention used by `_quat_wxyz_to_euler_xyz` (R = Rz·Ry·Rx)."""
    r = float(rpy[0]); p = float(rpy[1]); y = float(rpy[2])
    cr, sr = np.cos(r / 2.0), np.sin(r / 2.0)
    cp, sp = np.cos(p / 2.0), np.sin(p / 2.0)
    cy, sy = np.cos(y / 2.0), np.sin(y / 2.0)
    return np.array([
        cy * cp * cr + sy * sp * sr,
        cy * cp * sr - sy * sp * cr,
        cy * sp * cr + sy * cp * sr,
        sy * cp * cr - cy * sp * sr,
    ], dtype=np.float64)


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → unit quaternion (w, x, y, z) via Shepperd's method.

    Numerically stable: picks the largest of (1+tr, 1+R00-R11-R22, ...) as the
    denominator so we never divide by ~0 when one component is near zero.
    """
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def homogeneous16_to_pos_quat(mat16: np.ndarray, column_major: bool = True) -> np.ndarray:
    """4x4 matrix flattened to 16 → (3 pos, 4 quat_wxyz). Column-major by default."""
    m = np.asarray(mat16, dtype=np.float64).reshape(4, 4)
    if column_major:
        m = m.T
    pos = m[:3, 3]
    R = m[:3, :3]
    q = rotmat_to_quat_wxyz(R)
    return np.concatenate([pos, q]).astype(np.float64)


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
    """DEPRECATED — kept only for reference / backwards compat.

    Original implementation treated furniture_bench's `quat velocity` field as
    if it were a quaternion and ran it through quat→euler. That is
    geometrically meaningless (a quaternion derivative is not a unit quat).
    Use `furniture_state_delta` instead, which derives the action from
    successive state EE poses.
    """
    raise RuntimeError("delta_furniture_8d is deprecated — use furniture_state_delta")


def action_furniture_state_delta(step, prev_state) -> np.ndarray:
    """For furniture_bench: pack [ee_pos(3), ee_quat_wxyz(4), gripper_width(1)] from
    the observation state. The finalize pass computes the 7D delta from
    successive packed states."""
    obs = step["observation"]
    state = _ensure_1d(obs["state"])
    assert state.shape == (35,), f"expected 35D state, got {state.shape}"
    ee_pos  = state[0:3]
    ee_quat = state[3:7]          # (w, x, y, z) — verified empirically
    grip    = state[34:35]
    return np.concatenate([ee_pos, ee_quat, grip]).astype(np.float32)


def action_viola_state_delta(step, prev_state) -> np.ndarray:
    """viola: pack [ee_pos(3), ee_quat_wxyz(4), gripper(1)]. Pose from
    obs.ee_states (column-major 4×4); gripper from raw
    action.gripper_closedness_action (the COMMAND, mirrors what we do for the
    other *_state_delta datasets — gripper command is unaffected by OSC gain
    issues so we keep the original command signal)."""
    obs = step["observation"]
    pos_quat = homogeneous16_to_pos_quat(obs["ee_states"], column_major=True)  # 7D
    grip = _ensure_1d(step["action"]["gripper_closedness_action"])[:1]
    return np.concatenate([pos_quat, grip]).astype(np.float32)


def action_stanford_hydra_state_delta(step, prev_state) -> np.ndarray:
    """stanford_hydra: state is 27D [pos(3), quat_wxyz(4), euler(3), 7 joint,
    7 joint_vel, 3 gripper]. Pack [pos, quat, action[6]=gripper_close] — we use
    the raw action's gripper command (not state) because the state's 3 gripper
    channels are joint-level (finger positions), whereas action[6] is the
    command. Finalize derives delta."""
    obs = step["observation"]
    state = _ensure_1d(obs["state"])
    pos = state[0:3]
    quat_wxyz = state[3:7]
    raw_a = _ensure_1d(step["action"])
    grip = raw_a[6:7]
    return np.concatenate([pos, quat_wxyz, grip]).astype(np.float32)


def action_state_homog_at_slice_state_delta(step, prev_state, slice_start: int = 8) -> np.ndarray:
    """For austin_buds / utaustin_mutex: state is 24D [7 joint, 1 gripper, 16D
    column-major homog matrix]. Pack [pos, quat_wxyz, action[6]=grip]."""
    obs = step["observation"]
    state = _ensure_1d(obs["state"])
    mat16 = state[slice_start:slice_start + 16]
    if np.linalg.norm(mat16) < 1e-6:
        # degenerate frame — emit zeros; finalize will detect and emit zero delta
        pos_quat = np.zeros(7, dtype=np.float64)
    else:
        pos_quat = homogeneous16_to_pos_quat(mat16, column_major=True)
    raw_a = _ensure_1d(step["action"])
    grip = raw_a[6:7]
    return np.concatenate([pos_quat, grip]).astype(np.float32)


def action_state_ee_field_state_delta(step, prev_state) -> np.ndarray:
    """For austin_sailor / austin_sirius: obs.state_ee is 16D column-major homog
    matrix. Pack [pos, quat_wxyz, action[6]=grip]."""
    obs = step["observation"]
    mat16 = _ensure_1d(obs["state_ee"])
    if np.linalg.norm(mat16) < 1e-6:
        pos_quat = np.zeros(7, dtype=np.float64)
    else:
        pos_quat = homogeneous16_to_pos_quat(mat16, column_major=True)
    raw_a = _ensure_1d(step["action"])
    grip = raw_a[6:7]
    return np.concatenate([pos_quat, grip]).astype(np.float32)


def action_austin_buds_state_delta(step, prev_state) -> np.ndarray:
    return action_state_homog_at_slice_state_delta(step, prev_state, slice_start=8)


def action_utaustin_mutex_state_delta(step, prev_state) -> np.ndarray:
    return action_state_homog_at_slice_state_delta(step, prev_state, slice_start=8)


def action_austin_sailor_state_delta(step, prev_state) -> np.ndarray:
    return action_state_ee_field_state_delta(step, prev_state)


def action_austin_sirius_state_delta(step, prev_state) -> np.ndarray:
    return action_state_ee_field_state_delta(step, prev_state)


def action_abs_euler_7d(step, prev_state) -> np.ndarray:
    # ucsd_kitchen / cmu_franka_exploration: 8D absolute [xyz, euler, grip, terminate]
    # → drop terminate, return 7D (treated as either absolute or already-delta per dataset notes)
    a = _ensure_1d(step["action"])
    assert a.shape == (8,), f"expected 8D, got {a.shape}"
    return a[:7].astype(np.float32)


def action_abs_quat_7d(step, prev_state) -> np.ndarray:
    """For iamlab_cmu_pickup_insert: keep the raw 8D `[xyz, quat_wxyz, gripper]`
    target pose. The finalize pass converts successive target poses to a 7D
    EEF delta using quaternion-relative rotation (avoids Euler ±π wrap).

    Important: iamlab quat order is **(w, x, y, z)** — verified empirically by
    observing index 1 ≈ 1.0 (gripper-pointing-down corresponds to 180° about
    x-axis, i.e. (0, 1, 0, 0) in (w, x, y, z)).  The previous assumption
    (x, y, z, w) was wrong and silently rotated the euler axes by one slot.
    """
    a = _ensure_1d(step["action"])
    assert a.shape == (8,), f"expected 8D, got {a.shape}"
    return a.astype(np.float32)  # finalize_actions_to_delta will diff successive quats


def action_delta_xyz_only_4d(step, prev_state) -> np.ndarray:
    # ucsd_pick_and_place: 4D [3 lin_vel, 1 gripper] — pad rot with zeros
    a = _ensure_1d(step["action"])
    assert a.shape == (4,), f"expected 4D, got {a.shape}"
    return np.array([a[0], a[1], a[2], 0.0, 0.0, 0.0, a[3]], dtype=np.float32)


ACTION_FNS = {
    "delta_7d":                 action_delta_7d,
    "delta_7d_pack":            action_delta_7d_pack,
    "taco_rel_world":           action_taco_rel_world,
    "joint_vel_plus_delta":     action_joint_vel_plus_delta,
    "furniture_state_delta":    action_furniture_state_delta,
    "abs_euler_7d":             action_abs_euler_7d,            # cmu_franka_exploration: values are deltas
    "abs_euler_7d_to_delta":    action_abs_euler_7d,            # ucsd_kitchen: finalize does abs→delta
    "abs_quat_7d":              action_abs_quat_7d,
    "delta_xyz_only_4d":        action_delta_xyz_only_4d,
    # State-derived deltas — per-step fn packs [pos(3), quat_wxyz(4), grip(1)];
    # finalize diffs successive packed states (look-ahead).
    "viola_state_delta":            action_viola_state_delta,
    "stanford_hydra_state_delta":   action_stanford_hydra_state_delta,
    "austin_buds_state_delta":      action_austin_buds_state_delta,
    "utaustin_mutex_state_delta":   action_utaustin_mutex_state_delta,
    "austin_sailor_state_delta":    action_austin_sailor_state_delta,
    "austin_sirius_state_delta":    action_austin_sirius_state_delta,
}


# Action kinds whose per-step fn returns packed pose [pos(3), quat_wxyz(4), grip(1)]
# and whose finalize is _finalize_via_state_delta.
STATE_DELTA_KINDS = {
    "furniture_state_delta",
    "viola_state_delta",
    "stanford_hydra_state_delta",
    "austin_buds_state_delta",
    "utaustin_mutex_state_delta",
    "austin_sailor_state_delta",
    "austin_sirius_state_delta",
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


def _finalize_via_command_delta(canonical_steps: list[dict]) -> list[dict]:
    """For iamlab: each step's action is the raw 8D target pose
    `[xyz, quat_wxyz, gripper]`. Compute 7D delta with **look-ahead**
    convention to match OXE/Octo's `action[t] = state[t+1] - state[t]`:
        action[t] = command[t+1] - command[t]   for  t < N-1
        action[N-1] = 0 (no next target)
    Rotation uses quaternion-relative math (q_rel = q_next · q_curr⁻¹) and a
    single quat→euler — avoids ±π Euler wrap."""
    n = len(canonical_steps)
    out = []
    for t, s in enumerate(canonical_steps):
        raw = np.asarray(s["action"], dtype=np.float64)  # 8D
        if t == n - 1:
            delta = np.zeros(7, dtype=np.float32)
            delta[6] = float(raw[7])
        else:
            nxt = np.asarray(canonical_steps[t + 1]["action"], dtype=np.float64)
            dxyz = nxt[:3] - raw[:3]
            q_rel = quat_relative_wxyz(raw[3:7], nxt[3:7])
            drpy = _quat_wxyz_to_euler_xyz(q_rel)
            delta = np.concatenate([dxyz, drpy, [raw[7]]]).astype(np.float32)
        out.append({**s, "action": delta})
    return out


def _is_degenerate_pose(packed8: np.ndarray) -> bool:
    """A packed [pos(3), quat(4), grip(1)] is degenerate if both pos and quat
    are all-zero (some austin_sirius frames have an all-zero state_ee matrix)."""
    return (np.linalg.norm(packed8[:3]) < 1e-6
            and np.linalg.norm(packed8[3:7]) < 1e-6)


def _finalize_via_state_delta(canonical_steps: list[dict]) -> list[dict]:
    """For furniture_bench / *_state_delta datasets: each step's action is the
    packed `[ee_pos, ee_quat_wxyz, gripper]` from the state observation.
    Compute 7D delta to the NEXT state (not previous) — i.e. action[t] is the
    motion that takes state[t] → state[t+1]. This is the standard OXE/Octo
    convention.

    Last frame has no next state → emit zero delta + current gripper.
    Robust to degenerate (all-zero) pose entries: emits zero delta if either
    the current or next packed pose is degenerate (only seen in austin_sirius
    and austin_buds; corresponds to missing state_ee logging frames).
    """
    n = len(canonical_steps)
    out = []
    for t, s in enumerate(canonical_steps):
        raw = np.asarray(s["action"], dtype=np.float64)  # 8D from state
        if t == n - 1:
            delta = np.zeros(7, dtype=np.float32)
            delta[6] = float(raw[7])
        else:
            nxt = np.asarray(canonical_steps[t + 1]["action"], dtype=np.float64)
            if _is_degenerate_pose(raw) or _is_degenerate_pose(nxt):
                delta = np.zeros(7, dtype=np.float32)
                delta[6] = float(raw[7])
            else:
                dxyz = nxt[:3] - raw[:3]
                q_rel = quat_relative_wxyz(raw[3:7], nxt[3:7])
                drpy = _quat_wxyz_to_euler_xyz(q_rel)
                delta = np.concatenate([dxyz, drpy, [raw[7]]]).astype(np.float32)
        out.append({**s, "action": delta})
    return out


def _quat_wxyz_to_euler_xyz(q: np.ndarray) -> np.ndarray:
    """Same math as quat_to_euler_xyz, but on (w,x,y,z) directly. Returns
    XYZ Tait-Bryan euler (roll, pitch, yaw) as float32."""
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _finalize_via_euler_to_delta(canonical_steps: list[dict],
                                 pos_scale: float = 1.0,
                                 rot_in_degrees: bool = False) -> list[dict]:
    """For ucsd_kitchen-style datasets where the action is an absolute pose
    `[xyz, euler, gripper]` (after dropping `terminate`).  Convert to 7D delta
    with **look-ahead** convention.  Rotation goes through quaternions
    (euler→quat, q_rel, quat→euler) so ±π wraparound doesn't bite.  Optional
    unit normalization: ucsd_kitchen stores position in mm and rotation in
    degrees, so we pass `pos_scale=1e-3, rot_in_degrees=True` to land in
    standard SI units (m + rad)."""
    n = len(canonical_steps)
    out = []
    deg2rad = np.pi / 180.0 if rot_in_degrees else 1.0
    for t, s in enumerate(canonical_steps):
        raw = np.asarray(s["action"], dtype=np.float64)  # 7D [pos, euler, grip]
        if t == n - 1:
            delta = np.zeros(7, dtype=np.float32)
            delta[6] = float(raw[6])  # gripper as-is
        else:
            nxt = np.asarray(canonical_steps[t + 1]["action"], dtype=np.float64)
            dxyz = (nxt[:3] - raw[:3]) * pos_scale
            e_curr = raw[3:6] * deg2rad
            e_next = nxt[3:6] * deg2rad
            q_curr = euler_xyz_to_quat_wxyz(e_curr)
            q_next = euler_xyz_to_quat_wxyz(e_next)
            q_rel = quat_relative_wxyz(q_curr, q_next)
            drpy = _quat_wxyz_to_euler_xyz(q_rel)         # radians
            delta = np.concatenate([dxyz, drpy, [raw[6]]]).astype(np.float32)
        out.append({**s, "action": delta})
    return out


def finalize_actions_to_delta(canonical_steps: list[dict], cfg) -> list[dict]:
    """Second pass to compute the final 7D delta action for datasets whose
    per-step `action_fn` only emitted an intermediate (raw target pose or
    packed state). No-op for already-delta kinds."""
    if cfg.action_kind in STATE_DELTA_KINDS:
        return _finalize_via_state_delta(canonical_steps)
    if cfg.action_kind == "abs_quat_7d":
        return _finalize_via_command_delta(canonical_steps)
    if cfg.action_kind == "abs_euler_7d_to_delta":
        # ucsd_kitchen: 8D abs [xyz(mm), euler(deg), grip, terminate] → 7D delta in SI units.
        return _finalize_via_euler_to_delta(canonical_steps,
                                            pos_scale=1e-3,
                                            rot_in_degrees=True)
    return canonical_steps
