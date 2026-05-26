"""OpenVLA action conversion — faithful NumPy port of OpenVLA's per-dataset
RLDS `standardize_fn`s (prismatic/vla/datasets/rlds/oxe/transforms.py).

This is the *alternative* action-format pipeline used by `convert.py
--action-format openvla`, intended for an apples-to-apples experiment against
the project's own state-derived 7D-delta format (transforms.py).

Key differences from our own pipeline — these are intentional, we replicate
OpenVLA exactly, warts and all:

  * Per-step, raw-action-trusting. OpenVLA takes the dataset's *raw* action
    field, slices the relevant 6 motion DOF, and uses them as-is. It does NOT
    derive the action from successive state EE poses, does NOT do quaternion-
    relative rotation, and does NOT normalize units (e.g. ucsd_kitchen stays in
    mm / deg; taco_play stays in CALVIN-normalized command units).
  * Rotation is whatever the raw action stores. For the two datasets whose raw
    action carries a quaternion (furniture_bench, iamlab) OpenVLA converts it to
    XYZ-Tait-Bryan **euler** via `tensorflow_graphics.euler.from_quaternion`
    (NOT axis-angle). We reproduce that exact convention below.
  * Absolute poses stay absolute. OpenVLA does NOT convert iamlab's or
    ucsd_kitchen's absolute target poses to deltas — it trains on the raw
    absolute pose. We keep that.
  * Only the gripper channel is massaged, with per-dataset invert / clip /
    binarize / rel2abs (see helpers below) so that +1 = open, 0 = close.

Each transform takes the full list of raw RLDS step dicts for one trajectory
(needed because the gripper binarize/rel2abs ops are trajectory-scoped scans)
and returns an `(N, 7)` float32 array `[dx, dy, dz, drx, dry, drz, gripper]`.

tfg quaternion order note: `tensorflow_graphics` uses scalar-LAST quaternions
`(x, y, z, w)`, and `euler.from_quaternion` returns `(theta_x, theta_y,
theta_z)` for R = Rz·Ry·Rx — identical math to our `quat_to_euler_xyz` once the
quaternion is reordered to `(w, x, y, z)`. So `quat_xyzw_to_euler` below =
exactly what OpenVLA computes. (This deliberately assumes the raw quat is
stored xyzw, matching OpenVLA's assumption — even where our own pipeline found
the dataset actually stores wxyz. The experiment is to compare OpenVLA's
recipe as-is.)
"""
from __future__ import annotations

import numpy as np

from .transforms import quat_to_euler_xyz


# ---------------------------- small helpers ----------------------------

def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(-1)


def _stack_action(steps: list[dict]) -> np.ndarray:
    """Stack the flat `action` array across a trajectory → (N, D)."""
    return np.stack([_arr(s["action"]) for s in steps]).astype(np.float32)


def quat_xyzw_to_euler(quat_xyzw: np.ndarray) -> np.ndarray:
    """(..., 4) scalar-last quaternion → XYZ euler, matching
    `tensorflow_graphics.geometry.transformation.euler.from_quaternion`."""
    q = np.asarray(quat_xyzw, dtype=np.float64)
    q_wxyz = q[..., [3, 0, 1, 2]]
    return quat_to_euler_xyz(q_wxyz)


# ---------------------------- gripper ops (NumPy ports) ----------------------------

def invert_gripper_actions(a: np.ndarray) -> np.ndarray:
    return 1.0 - a


def binarize_gripper_actions(actions: np.ndarray) -> np.ndarray:
    """Port of OpenVLA's `binarize_gripper_actions` (reverse scan). 1D in/out.

    Relabels intermediate gripper values to the *next* settled state; carry is
    seeded with the raw last action (matches the TF `tf.scan(..., actions[-1],
    reverse=True)`)."""
    actions = np.asarray(actions, dtype=np.float64).reshape(-1)
    n = actions.shape[0]
    open_mask = actions > 0.95
    closed_mask = actions < 0.05
    in_between = ~(open_mask | closed_mask)
    is_open = open_mask.astype(np.float64)
    out = np.empty(n, dtype=np.float64)
    carry = actions[-1]
    for i in range(n - 1, -1, -1):
        carry = carry if in_between[i] else is_open[i]
        out[i] = carry
    return out.astype(np.float32)


def rel2abs_gripper_actions(actions: np.ndarray) -> np.ndarray:
    """Port of OpenVLA's `rel2abs_gripper_actions` (forward scan). 1D in/out.
    Relative (+1 close / -1 open) → absolute (0 close / 1 open)."""
    actions = np.asarray(actions, dtype=np.float64).reshape(-1)
    n = actions.shape[0]
    opening = actions < -0.1
    closing = actions > 0.1
    thr = np.where(opening, 1.0, np.where(closing, -1.0, 0.0))
    nz = np.nonzero(thr != 0)[0]
    if nz.size == 0:
        start = 1.0
    else:
        start = -1.0 * thr[nz[0]]
        if start == 0:
            start = 1.0
    out = np.empty(n, dtype=np.float64)
    carry = start
    for i in range(n):
        carry = carry if thr[i] == 0 else thr[i]
        out[i] = carry
    return (out / 2.0 + 0.5).astype(np.float32)


# ---------------------------- per-dataset transforms ----------------------------

def ov_viola(steps: list[dict]) -> np.ndarray:
    """world_vector(3) + rotation_delta(3) + invert(clip(gripper, 0, 1))."""
    wv = np.stack([_arr(s["action"]["world_vector"]) for s in steps])
    rd = np.stack([_arr(s["action"]["rotation_delta"]) for s in steps])
    g = np.stack([_arr(s["action"]["gripper_closedness_action"])[:1] for s in steps]).reshape(-1)
    g = invert_gripper_actions(np.clip(g, 0.0, 1.0))
    return np.concatenate([wv, rd, g[:, None]], axis=1).astype(np.float32)


def ov_invert_gripper_7d(steps: list[dict], clip: bool) -> np.ndarray:
    """7D raw delta action; keep [:6], invert (optionally clip) the gripper.
    stanford_hydra (no clip); austin_buds / sailor / sirius / utaustin_mutex (clip)."""
    a = _stack_action(steps)
    g = a[:, -1:]
    if clip:
        g = np.clip(g, 0.0, 1.0)
    g = invert_gripper_actions(g)
    return np.concatenate([a[:, :6], g], axis=1).astype(np.float32)


def ov_furniture(steps: list[dict]) -> np.ndarray:
    """8D [xyz, quat_xyzw, gripper] → [xyz, euler(quat), invert(clip(grip,0,1))]."""
    a = _stack_action(steps)
    eul = quat_xyzw_to_euler(a[:, 3:7])
    g = invert_gripper_actions(np.clip(a[:, -1:], 0.0, 1.0))
    return np.concatenate([a[:, :3], eul, g], axis=1).astype(np.float32)


def ov_iamlab(steps: list[dict]) -> np.ndarray:
    """8D ABSOLUTE [xyz, quat_xyzw, gripper] → [xyz, euler(quat), gripper] —
    kept absolute (OpenVLA does NOT convert to delta), gripper untouched."""
    a = _stack_action(steps)
    eul = quat_xyzw_to_euler(a[:, 3:7])
    return np.concatenate([a[:, :3], eul, a[:, 7:8]], axis=1).astype(np.float32)


def ov_taco(steps: list[dict]) -> np.ndarray:
    """rel_actions_world: [:6] + clip(gripper, 0, 1) (no invert)."""
    a = np.stack([_arr(s["action"]["rel_actions_world"]) for s in steps]).astype(np.float32)
    g = np.clip(a[:, 6:7], 0.0, 1.0)
    return np.concatenate([a[:, :6], g], axis=1).astype(np.float32)


def ov_nyu_franka(steps: list[dict]) -> np.ndarray:
    """15D action → [a[-8:-2] (Δxyz, Δrpy), clip(a[-2:-1] gripper, 0, 1)]."""
    a = _stack_action(steps)
    return np.concatenate([a[:, -8:-2], np.clip(a[:, -2:-1], 0.0, 1.0)], axis=1).astype(np.float32)


def ov_drop_terminate(steps: list[dict]) -> np.ndarray:
    """8D [..., terminate] → drop last column → 7D, as-is (no unit/delta change).
    ucsd_kitchen (absolute pose, mm/deg kept) and cmu_franka_exploration (delta)."""
    a = _stack_action(steps)
    return a[:, :-1].astype(np.float32)


def ov_ucsd_pick(steps: list[dict]) -> np.ndarray:
    """4D [lin_vel(3), gripper] → [lin_vel(3), zeros(3), gripper]."""
    a = _stack_action(steps)
    z = np.zeros((a.shape[0], 3), dtype=np.float32)
    return np.concatenate([a[:, :3], z, a[:, -1:]], axis=1).astype(np.float32)


# Keyed by the SHORT dataset name (DatasetConfig.name).
OPENVLA_ACTION_FNS = {
    "viola":                  ov_viola,
    "stanford_hydra":         lambda s: ov_invert_gripper_7d(s, clip=False),
    "austin_buds":            lambda s: ov_invert_gripper_7d(s, clip=True),
    "austin_sailor":          lambda s: ov_invert_gripper_7d(s, clip=True),
    "austin_sirius":          lambda s: ov_invert_gripper_7d(s, clip=True),
    "utaustin_mutex":         lambda s: ov_invert_gripper_7d(s, clip=True),
    "furniture_bench":        ov_furniture,
    "iamlab_cmu_pickup_insert": ov_iamlab,
    "taco_play":              ov_taco,
    "nyu_franka_play":        ov_nyu_franka,
    "ucsd_kitchen":           ov_drop_terminate,
    "ucsd_pick_and_place":    ov_ucsd_pick,
    "cmu_franka_exploration": ov_drop_terminate,
}


# Short per-dataset note on the action semantics, recorded into info.json so the
# resulting LeRobot dataset is self-describing.
OPENVLA_ACTION_NOTES = {
    "viola":                  "raw world_vector(3)+rotation_delta(3) OSC delta; gripper invert(clip).",
    "stanford_hydra":         "raw 7D OSC delta [Δxyz, Δrpy]; gripper inverted.",
    "austin_buds":            "raw 7D OSC delta [Δxyz, Δrpy]; gripper invert(clip).",
    "austin_sailor":          "raw 7D OSC delta [Δxyz, Δrpy]; gripper invert(clip).",
    "austin_sirius":          "raw 7D OSC delta [Δxyz, Δrpy]; gripper invert(clip).",
    "utaustin_mutex":         "raw 7D OSC delta [Δxyz, Δrpy]; gripper invert(clip).",
    "furniture_bench":        "raw [Δxyz, quat_xyzw→euler, gripper]; quat treated xyzw; gripper invert(clip).",
    "iamlab_cmu_pickup_insert": "raw ABSOLUTE pose [xyz, quat_xyzw→euler, gripper]; kept absolute (NOT delta).",
    "taco_play":              "raw rel_actions_world [Δxyz, Δrpy] (CALVIN-normalized units); gripper clip(0,1).",
    "nyu_franka_play":        "raw 15D action sliced to [Δxyz, Δrpy, gripper]; gripper clipped.",
    "ucsd_kitchen":           "raw ABSOLUTE pose [xyz(mm), euler(deg), gripper] minus terminate; kept absolute, units unchanged.",
    "ucsd_pick_and_place":    "raw [lin_vel(3), gripper] with rotation zero-padded.",
    "cmu_franka_exploration": "raw 7D delta [Δxyz, Δrpy] minus terminate.",
}


def _raw_grip_col(steps: list[dict], idx: int) -> np.ndarray:
    return np.stack([_arr(s["action"])[idx] for s in steps]).astype(np.float32)


def unit_gripper_for_trajectory(cfg, raw_steps: list[dict]) -> np.ndarray:
    """Return an `(N,)` gripper signal normalized to **[0, 1] with +1=open /
    0=close** for one trajectory.

    Reuses OpenVLA's per-dataset gripper convention (invert / clip) so that the
    project's own ('ours') action format can carry the SAME gripper semantics as
    the OpenVLA format — only the motion channels then differ between the two.

    For the four datasets OpenVLA leaves untouched we still clamp to [0, 1] to
    honor the unit requirement:
      * iamlab / ucsd_kitchen / cmu_franka: raw gripper is already 0=close /
        1=open, so just clip (no invert).
      * ucsd_pick_and_place: raw gripper is a torque in [-1, 1]; empirically
        +torque opens the fingers (corr with Δfinger-distance > 0), so map
        (t+1)/2 → open=1 / close=0.
    """
    name = cfg.name
    if name == "viola":
        g = np.stack([_arr(s["action"]["gripper_closedness_action"])[:1]
                      for s in raw_steps]).reshape(-1)
        return invert_gripper_actions(np.clip(g, 0.0, 1.0))
    if name == "stanford_hydra":
        return invert_gripper_actions(_raw_grip_col(raw_steps, -1))          # 7D, OpenVLA: no clip
    if name in ("austin_buds", "austin_sailor", "austin_sirius",
                "utaustin_mutex", "furniture_bench"):
        # austin_*/mutex: raw action[-1]; furniture: action[-1] command (NOT the
        # state width 'ours' used before — switched so the gripper lands in [0,1]).
        return invert_gripper_actions(np.clip(_raw_grip_col(raw_steps, -1), 0.0, 1.0))
    if name == "taco_play":
        g = np.stack([_arr(s["action"]["rel_actions_world"])[6] for s in raw_steps]).astype(np.float32)
        return np.clip(g, 0.0, 1.0)                                          # CALVIN: +1=open, no invert
    if name == "nyu_franka_play":
        return np.clip(_raw_grip_col(raw_steps, -2), 0.0, 1.0)               # action[-2], no invert
    if name in ("iamlab_cmu_pickup_insert",):
        return np.clip(_raw_grip_col(raw_steps, 7), 0.0, 1.0)                # already 0=close/1=open
    if name in ("ucsd_kitchen", "cmu_franka_exploration"):
        return np.clip(_raw_grip_col(raw_steps, 6), 0.0, 1.0)                # already 0=close/1=open
    if name == "ucsd_pick_and_place":
        t = _raw_grip_col(raw_steps, 3)                                      # torque [-1,1], +=open
        return np.clip((t + 1.0) / 2.0, 0.0, 1.0)
    raise KeyError(f"No unit-gripper rule for dataset '{name}'")


def openvla_actions_for_trajectory(cfg, raw_steps: list[dict]) -> np.ndarray:
    """Return the OpenVLA-format `(N, 7)` action array for one trajectory."""
    fn = OPENVLA_ACTION_FNS.get(cfg.name)
    if fn is None:
        raise KeyError(
            f"No OpenVLA action transform registered for dataset '{cfg.name}'. "
            f"Known: {sorted(OPENVLA_ACTION_FNS)}"
        )
    actions = fn(raw_steps)
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"[{cfg.name}] OpenVLA action must be (N,7), got {actions.shape}")
    return actions
