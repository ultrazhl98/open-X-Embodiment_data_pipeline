"""Per-dataset config for the 13 OXE Franka subsets used in this project.

Mirrors the role of OpenVLA's `configs.py` and Octo's `oxe_dataset_configs.py`:
each entry describes how to pull the canonical fields out of a raw RLDS step.

Action conversion kinds:
  - "delta_7d":         action is already 7D [Δxyz, Δrpy, gripper] — use as-is.
  - "delta_7d_pack":    action is a dict with separate world_vector/rotation_delta/gripper.
  - "abs_quat_7d":      action is 8D absolute [xyz, quat_wxyz, gripper] — finalize pass
                        diffs successive commands using quaternion-relative math.
  - "abs_euler_7d":     action is 8D [xyz, euler, gripper, terminate] — drop terminate.
                        Used by cmu_franka_exploration where values are already deltas.
  - "abs_euler_7d_to_delta": action is 8D ABSOLUTE [xyz, euler, gripper, terminate].
                        Finalize converts to 7D delta with quat-relative rotation and
                        optional unit normalization (mm→m, deg→rad). Used by ucsd_kitchen.
  - "joint_vel_plus_delta": action is 15D, slice [7:13] (Δxyz, Δrpy) + [13] (gripper) → 7D.
  - "delta_xyz_only_4d":    action is 4D [3 lin_vel, 1 gripper] — pad rotation with zeros.
  - "taco_state_delta": action is rel_actions_world (CALVIN-normalized, ~50× larger
                        than physical motion), so we replace it with state-delta
                        derived from robot_obs (pos + euler) — same pattern as
                        viola/stanford_hydra/austin_* — to keep units consistent
                        (m + rad) across all 13 datasets. Gripper command kept
                        from rel_actions_world[6] (±1).
  - "*_state_delta" family: action derived from state ΔEE pose. Per-step fn packs
                        [pos(3), quat_wxyz(4), grip(1)] from the state observation; finalize
                        diffs successive packed states (look-ahead). Used by furniture_bench,
                        viola, stanford_hydra, austin_buds/sailor/sirius, utaustin_mutex —
                        all datasets where the raw action field is a controller command
                        whose magnitude differs from physical motion.

FPS values come from each dataset's published source paper / OXE configs.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DatasetConfig:
    name: str
    rlds_name: str
    fps: int
    image_primary_key: str           # under steps/observation/*
    image_wrist_key: Optional[str]   # under steps/observation/*  (None if absent)
    state_key: Optional[str]         # under steps/observation/*
    action_kind: str
    # Some datasets nest language_instruction differently:
    language_key: str = "language_instruction"          # default: steps/language_instruction
    language_in_observation: bool = False               # True ⇒ steps/observation/<language_key>
    # Optional secondary view name if no wrist exists but a second view does
    image_secondary_key: Optional[str] = None
    notes: str = ""


DATASETS: dict[str, DatasetConfig] = {
    "viola": DatasetConfig(
        name="viola",
        rlds_name="viola",
        fps=20,
        image_primary_key="agentview_rgb",
        image_wrist_key="eye_in_hand_rgb",
        state_key=None,  # synthesized: ee_states (homogeneous matrix) + gripper_states
        action_kind="viola_state_delta",
        language_key="natural_language_instruction",
        language_in_observation=True,
        notes="action derived from state ΔEE pose (ee_states[t+1]-ee_states[t]); "
              "raw world_vector is a normalized OSC command (range ±1), ~100× larger "
              "than physical motion, so we replace it with state-delta. Gripper from raw "
              "action.gripper_closedness_action. Original command→state-delta scale "
              "stored in info.json/action_metadata.",
    ),
    "stanford_hydra_dataset_converted_externally_to_rlds": DatasetConfig(
        name="stanford_hydra",
        rlds_name="stanford_hydra_dataset_converted_externally_to_rlds",
        fps=10,
        image_primary_key="image",
        image_wrist_key="wrist_image",
        state_key="state",
        action_kind="stanford_hydra_state_delta",
        notes="action derived from state ΔEE pose (state[0:3] pos + state[3:7] quat_wxyz); "
              "raw action is OSC command ~2× larger than physical motion. "
              "Gripper from raw action[6]. Scale in info.json/action_metadata.",
    ),
    "austin_buds_dataset_converted_externally_to_rlds": DatasetConfig(
        name="austin_buds",
        rlds_name="austin_buds_dataset_converted_externally_to_rlds",
        fps=20,
        image_primary_key="image",
        image_wrist_key="wrist_image",
        state_key="state",
        action_kind="austin_buds_state_delta",
        notes="action derived from state ΔEE pose (state[8:24] is 16D col-major homog matrix). "
              "Gripper from raw action[6]. Scale in info.json/action_metadata.",
    ),
    "austin_sailor_dataset_converted_externally_to_rlds": DatasetConfig(
        name="austin_sailor",
        rlds_name="austin_sailor_dataset_converted_externally_to_rlds",
        fps=20,
        image_primary_key="image",
        image_wrist_key="wrist_image",
        state_key="state",
        action_kind="austin_sailor_state_delta",
        notes="action derived from obs.state_ee ΔEE pose (16D col-major homog matrix). "
              "Gripper from raw action[6]. Scale in info.json/action_metadata.",
    ),
    "austin_sirius_dataset_converted_externally_to_rlds": DatasetConfig(
        name="austin_sirius",
        rlds_name="austin_sirius_dataset_converted_externally_to_rlds",
        fps=20,
        image_primary_key="image",
        image_wrist_key="wrist_image",
        state_key="state",
        action_kind="austin_sirius_state_delta",
        notes="action derived from obs.state_ee ΔEE pose. ~10 frames per episode have an "
              "all-zero state_ee matrix (logging gap); finalize emits zero delta there. "
              "Gripper from raw action[6]. Scale in info.json/action_metadata.",
    ),
    "utaustin_mutex": DatasetConfig(
        name="utaustin_mutex",
        rlds_name="utaustin_mutex",
        fps=20,
        image_primary_key="image",
        image_wrist_key="wrist_image",
        state_key="state",
        action_kind="utaustin_mutex_state_delta",
        notes="action derived from state ΔEE pose (state[8:24] is 16D col-major homog matrix). "
              "Gripper from raw action[6]. Scale in info.json/action_metadata.",
    ),
    "furniture_bench_dataset_converted_externally_to_rlds": DatasetConfig(
        name="furniture_bench",
        rlds_name="furniture_bench_dataset_converted_externally_to_rlds",
        fps=10,
        image_primary_key="image",
        image_wrist_key="wrist_image",
        state_key="state",
        action_kind="furniture_state_delta",
        notes="action derived from state ΔEE pose (state[t+1]-state[t]); raw action's "
              "quat_velocity is geometrically ambiguous so we sidestep it. Gripper = state[34] width.",
    ),
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": DatasetConfig(
        name="iamlab_cmu_pickup_insert",
        rlds_name="iamlab_cmu_pickup_insert_converted_externally_to_rlds",
        fps=20,
        image_primary_key="image",
        image_wrist_key="wrist_image",
        state_key="state",
        action_kind="abs_quat_7d",
        notes="action is absolute [xyz, quat, gripper] — converted to delta vs previous step",
    ),
    "taco_play": DatasetConfig(
        name="taco_play",
        rlds_name="taco_play",
        fps=15,
        image_primary_key="rgb_static",
        image_wrist_key="rgb_gripper",
        state_key="robot_obs",
        action_kind="taco_state_delta",
        language_key="natural_language_instruction",
        language_in_observation=True,
        notes="action derived from state ΔEE pose (robot_obs[0:3] pos m + "
              "robot_obs[3:6] euler rad → quat → diff next). Raw rel_actions_world "
              "is a CALVIN-normalized command (~40-50× pos, ~20× rot z), not "
              "physical Δm/Δrad — replaced for unit consistency with other "
              "datasets. Gripper from rel_actions_world[6] (±1). Scale stored in "
              "info.json/action_metadata.",
    ),
    "nyu_franka_play_dataset_converted_externally_to_rlds": DatasetConfig(
        name="nyu_franka_play",
        rlds_name="nyu_franka_play_dataset_converted_externally_to_rlds",
        fps=3,
        image_primary_key="image",
        image_wrist_key=None,
        image_secondary_key="image_additional_view",
        state_key="state",
        action_kind="nyu_state_delta",
        notes="no wrist; secondary 'image_additional_view' kept as image_secondary. "
              "Rotation derived from state EE pose (quat-relative), NOT the raw "
              "command Δrpy: the command is a per-component euler difference, which "
              "is not the true relative rotation and drifts a few degrees (verified "
              "via rotation round-trip). Gripper still from the raw command a[13].",
    ),
    "ucsd_kitchen_dataset_converted_externally_to_rlds": DatasetConfig(
        name="ucsd_kitchen",
        rlds_name="ucsd_kitchen_dataset_converted_externally_to_rlds",
        fps=2,
        image_primary_key="image",
        image_wrist_key=None,
        state_key="state",
        action_kind="abs_euler_7d_to_delta",
        notes="action is absolute pose [xyz(mm), euler(deg), grip, terminate]. "
              "Finalize converts to 7D delta with mm→m + deg→rad normalization; "
              "rotation uses quat-relative math to avoid ±π wrap (range is full ±180°).",
    ),
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": DatasetConfig(
        name="ucsd_pick_and_place",
        rlds_name="ucsd_pick_and_place_dataset_converted_externally_to_rlds",
        fps=3,
        image_primary_key="image",
        image_wrist_key=None,
        state_key="state",
        action_kind="ucsd_pick_state_delta",
        notes="raw action is a normalized gripper velocity command (3 lin_vel "
              "clipped to ±1 + gripper torque) that can't be integrated to the EE "
              "path. Derive the 7D delta from successive state EE poses instead "
              "(state = [pos(3), euler(3), finger(1)]); gripper = raw torque cmd. "
              "Same unit-consistency fix as taco_play / nyu_franka_play.",
    ),
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": DatasetConfig(
        name="cmu_franka_exploration",
        rlds_name="cmu_franka_exploration_dataset_converted_externally_to_rlds",
        fps=10,
        image_primary_key="image",
        image_wrist_key=None,
        image_secondary_key="highres_image",
        state_key=None,  # no proprio state field
        action_kind="abs_euler_7d",
        notes="action 8D [Δxyz, Δrpy, gripper, terminate]. Schema description "
              "says 'position/orientation' but empirically values are deltas "
              "(rotation range only ±0.06 rad, position range ≈ consec-Δ). "
              "We trust the Excel doc + data; just drop terminate.",
    ),
}


def get_config(name_or_rlds: str) -> DatasetConfig:
    if name_or_rlds in DATASETS:
        return DATASETS[name_or_rlds]
    # also accept the short `name`
    for cfg in DATASETS.values():
        if cfg.name == name_or_rlds:
            return cfg
    raise KeyError(f"Unknown dataset: {name_or_rlds}")
