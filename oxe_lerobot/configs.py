"""Per-dataset config for the 13 OXE Franka subsets used in this project.

Mirrors the role of OpenVLA's `configs.py` and Octo's `oxe_dataset_configs.py`:
each entry describes how to pull the canonical fields out of a raw RLDS step.

Action conversion kinds:
  - "delta_7d":         action is already 7D [Δxyz, Δrpy, gripper] — use as-is.
  - "delta_7d_pack":    action is a dict with separate world_vector/rotation_delta/gripper.
  - "abs_quat_7d":      action is 8D absolute [xyz, quat, gripper] — convert to 7D delta.
  - "abs_euler_7d":     action is 8D absolute [xyz, euler, gripper, terminate] — drop terminate, convert to delta.
  - "joint_vel_plus_delta": action is 15D, slice [7:13] (Δxyz, Δrpy) + [13] (gripper) → 7D.
  - "delta_furniture_8d":   action is 8D [3 lin_vel, 4 quat_vel, 1 gripper] — convert quat_vel→euler_vel.
  - "delta_xyz_only_4d":    action is 4D [3 lin_vel, 1 gripper] — pad rotation with zeros.
  - "taco_rel_world":   pick action['rel_actions_world'] (7D) directly.

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
        action_kind="delta_7d_pack",
        language_key="natural_language_instruction",
        language_in_observation=True,
        notes="action = world_vector(3)+rotation_delta(3)+gripper_closedness(1); state from ee_states+gripper_states",
    ),
    "stanford_hydra_dataset_converted_externally_to_rlds": DatasetConfig(
        name="stanford_hydra",
        rlds_name="stanford_hydra_dataset_converted_externally_to_rlds",
        fps=10,
        image_primary_key="image",
        image_wrist_key="wrist_image",
        state_key="state",
        action_kind="delta_7d",
    ),
    "austin_buds_dataset_converted_externally_to_rlds": DatasetConfig(
        name="austin_buds",
        rlds_name="austin_buds_dataset_converted_externally_to_rlds",
        fps=20,
        image_primary_key="image",
        image_wrist_key="wrist_image",
        state_key="state",
        action_kind="delta_7d",
    ),
    "austin_sailor_dataset_converted_externally_to_rlds": DatasetConfig(
        name="austin_sailor",
        rlds_name="austin_sailor_dataset_converted_externally_to_rlds",
        fps=20,
        image_primary_key="image",
        image_wrist_key="wrist_image",
        state_key="state",
        action_kind="delta_7d",
    ),
    "austin_sirius_dataset_converted_externally_to_rlds": DatasetConfig(
        name="austin_sirius",
        rlds_name="austin_sirius_dataset_converted_externally_to_rlds",
        fps=20,
        image_primary_key="image",
        image_wrist_key="wrist_image",
        state_key="state",
        action_kind="delta_7d",
    ),
    "utaustin_mutex": DatasetConfig(
        name="utaustin_mutex",
        rlds_name="utaustin_mutex",
        fps=20,
        image_primary_key="image",
        image_wrist_key="wrist_image",
        state_key="state",
        action_kind="delta_7d",
    ),
    "furniture_bench_dataset_converted_externally_to_rlds": DatasetConfig(
        name="furniture_bench",
        rlds_name="furniture_bench_dataset_converted_externally_to_rlds",
        fps=10,
        image_primary_key="image",
        image_wrist_key="wrist_image",
        state_key="state",
        action_kind="delta_furniture_8d",
        notes="action = 3 lin_vel + 4 quat_vel + 1 gripper → convert quat→euler for unified 7D",
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
        action_kind="taco_rel_world",
        language_key="natural_language_instruction",
        language_in_observation=True,
    ),
    "nyu_franka_play_dataset_converted_externally_to_rlds": DatasetConfig(
        name="nyu_franka_play",
        rlds_name="nyu_franka_play_dataset_converted_externally_to_rlds",
        fps=3,
        image_primary_key="image",
        image_wrist_key=None,
        image_secondary_key="image_additional_view",
        state_key="state",
        action_kind="joint_vel_plus_delta",
        notes="no wrist; secondary 'image_additional_view' kept as image_secondary",
    ),
    "ucsd_kitchen_dataset_converted_externally_to_rlds": DatasetConfig(
        name="ucsd_kitchen",
        rlds_name="ucsd_kitchen_dataset_converted_externally_to_rlds",
        fps=2,
        image_primary_key="image",
        image_wrist_key=None,
        state_key="state",
        action_kind="abs_euler_7d",
    ),
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": DatasetConfig(
        name="ucsd_pick_and_place",
        rlds_name="ucsd_pick_and_place_dataset_converted_externally_to_rlds",
        fps=3,
        image_primary_key="image",
        image_wrist_key=None,
        state_key="state",
        action_kind="delta_xyz_only_4d",
        notes="action is only 4D (3 lin_vel + gripper); rotation channels zero-padded",
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
        notes="action 8D [3 pos, 3 ori, gripper, terminate] — drop terminate, treat as delta per Excel doc",
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
