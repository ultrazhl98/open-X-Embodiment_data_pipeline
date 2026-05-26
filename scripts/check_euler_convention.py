"""Quantify the intrinsic-vs-extrinsic euler-decode gap on every euler-source OXE dataset.

Background
----------
For OXE datasets both pipelines are state/command-derived, world-frame relative,
axis-angle output.  The ONLY real divergence is how an euler triple is turned into
a rotation matrix:

  - ours  : EXTRINSIC xyz  =  Rz·Ry·Rx   (scipy 'xyz' / ROS rpy — matches the
            convention OXE/CALVIN source euler was authored in)
  - ABot  : INTRINSIC XYZ  =  Rx·Ry·Rz   (pytorch3d `euler_angles_to_matrix(e,"XYZ")`,
            used in concat.py / data_config.py / compute_delta_state_stats.py)

This script decodes the SAME euler numbers both ways and reports:

  abs-gap   : per-frame geodesic angle between the two ABSOLUTE orientations
              (how differently the conventions read the pose — large)
  delta-gap : geodesic angle between the two world-frame relative DELTAS
              (the number that actually reaches the model — small, because a
              near-constant abs bias cancels in the relative computation)

Oracle: only `stanford_hydra` carries an INDEPENDENT rotation reference in its raw
data (a quaternion alongside the euler), so only there can we say which convention
is *correct* rather than merely *different*.  For the others there is no quat/matrix
oracle — the gap below is "ABot vs ours", not "vs physical truth".

Usage:  python scripts/check_euler_convention.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oxe_lerobot.rlds_reader import iter_episodes      # noqa: E402
from oxe_lerobot.transforms import _ensure_1d          # noqa: E402

# --------------------------- rotation helpers --------------------------- #
def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def R_ext(e): return Rz(e[2]) @ Ry(e[1]) @ Rx(e[0])    # ours / scipy 'xyz' / ROS rpy
def R_int(e): return Rx(e[0]) @ Ry(e[1]) @ Rz(e[2])    # ABot pytorch3d "XYZ"

def quat_xyzw_to_R(q):
    x, y, z, w = np.asarray(q, float) / (np.linalg.norm(q) + 1e-12)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def geo(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(min(1.0, max(-1.0, c))))

def world_delta(Rs):  # list of abs R -> per-step world-frame relative R (len-1)
    return [Rs[t+1] @ Rs[t].T for t in range(len(Rs) - 1)]


# --------- per-dataset euler source (field, slice, abs|delta, optional quat oracle) --------- #
# loc=("obs",key) reads observation[key]; loc=("action",None) reads the action array.
# Verified against the project's own standardizers (oxe_lerobot/transforms.py).
SPECS = {
    "taco_play": dict(rlds="taco_play",
                      loc=("obs", "robot_obs"), sl=slice(3, 6), kind="abs"),
    "nyu_franka_play": dict(rlds="nyu_franka_play_dataset_converted_externally_to_rlds",
                            loc=("obs", "state"), sl=slice(10, 13), kind="abs"),
    "ucsd_pick_and_place": dict(rlds="ucsd_pick_and_place_dataset_converted_externally_to_rlds",
                                loc=("obs", "state"), sl=slice(3, 6), kind="abs"),
    "ucsd_kitchen": dict(rlds="ucsd_kitchen_dataset_converted_externally_to_rlds",
                         loc=("action", None), sl=slice(3, 6), kind="abs"),
    "cmu_franka_exploration": dict(rlds="cmu_franka_exploration_dataset_converted_externally_to_rlds",
                                   loc=("action", None), sl=slice(3, 6), kind="delta"),
    # the only dataset with an independent rotation oracle (raw state quat, xyzw)
    "stanford_hydra": dict(rlds="stanford_hydra_dataset_converted_externally_to_rlds",
                           loc=("obs", "state"), sl=slice(7, 10), kind="abs",
                           quat=lambda s: _ensure_1d(s["observation"]["state"])[3:7]),  # xyzw
}


def _read_euler(step, spec):
    where, key = spec["loc"]
    arr = _ensure_1d(step["observation"][key]) if where == "obs" else _ensure_1d(step["action"])
    return arr[spec["sl"]].astype(np.float64)


def run(short, max_eps=6):
    spec = SPECS[short]
    raw = f"data/raw/{spec['rlds']}/0.1.0"
    abs_gaps, dlt_gaps, ext_vs_gt, int_vs_gt = [], [], [], []
    n_steps = 0
    try:
        for ep in iter_episodes(raw, max_episodes=max_eps):
            steps = ep["steps"]
            if len(steps) < 2:
                continue
            E = [_read_euler(s, spec) for s in steps]
            Rext = [R_ext(e) for e in E]
            Rint = [R_int(e) for e in E]
            n_steps += len(steps)

            # per-frame absolute gap (intrinsic vs extrinsic reading of the same euler)
            abs_gaps += [geo(Rext[t], Rint[t]) for t in range(len(E))]

            if spec["kind"] == "abs":
                dext, dint = world_delta(Rext), world_delta(Rint)
            else:  # "delta": the euler IS the per-step rotation command
                dext, dint = Rext, Rint
            dlt_gaps += [geo(dext[t], dint[t]) for t in range(len(dext))]

            if "quat" in spec:  # oracle present: which convention is actually correct?
                Rgt = [quat_xyzw_to_R(spec["quat"](s)) for s in steps]
                dgt = world_delta(Rgt)
                ext_vs_gt += [geo(dext[t], dgt[t]) for t in range(len(dgt))]
                int_vs_gt += [geo(dint[t], dgt[t]) for t in range(len(dgt))]
    except Exception as ex:
        return f"{short:24s} [ERROR] {ex}"

    a, d = np.array(abs_gaps), np.array(dlt_gaps)
    line = (f"{short:24s} eps≈{max_eps:<2d} steps={n_steps:<5d} "
            f"abs-gap mean={a.mean():6.2f}° max={a.max():6.2f}°  | "
            f"delta-gap mean={d.mean():5.2f}° med={np.median(d):5.2f}° max={d.max():5.2f}°")
    if ext_vs_gt:
        e2, i2 = np.array(ext_vs_gt), np.array(int_vs_gt)
        line += (f"\n{'':24s} ORACLE(raw quat): ours/extrinsic vs truth={e2.mean():.3f}°  "
                 f"ABot/intrinsic vs truth={i2.mean():.2f}°  -> {'EXTRINSIC correct' if e2.mean()<i2.mean() else 'INTRINSIC correct'}")
    return line


if __name__ == "__main__":
    print("intrinsic(ABot) vs extrinsic(ours) euler-decode gap   [abs-gap=per-frame pose, delta-gap=what hits the model]")
    print("-" * 118)
    for short in SPECS:
        print(run(short))
    print("-" * 118)
    print("note: only stanford_hydra has an independent rotation oracle; elsewhere the gap is ABot-vs-ours, not vs-truth.")
