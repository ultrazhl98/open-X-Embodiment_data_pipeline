"""Batch rotation-convention / correctness check for all OXE datasets.

For each dataset it runs the project's actual pipeline (standardize_step +
finalize_actions_to_delta) to get the produced 7D delta, then cross-checks the
rotation part against an INDEPENDENT ground truth derived from the raw RLDS:

  - matrix  : pose comes from a homogeneous matrix -> R is convention-free.
              (validates code: column-major / transpose / source-field bugs)
  - euler_x : an independent euler field exists -> GT = R under extrinsic-xyz
              (Rz.Ry.Rx, our verified convention). Cross-checks quat handling.
  - command : a world-frame rotation command exists -> compare DIRECTION
              (per-axis corr + dir-cosine). Discriminates euler conventions
              (ours Rz.Ry.Rx vs ABot pytorch3d "XYZ" Rx.Ry.Rz).
  - quat    : pose comes from a quaternion -> auto-detect (w,x,y,z) vs
              (x,y,z,w) by inter-frame smoothness, flag if the pipeline's
              assumed order looks wrong.
  - self    : no independent reference available (euler-only / command-only).

Usage:  python scripts/verify_rotation.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oxe_lerobot.configs import get_config, DATASETS
from oxe_lerobot.rlds_reader import iter_episodes
from oxe_lerobot.transforms import _ensure_1d, standardize_step, finalize_actions_to_delta


# --------------------------- rotation helpers --------------------------- #
def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
def R_ours(e): return Rz(e[2]) @ Ry(e[1]) @ Rx(e[0])   # extrinsic xyz / ROS rpy
def R_abot(e): return Rx(e[0]) @ Ry(e[1]) @ Rz(e[2])   # pytorch3d "XYZ"

def quat_to_R(q):  # wxyz
    q = np.asarray(q, float); q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def homog16_to_R(m16):  # column-major 4x4 -> R
    m = np.asarray(m16, float).reshape(4, 4).T
    return m[:3, :3]

def rotvec_to_R(rv):
    th = np.linalg.norm(rv)
    if th < 1e-9: return np.eye(3)
    k = rv / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th)*K + (1-np.cos(th))*K@K

def geo(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(min(1, max(-1, c))))


# --------------------- per-dataset ground-truth specs --------------------- #
# kind: how to get the absolute orientation R(t) for each raw step, or a command.
GT = {
    # matrix-pose datasets (convention-free)
    "viola":         ("matrix", lambda s: homog16_to_R(_ensure_1d(s["observation"]["ee_states"]))),
    "austin_buds":   ("matrix", lambda s: homog16_to_R(_ensure_1d(s["observation"]["state"])[8:24])),
    "utaustin_mutex":("matrix", lambda s: homog16_to_R(_ensure_1d(s["observation"]["state"])[8:24])),
    "austin_sailor": ("matrix", lambda s: homog16_to_R(_ensure_1d(s["observation"]["state_ee"]))),
    "austin_sirius": ("matrix", lambda s: homog16_to_R(_ensure_1d(s["observation"]["state_ee"]))),
    # euler cross-check (independent of the quat the pipeline uses)
    "stanford_hydra":("euler_x", lambda s: _ensure_1d(s["observation"]["state"])[7:10]),
    # command-direction (discriminates euler convention)
    "taco_play":        ("command", lambda s: _ensure_1d(s["observation"]["robot_obs"])[3:6],
                                     lambda s: _ensure_1d(s["action"]["rel_actions_world"])[3:6]),
    "nyu_franka_play":  ("command", lambda s: _ensure_1d(s["observation"]["state"])[10:13],
                                     lambda s: _ensure_1d(s["action"])[10:13]),
    # quat-order auto-detect
    "furniture_bench":  ("quat", lambda s: _ensure_1d(s["observation"]["state"])[3:7]),
    "iamlab_cmu_pickup_insert": ("quat", lambda s: _ensure_1d(s["action"])[3:7]),
    # self-only (no independent rotation reference)
    "ucsd_pick_and_place":   ("self", None),
    "ucsd_kitchen":          ("self", None),
    "cmu_franka_exploration":("self", None),
}


def pipeline_rotvec(cfg, steps):
    canon = [standardize_step(cfg, s, None) for s in steps]
    canon = finalize_actions_to_delta(canon, cfg)
    A = np.array([c["action"] for c in canon], dtype=float)
    return A[:, 3:6]


def run_dataset(short):
    cfg = get_config(short)
    spec = GT.get(short)
    raw = f"data/raw/{cfg.rlds_name}/0.1.0"
    kind = spec[0] if spec else "self"

    eulers, mats, cmds, rvs = [], [], [], []
    try:
        for ep in iter_episodes(raw, max_episodes=6):
            steps = ep["steps"]
            rv = pipeline_rotvec(cfg, steps)
            n = len(steps)
            if kind == "matrix":
                R = [spec[1](s) for s in steps]
                for t in range(n-1):
                    mats.append((R[t], R[t+1], rv[t]))
            elif kind == "euler_x":
                e = [spec[1](s) for s in steps]
                for t in range(n-1):
                    eulers.append((e[t], e[t+1], rv[t]))
            elif kind == "command":
                e = [spec[1](s) for s in steps]; c = [spec[2](s) for s in steps]
                for t in range(n-1):
                    cmds.append((e[t], e[t+1], c[t]))
            elif kind == "quat":
                q = [spec[1](s) for s in steps]
                for t in range(n-1):
                    rvs.append((q[t], q[t+1], rv[t]))
    except Exception as ex:
        return f"{short:28s} [ERROR] {ex}"

    if kind == "matrix":
        # skip degenerate frames (all-zero / non-orthonormal homog matrices) — the
        # pipeline correctly emits zero delta for these, so they aren't a mismatch.
        def ok_rot(R):
            return abs(np.linalg.det(R) - 1.0) < 0.1
        d = np.array([geo(rotvec_to_R(rv), Rn @ Rp.T)
                      for Rp, Rn, rv in mats if ok_rot(Rp) and ok_rot(Rn)])
        skipped = len(mats) - len(d)
        ok = "PASS" if d.mean() < 1.0 else "FAIL"
        return (f"{short:28s} matrix   your-vs-GT: mean={d.mean():6.3f}° max={d.max():6.3f}° "
                f"(skipped {skipped} degenerate)  [{ok}]")

    if kind == "euler_x":
        d = np.array([geo(rotvec_to_R(rv), R_ours(en) @ R_ours(ep).T) for ep, en, rv in eulers])
        ok = "PASS" if d.mean() < 1.0 else "FAIL"
        return f"{short:28s} euler_x  your-vs-GT: mean={d.mean():6.3f}° max={d.max():6.3f}°   [{ok}]"

    if kind == "command":
        out = []
        for nm, Rf in [("ours", R_ours), ("ABot", R_abot)]:
            rv = np.array([_rv(Rf(en) @ Rf(ep).T) for ep, en, c in cmds])
            C = np.array([c for _, _, c in cmds])
            corz = np.corrcoef(rv[:, 2], C[:, 2])[0, 1]
            a = rv/(np.linalg.norm(rv, axis=1, keepdims=True)+1e-9)
            b = C/(np.linalg.norm(C, axis=1, keepdims=True)+1e-9)
            m = np.linalg.norm(rv, axis=1) > 1e-4
            out.append((nm, corz, np.sum(a*b, 1)[m].mean()))
        win = "ours" if out[0][2] > out[1][2] else "ABot"
        return (f"{short:28s} command  yaw-corr ours={out[0][1]:+.2f}/ABot={out[1][1]:+.2f}  "
                f"dir-cos ours={out[0][2]:+.2f}/ABot={out[1][2]:+.2f}   [WIN: {win}]")

    if kind == "quat":
        # NOTE: inter-frame angle is order-invariant (wrong order ~ conjugation, which
        # preserves rotation angle), so it can't tell order. With no independent local
        # reference (no euler/matrix), report the component profile so the scalar (w)
        # part can be identified: |w| is largest for small rotations, ~0 for 180° flips.
        Q = np.array([qp for qp, _, _ in rvs], dtype=float)
        absmean = np.abs(Q).mean(0)
        # pipeline assumes the FIRST index is the scalar (wxyz). Heuristic guess:
        scalar_idx = int(np.argmax(absmean))
        guess = "wxyz" if scalar_idx == 0 else ("xyzw" if scalar_idx == 3 else f"idx{scalar_idx}")
        note = "matches pipeline(wxyz)" if guess == "wxyz" else f"scalar looks like idx{scalar_idx} -> {guess}? verify"
        return (f"{short:28s} quat     comp |mean|=[{absmean[0]:.2f},{absmean[1]:.2f},"
                f"{absmean[2]:.2f},{absmean[3]:.2f}]  scalar≈idx{scalar_idx} -> {note}")

    return f"{short:28s} self     (no independent rotation reference; not checked here)"


def _rv(R):
    c = (np.trace(R)-1)/2; c = min(1, max(-1, c)); th = np.arccos(c)
    if th < 1e-7: return np.zeros(3)
    return th/(2*np.sin(th))*np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])


if __name__ == "__main__":
    print(f"{'dataset':28s} {'check':8s} result")
    print("-" * 92)
    for short in GT:
        print(run_dataset(short))
