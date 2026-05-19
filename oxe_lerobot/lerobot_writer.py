"""Minimal writer for the LeRobot v2.0 on-disk format.

Layout produced (matches huggingface/lerobot reference):
  <root>/
    meta/
      info.json
      episodes.jsonl
      tasks.jsonl
      stats.json
    data/
      chunk-000/
        episode_000000.parquet
        ...
    videos/
      chunk-000/
        observation.images.primary/
          episode_000000.mp4
        observation.images.wrist/
          episode_000000.mp4   (only if dataset has wrist)
        ...

Each parquet row is one frame, with columns:
  action                       list<float32, 7>
  observation.state            list<float32, Dproprio>
  observation.images.primary   string (relative video path; per LeRobot v2 video features)
  observation.images.wrist     string
  timestamp                    float32
  frame_index                  int64
  episode_index                int64
  index                        int64
  task_index                   int64

`stats.json` holds per-feature mean/std/min/max/q01/q99, consistent with what
Octo / OpenVLA's `compute_normalization_stats` produces.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CODEBASE_VERSION = "v2.0"


@dataclass
class _RunningStats:
    n: int = 0
    sum: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sumsq: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mn: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mx: np.ndarray = field(default_factory=lambda: np.zeros(0))
    samples: list = field(default_factory=list)  # for q01/q99 over a capped sample

    def update(self, v: np.ndarray, sample_cap: int = 20000):
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        if self.n == 0:
            self.sum = np.zeros_like(v)
            self.sumsq = np.zeros_like(v)
            self.mn = v.copy()
            self.mx = v.copy()
        self.sum += v
        self.sumsq += v * v
        self.mn = np.minimum(self.mn, v)
        self.mx = np.maximum(self.mx, v)
        self.n += 1
        if len(self.samples) < sample_cap:
            self.samples.append(v.astype(np.float32))

    def finalize(self) -> dict:
        if self.n == 0:
            return {}
        mean = self.sum / self.n
        var = self.sumsq / self.n - mean ** 2
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        arr = np.stack(self.samples, axis=0) if self.samples else None
        q01 = np.quantile(arr, 0.01, axis=0) if arr is not None else mean
        q99 = np.quantile(arr, 0.99, axis=0) if arr is not None else mean
        return {
            "mean": mean.astype(np.float32).tolist(),
            "std":  std.astype(np.float32).tolist(),
            "min":  self.mn.astype(np.float32).tolist(),
            "max":  self.mx.astype(np.float32).tolist(),
            "q01":  q01.astype(np.float32).tolist(),
            "q99":  q99.astype(np.float32).tolist(),
            "count": int(self.n),
        }


class LeRobotV2Writer:
    """Streams episodes to disk. Use as a context manager."""

    def __init__(
        self,
        root: str,
        repo_id: str,
        fps: int,
        robot_type: str = "franka_panda",
        image_keys: tuple[str, ...] = ("primary",),
        state_dim: int = 7,
        action_dim: int = 7,
        chunk_size: int = 1000,
        video_codec: str = "libx264",
        video_quality: int = 23,
    ):
        self.root = Path(root)
        self.repo_id = repo_id
        self.fps = int(fps)
        self.robot_type = robot_type
        self.image_keys = tuple(image_keys)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.video_codec = video_codec
        self.video_quality = video_quality

        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "meta").mkdir(exist_ok=True)
        (self.root / "data").mkdir(exist_ok=True)
        (self.root / "videos").mkdir(exist_ok=True)

        self._episodes_meta: list[dict] = []
        self._tasks: dict[str, int] = {}
        self._stats = {
            "action":             _RunningStats(),
            "observation.state":  _RunningStats(),
        }
        self._global_index = 0
        self._image_shapes: dict[str, tuple[int, int, int]] = {}

    # -- task table ---------------------------------------------------------
    def _task_index(self, task: str) -> int:
        if task not in self._tasks:
            self._tasks[task] = len(self._tasks)
        return self._tasks[task]

    # -- per-episode ingest -------------------------------------------------
    def write_episode(self, ep_index: int, frames: list[dict]):
        """frames: list of dicts with keys:
              action: (action_dim,) float32
              state:  (state_dim,)  float32
              images: {image_key: HxWx3 uint8}
              language_instruction: str
        """
        if len(frames) == 0:
            return
        chunk_dir_idx = ep_index // self.chunk_size
        chunk_name = f"chunk-{chunk_dir_idx:03d}"

        # 1) videos -- one mp4 per image_key
        video_paths_per_key: dict[str, str] = {}
        for k in self.image_keys:
            vid_dir = self.root / "videos" / chunk_name / f"observation.images.{k}"
            vid_dir.mkdir(parents=True, exist_ok=True)
            rel_path = f"videos/{chunk_name}/observation.images.{k}/episode_{ep_index:06d}.mp4"
            out_path = self.root / rel_path
            imgs = [f["images"][k] for f in frames]
            if k not in self._image_shapes:
                self._image_shapes[k] = imgs[0].shape  # (H, W, 3)
            with imageio.get_writer(
                str(out_path),
                fps=self.fps,
                codec=self.video_codec,
                quality=None,
                macro_block_size=1,
                ffmpeg_params=["-crf", str(self.video_quality), "-pix_fmt", "yuv420p"],
            ) as w:
                for im in imgs:
                    w.append_data(im)
            video_paths_per_key[k] = rel_path

        # 2) parquet rows
        n = len(frames)
        actions = np.stack([f["action"] for f in frames]).astype(np.float32)
        states = np.stack([f["state"]  for f in frames]).astype(np.float32)
        # running stats
        for a in actions:
            self._stats["action"].update(a)
        for s in states:
            self._stats["observation.state"].update(s)

        task_str = frames[0]["language_instruction"] or ""
        task_idx = self._task_index(task_str)

        timestamps = (np.arange(n) / self.fps).astype(np.float32)
        frame_indices = np.arange(n, dtype=np.int64)
        episode_indices = np.full(n, ep_index, dtype=np.int64)
        global_indices = (np.arange(n) + self._global_index).astype(np.int64)
        task_indices = np.full(n, task_idx, dtype=np.int64)
        self._global_index += n

        columns = {
            "action":                                    [a.tolist() for a in actions],
            "observation.state":                         [s.tolist() for s in states],
            "timestamp":                                 timestamps,
            "frame_index":                               frame_indices,
            "episode_index":                             episode_indices,
            "index":                                     global_indices,
            "task_index":                                task_indices,
        }
        # Per LeRobot v2: video features are referenced by a single path per row;
        # the actual frame is decoded at load time. We store the relative mp4 path.
        for k in self.image_keys:
            columns[f"observation.images.{k}"] = [video_paths_per_key[k]] * n

        table = pa.table(columns)
        data_dir = self.root / "data" / chunk_name
        data_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, data_dir / f"episode_{ep_index:06d}.parquet")

        self._episodes_meta.append({
            "episode_index": ep_index,
            "tasks": [task_str],
            "length": n,
        })

    # -- finalize -----------------------------------------------------------
    def close(self):
        meta = self.root / "meta"

        # episodes.jsonl
        with open(meta / "episodes.jsonl", "w") as f:
            for e in self._episodes_meta:
                f.write(json.dumps(e) + "\n")

        # tasks.jsonl
        with open(meta / "tasks.jsonl", "w") as f:
            for task, idx in sorted(self._tasks.items(), key=lambda kv: kv[1]):
                f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

        # stats.json
        stats = {k: v.finalize() for k, v in self._stats.items()}
        with open(meta / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # info.json
        total_frames = sum(e["length"] for e in self._episodes_meta)
        total_episodes = len(self._episodes_meta)
        features = {
            "action": {
                "dtype": "float32",
                "shape": [self.action_dim],
                "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [self.state_dim],
                "names": None,
            },
            "timestamp":      {"dtype": "float32", "shape": [1], "names": None},
            "frame_index":    {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index":  {"dtype": "int64",   "shape": [1], "names": None},
            "index":          {"dtype": "int64",   "shape": [1], "names": None},
            "task_index":     {"dtype": "int64",   "shape": [1], "names": None},
        }
        for k in self.image_keys:
            H, W, C = self._image_shapes.get(k, (0, 0, 3))
            features[f"observation.images.{k}"] = {
                "dtype": "video",
                "shape": [H, W, C],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.fps": float(self.fps),
                    "video.height": H,
                    "video.width": W,
                    "video.channels": C,
                    "video.codec": self.video_codec,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            }
        info = {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(self._tasks),
            "total_videos": total_episodes * len(self.image_keys),
            "total_chunks": (total_episodes + self.chunk_size - 1) // self.chunk_size,
            "chunks_size": self.chunk_size,
            "fps": self.fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
            "repo_id": self.repo_id,
        }
        with open(meta / "info.json", "w") as f:
            json.dump(info, f, indent=2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
