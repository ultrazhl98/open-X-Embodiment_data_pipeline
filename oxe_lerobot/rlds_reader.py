"""Read locally downloaded OXE RLDS shards.

We deliberately bypass `tfds.builder_from_directory(...).as_dataset()` because
it consults `dataset_info.json` and demands every shard listed there be on disk
(we only download one shard per dataset). Instead we:
  1. Parse `features.json` ourselves into a `tfds.features.FeaturesDict`.
  2. Glob the locally-present `<name>-train.tfrecord-*` files.
  3. Decode each `tf.train.Example` via `features.deserialize_example_np`.

This is the same code path TFDS uses internally for RLDS datasets.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Iterator

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def _to_numpy(x):
    if isinstance(x, dict):
        return {k: _to_numpy(v) for k, v in x.items()}
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return x


def _load_features(version_dir: str) -> tfds.features.FeaturesDict:
    fpath = os.path.join(version_dir, "features.json")
    with open(fpath) as f:
        spec = json.load(f)
    return tfds.features.FeaturesDict.from_json(spec)


def iter_episodes(version_dir: str, max_episodes: int | None = None) -> Iterator[dict]:
    """version_dir: e.g. data/raw/viola/0.1.0  (must contain features.json + at
    least one tfrecord shard)."""
    version_dir = version_dir.rstrip("/")
    features = _load_features(version_dir)

    shards = sorted(glob.glob(os.path.join(version_dir, "*-train.tfrecord-*")))
    if not shards:
        raise FileNotFoundError(f"No train shards under {version_dir}")

    ds = tf.data.TFRecordDataset(shards, num_parallel_reads=1)

    count = 0
    for raw_record in ds:
        # `deserialize_example` (TFDS >= 4.9) returns a nested dict of tf.Tensors,
        # with the inner `steps` materialized as a `tf.data.Dataset`.
        example = features.deserialize_example(raw_record)
        steps_ds = example["steps"]
        steps = []
        for step in steps_ds:
            steps.append(_to_numpy(step))
        meta = {}
        if "episode_metadata" in example:
            meta = _to_numpy(example["episode_metadata"])
        yield {"steps": steps, "episode_metadata": meta}
        count += 1
        if max_episodes is not None and count >= max_episodes:
            break
