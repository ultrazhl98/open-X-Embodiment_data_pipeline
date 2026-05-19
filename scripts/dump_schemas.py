"""Compact dump of step.observation / step.action / step.* schemas per dataset.

Reads each downloaded features.json and prints a flat per-dataset summary so
we can write per-dataset standardize_fn's accurately.
"""
import json
import os
import glob


def _resolve_shape(t):
    if isinstance(t, dict):
        dims = t.get("shape", {}).get("dimensions", [])
        return tuple(int(d) for d in dims) if dims else ()
    return ()


def _flatten(node, prefix=""):
    """Yield (path, kind, shape, dtype, desc) tuples for leaf features."""
    if not isinstance(node, dict):
        return
    if "featuresDict" in node:
        feats = node["featuresDict"].get("features", {})
        for k, v in feats.items():
            yield from _flatten(v, f"{prefix}/{k}" if prefix else k)
        return
    if "sequence" in node:
        feat = node["sequence"].get("feature", {})
        yield from _flatten(feat, prefix)
        return
    desc = node.get("description", "")
    if "image" in node:
        shape = _resolve_shape(node["image"])
        dtype = node["image"].get("dtype", "uint8")
        yield (prefix, "image", shape, dtype, desc)
        return
    if "tensor" in node:
        shape = _resolve_shape(node["tensor"])
        dtype = node["tensor"].get("dtype", "")
        yield (prefix, "tensor", shape, dtype, desc)
        return
    if "text" in node:
        yield (prefix, "text", (), "string", desc)
        return


def dump_dataset(features_json_path: str):
    with open(features_json_path) as f:
        feats = json.load(f)
    ds = os.path.basename(os.path.dirname(os.path.dirname(features_json_path)))
    print(f"\n========== {ds} ==========")
    for path, kind, shape, dtype, desc in _flatten(feats):
        d = f"  # {desc}" if desc else ""
        print(f"  {path:55s} {kind:6s} {str(shape):20s} {dtype}{d}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in sorted(glob.glob(os.path.join(here, "data/raw/*/0.1.0/features.json"))):
        dump_dataset(p)
