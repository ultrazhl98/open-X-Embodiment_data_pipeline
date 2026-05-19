"""OXE → LeRobot v2 conversion utilities.

Layered like Octo / OpenVLA:
- configs.py:    per-dataset metadata (fps, image keys, action format flag).
- transforms.py: per-dataset `standardize_fn` that maps a raw RLDS step into
                 a canonical schema:
                     {
                         "image_primary":   uint8 (H, W, 3),
                         "image_wrist":     uint8 (H, W, 3)  | None,
                         "state":           float32 (Dproprio,),
                         "action":          float32 (7,)  -- 7D EEF delta,
                         "language_instruction": str,
                     }
- lerobot_writer.py: builds a LeRobot v2 dataset on disk from canonical episodes.
- convert.py: CLI that wires it all together over the local raw shards.
"""
