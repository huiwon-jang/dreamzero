#!/usr/bin/env python
"""huiwon 2026-09-16: generate meta/relative_stats_dreamzero.json for the OpenArm subsets.

DreamZero normalizes relative action keys with GLOBAL per-dim q01/q99 of the relative deltas
(lerobot.py:_calculate_relative_stats_for_key): for every anchor i of every episode and every horizon step
h in 0..23, delta = action[i+h] - state[i]; all (i, h) are pooled and q01/q99/min/max/mean/std taken per dim.
Upstream computes it per dataset and WRITES it into the dataset root on first use. We want ONE pooled table
(WAM convention: pooled robot stats, identical file in every subset, human subsets included so the loader
never computes/writes on its own), so this script:

  * streams the 6 robot subsets' parquets (state/action 28-dim, modality.json slices) with EXACTLY the
    upstream semantics (anchor state at i, 24-step chunk, i <= len-24, pooled over horizon),
  * skips groups that a subset marks invalid in meta/wam_action_groups.json (banana_v21_openarm28: only
    right_arm/right_hand are real, its zero-filled left_arm would otherwise pollute the left-arm quantiles),
  * writes the same relative_stats_dreamzero.json (keys: left_arm_joints, right_arm_joints) into all 11
    subsets, plus relative_stats_dreamzero.provenance.json next to it.

The existing meta/relative_stats.json (WAM per-timestep 24x7 tables) is a different object (per-horizon
quantiles cannot be combined into a global quantile), hence a fresh streaming pass. CPU only, ~300 MB RAM.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROBOT = [
    "robot/openarm_ego_jungwook",
    "robot/openarm_teleop_v3/bottle",
    "robot/openarm_teleop_v3/cup",
    "robot/openarm_teleop_v3/doll",
    "robot/openarm_teleop_v3/snack",
    "robot/banana_v21_openarm28",
]
HUMAN = [
    "human_as_openarm28/rlwrld_human_lerobot",
    "human_as_openarm28/openarm_validation_v2_junhyeong/close_air_fryer",
    "human_as_openarm28/openarm_validation_v2_junhyeong/left_hand_box_white_container",
    "human_as_openarm28/openarm_validation_v2_junhyeong/open_air_fryer",
    "human_as_openarm28/anyh2r",
]
KEYS = ["left_arm_joints", "right_arm_joints"]
HORIZON = 24  # action delta_indices 0..23 in the openarm data config
MARKER = "huiwon_openarm_relative_stats_dreamzero_v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/huiwon/data/openarm_wam_v1")
    ap.add_argument("--dry-run", action="store_true", help="compute + print, do not write")
    ap.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = ap.parse_args()

    t0 = time.time()
    pooled = {k: [] for k in KEYS}
    prov = {"marker": MARKER, "horizon": HORIZON, "keys": KEYS, "subsets": {}, "semantics":
            "delta[i,h,:] = action[i+h, slice] - state[i, slice] for i in [0, len-24], h in [0,24); pooled over i,h"}
    for sub in ROBOT:
        d = os.path.join(args.root, sub)
        mod = json.load(open(os.path.join(d, "meta/modality.json")))
        valid = None
        agp = os.path.join(d, "meta/wam_action_groups.json")
        if os.path.exists(agp):
            valid = set(json.load(open(agp))["valid_groups"])
        files = sorted(glob.glob(os.path.join(d, "data/chunk-*/episode_*.parquet")))
        assert files, f"no parquets under {d}"
        n_ep, n_fr, n_anchor = 0, 0, 0
        used_keys = [k for k in KEYS if valid is None or k in valid]
        for f in files:
            df = pd.read_parquet(f, columns=["observation.state", "action"])
            S = np.stack(df["observation.state"].values).astype(np.float32)
            A = np.stack(df["action"].values).astype(np.float32)
            T = len(df)
            n_ep += 1
            n_fr += T
            n_anchor_ep = T - HORIZON + 1  # i in [0, T-24]
            if n_anchor_ep <= 0:
                continue
            n_anchor += n_anchor_ep
            for k in used_keys:
                s0, s1 = mod["state"][k]["start"], mod["state"][k]["end"]
                a0, a1 = mod["action"][k]["start"], mod["action"][k]["end"]
                assert (s0, s1) == (a0, a1), (sub, k, s0, s1, a0, a1)
                ref = S[: n_anchor_ep, s0:s1]  # (N, D) anchor state
                # (N, 24, D) chunk of actions starting at each anchor
                win = np.lib.stride_tricks.sliding_window_view(A[:, a0:a1], (HORIZON, a1 - a0))[:, 0]  # (N, 24, D)
                delta = (win - ref[:, None, :]).reshape(-1, a1 - a0)
                pooled[k].append(delta)
        prov["subsets"][sub] = {"episodes": n_ep, "frames": n_fr, "anchors": n_anchor, "keys_used": used_keys,
                                "keys_skipped": [k for k in KEYS if k not in used_keys]}
        print(f"[{sub}] ep={n_ep} frames={n_fr} anchors={n_anchor} keys={used_keys}", flush=True)

    stats = {}
    for k in KEYS:
        X = np.concatenate(pooled[k], axis=0)
        print(f"[{k}] pooled samples={X.shape[0]} dims={X.shape[1]}", flush=True)
        stats[k] = {
            "max": X.max(axis=0).tolist(),
            "min": X.min(axis=0).tolist(),
            "mean": X.mean(axis=0).tolist(),
            "std": X.std(axis=0).tolist(),
            "q01": np.quantile(X, 0.01, axis=0).tolist(),
            "q99": np.quantile(X, 0.99, axis=0).tolist(),
        }
        prov["subsets"].setdefault("_pooled_samples", {})[k] = int(X.shape[0])
        del X
    payload = json.dumps(stats, indent=4)
    prov["relative_stats_dreamzero_md5"] = hashlib.md5(payload.encode()).hexdigest()
    prov["seconds"] = round(time.time() - t0, 1)
    print(json.dumps({k: {"q01": [round(x, 4) for x in v["q01"]], "q99": [round(x, 4) for x in v["q99"]]}
                      for k, v in stats.items()}, indent=1))
    if args.dry_run:
        print("[dry-run] not writing")
        return
    for sub in ROBOT + HUMAN:
        d = os.path.join(args.root, sub, "meta")
        out = os.path.join(d, "relative_stats_dreamzero.json")
        if os.path.exists(out) and not args.force:
            print(f"SKIP existing {out} (use --force)")
            sys.exit(2)
        with open(out, "w") as fh:
            fh.write(payload)
        with open(os.path.join(d, "relative_stats_dreamzero.provenance.json"), "w") as fh:
            json.dump(prov, fh, indent=2)
        print(f"wrote {out}")
    print(f"done in {time.time() - t0:.1f}s md5={prov['relative_stats_dreamzero_md5']}")


if __name__ == "__main__":
    main()
