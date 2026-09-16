"""huiwon 2026-09-16: CPU data smoke for the OpenArm DreamZero adapter (no GPU, no model).

Builds the real ShardedLeRobotMixtureDataset from the composed config (same overrides as the launcher, via
dz_openarm_preflight_compose_check.compose_cfg), then for a few (subset, episode) pairs loads a ONE-episode
shard (get_shard on a single trajectory — a full 1e4-frame shard would not fit the 4 GiB debug pod), runs
get_step_data + transforms for EVERY allowed anchor of that episode and checks:
  * per-sample shapes: images (33, 288, 192, 3) uint8 canvas, state (4, 64), action (96, 32), masks
  * has_real_action / embodiment_id / prompt text per embodiment
  * video/state/action chunk-anchor ALIGNMENT (normal sampler path and the clipped fallback path)
  * robot: normalized relative arm dims in [-1, 1], clip fraction, raw deltas vs relative_stats q01/q99
  * human right-only subset: top half of the canvas (camera_ego_left) is black; anyh2r: both views live
  * a mixed [robot, human] batch through DefaultDataCollator (umT5 tokenizer): text ids / masks / dtypes
Env: same as the preflight (DZ_VARIANT must be robothuman to cover the human subsets).
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dz_openarm_preflight_compose_check import compose_cfg  # noqa: E402

from hydra.utils import instantiate  # noqa: E402
import torch  # noqa: E402

FAIL = 0


def check(cond, msg):
    global FAIL
    print(("  OK   " if cond else "  FAIL ") + msg)
    FAIL += (not cond)


def load_one_episode(ds, tid):
    frames, starts, df = ds.get_shard(
        [tid], ds.modality_keys, ds.all_video_paths, ds.all_parquet_paths, ds.video_backend, ds.video_backend_kwargs, ds.fps
    )
    ds.cached_shard, ds.shard_start_indices, ds.cached_df = frames, starts, df
    return frames, df


def run_episode(ds, tid, tag, human, max_anchors=None):
    t0 = time.time()
    frames, df = load_one_episode(ds, tid)
    L = len(df)
    vkeys = ds.modality_keys["video"]
    print(f"\n=== {tag} ({ds.dataset_name}) episode {tid}: len={L} fps={ds.lerobot_info_meta['fps']} "
          f"black_keys={sorted(ds.black_video_keys)} cached {[(k, frames[k].shape) for k in vkeys]}")
    S = np.stack(df["observation.state"].values).astype(np.float64)
    A = np.stack(df["action"].values).astype(np.float64)
    allowed = L - ds.max_delta_index  # mixture __iter__: allowed_indices <= trajectory_length - max_delta_index
    anchors = list(range(0, allowed + 1))
    if max_anchors and len(anchors) > max_anchors:
        step = max(1, len(anchors) // max_anchors)
        anchors = sorted(set(anchors[::step]) | set(anchors[:3]) | set(anchors[-3:]))
    n_fb = 0
    shapes_ok = True
    align_ok = True
    rng_ok = True
    clip_frac = []
    absmax = 0.0
    first = None
    for a in anchors:
        indices = {k: d + a for k, d in ds.delta_indices.items()}
        raw = ds.get_step_data(tid, indices)
        if raw is None:
            print(f"  anchor {a}: get_step_data returned None (skipped sample)")
            shapes_ok = False
            continue
        used_fb = a in getattr(ds, "_fallback_anchors", {})
        n_fb += used_fb
        # --- alignment of chunk anchors across video / state / action (raw, pre-transform) ---
        vid = raw[vkeys[0]]  # (33, H, W, 3)
        st = raw["state.neck_joints"]  # (4, 2)
        # video: recover frame index of the first frame of each chunk by matching against the cached frames
        ref = frames[vkeys[0]]
        v_anchor = []
        for c in range(ds.max_chunk_size):
            f = vid[8 * c]
            hits = np.where((ref == f).reshape(len(ref), -1).all(axis=1))[0]
            v_anchor.append(int(hits[0]) if len(hits) else -1)
        raw_right_arm = raw["action.right_arm_joints"].copy() if not human else None  # transforms mutate raw in place
        if not human:
            if -1 in v_anchor:
                align_ok = False
                print(f"  anchor {a} fb={used_fb}: video chunk frames not found in cache {v_anchor}")
            else:
                # state row c must equal the parquet state at the video chunk anchor; absolute action groups
                # (neck/hands) at row 24c must equal the parquet action there; relative arm rows must equal
                # action[anchor+h] - state[anchor] (clipped to the last frame, like the fallback window)
                s_full = np.concatenate([raw[f"state.{k}"] for k in ["neck_joints", "left_arm_joints", "right_arm_joints", "left_hand_joints", "right_hand_joints"]], axis=1)
                a_abs = np.concatenate([raw["action.neck_joints"], raw["action.left_hand_joints"], raw["action.right_hand_joints"]], axis=1)
                A_abs = np.concatenate([A[:, 0:2], A[:, 16:22], A[:, 22:28]], axis=1)
                for c, ac in enumerate(v_anchor):
                    hs = np.minimum(ac + np.arange(24), L - 1)
                    ok_s = np.allclose(s_full[c], S[ac], atol=1e-5)
                    ok_a = np.allclose(a_abs[c * 24:(c + 1) * 24], A_abs[hs], atol=1e-5)
                    ok_r = np.allclose(raw_right_arm[c * 24:(c + 1) * 24], A[hs, 9:16] - S[ac, 9:16], atol=1e-5)
                    if not (ok_s and ok_a and ok_r):
                        align_ok = False
                        print(f"  anchor {a} fb={used_fb}: MISALIGNED at chunk {c} (video anchor {ac}): state={ok_s} abs_action={ok_a} rel_right_arm={ok_r}")
                        break
        else:
            if -1 in v_anchor:
                align_ok = False
                print(f"  anchor {a}: video chunk frames not found in cache {v_anchor}")
        # --- transform ---
        out = ds.transforms(raw)
        img, state, act = out["images"], out["state"], out["action"]
        ok = (img.shape == (33, 288, 192, 3) and img.dtype == np.uint8 and state.shape == (4, 64)
              and act.shape == (96, 32) and out["action_mask"].shape == (96, 32) and out["state_mask"].shape == (4, 64))
        if not ok:
            shapes_ok = False
            print(f"  anchor {a}: bad shapes images={img.shape} {img.dtype} state={state.shape} action={act.shape}")
        if not human:
            arm = act[:, 2:16]
            absmax = max(absmax, float(np.abs(act[:, :28]).max()))
            clip_frac.append(float((np.abs(arm) >= 0.999).mean()))
            if act.min() < -1 or act.max() > 1:
                rng_ok = False
        if first is None:
            first = (raw_right_arm, out)
    dt = time.time() - t0
    print(f"  anchors={len(anchors)} fallback={n_fb} ({n_fb/len(anchors):.1%}) time={dt:.1f}s")
    check(shapes_ok, f"{tag}: every sample is images(33,288,192,3) uint8 / state(4,64) / action(96,32)")
    check(align_ok, f"{tag}: video/state/action chunk anchors aligned for every anchor (incl. {n_fb} fallback windows)")
    raw_right_arm, out = first
    check(bool(out["has_real_action"]) == (not human), f"{tag}: has_real_action={bool(out['has_real_action'])}")
    check(int(out["embodiment_id"]) == (34 if human else 33), f"{tag}: embodiment_id={int(out['embodiment_id'])}")
    print(f"  text: {out['text']!r}")
    if not human:
        check(rng_ok and absmax <= 1.0, f"{tag}: normalized action in [-1,1] (|max|={absmax:.3f}); arm clip fraction mean={np.mean(clip_frac):.4f}")
        # raw relative deltas vs the loaded relative stats
        rel = ds.lerobot_relative_stats_meta["right_arm_joints"]
        d = raw_right_arm
        print(f"  right_arm raw delta range this sample: [{d.min():+.3f}, {d.max():+.3f}]; stats q01={np.round(rel.q01,3).tolist()} q99={np.round(rel.q99,3).tolist()}")
        print(f"  normalized state[0,:28]={np.round(state[0,:28],2).tolist()}")
        check(np.all(np.abs(state[:, :28]) <= 1.0) and np.all(state[:, 28:] == 0), f"{tag}: state normalized to [-1,1], padded dims zero")
    else:
        check(np.all(out["action"] == 0) and np.all(out["action_mask"] == 0) and np.all(out["state"] == 0),
              f"{tag}: action/action_mask/state all zero (video-only)")
        top, bot = out["images"][:, :144], out["images"][:, 144:]
        if ds.black_video_keys:
            check(top.max() == 0 and bot.max() > 0, f"{tag}: camera_ego_left missing -> top half black (max {top.max()}), bottom live (max {bot.max()})")
        else:
            check(top.max() > 0 and bot.max() > 0, f"{tag}: both views live (top max {top.max()}, bottom max {bot.max()})")
    ds.delete_cached_shard()
    return out


def main():
    cfg, e = compose_cfg()
    print(f"variant={e['variant']} data={e['data_cfg']}")
    t0 = time.time()
    mix = instantiate(cfg.train_dataset)
    print(f"mixture built in {time.time()-t0:.1f}s: {len(mix.datasets)} datasets, len={len(mix)}")
    names = [d.dataset_name for d in mix.datasets]
    w = mix.dataset_sampling_weights
    lens = mix.dataset_lengths
    for n, d, wi, li in zip(names, mix.datasets, w, lens):
        print(f"  {d.tag.value:14s} {n:32s} steps={li:7d} weight={wi:.4f} shards={d.num_shards} black={sorted(d.black_video_keys)}")
    robot_w = float(sum(wi for d, wi in zip(mix.datasets, w) if d.tag.value == "openarm"))
    human_w = float(sum(wi for d, wi in zip(mix.datasets, w) if d.tag.value == "openarm_human"))
    print(f"  robot weight sum={robot_w:.4f} human weight sum={human_w:.4f}")
    rl = np.array([li for d, li in zip(mix.datasets, lens) if d.tag.value == "openarm"], dtype=float)
    rw = np.array([wi for d, wi in zip(mix.datasets, w) if d.tag.value == "openarm"], dtype=float)
    check(np.allclose(rw / rw.sum(), rl / rl.sum(), atol=1e-6), "robot subset weights are frame-proportional")
    if e["variant"] == "robothuman":
        check(abs(robot_w - 0.5) < 1e-6 and abs(human_w - 0.5) < 1e-6, "robot:human = 0.5:0.5 (in expectation, shard-level sampling)")
        check(len(mix.datasets) == 11, "11 datasets")
    else:
        check(len(mix.datasets) == 6 and abs(robot_w - 1.0) < 1e-6, "6 robot datasets, weight 1.0")

    by_name = {d.dataset_name: d for d in mix.datasets}
    outs = {}
    outs["robot"] = run_episode(by_name["openarm_ego_jungwook"], int(by_name["openarm_ego_jungwook"].trajectory_ids[0]), "robot/jungwook", human=False, max_anchors=60)
    b = by_name["banana_v21_openarm28"]
    run_episode(b, int(b.trajectory_ids[0]), "robot/banana", human=False, max_anchors=40)
    if e["variant"] == "robothuman":
        h = by_name["rlwrld_human_lerobot"]
        outs["human"] = run_episode(h, int(h.trajectory_ids[0]), "human/rlwrld(right-only)", human=True, max_anchors=60)
        h2 = by_name["anyh2r"]
        run_episode(h2, int(h2.trajectory_ids[0]), "human/anyh2r(2 views)", human=True, max_anchors=40)

    # --- collate a mixed batch through the real collator ---
    coll = instantiate(cfg.data_collator)
    feats = [outs["robot"]] + ([outs["human"]] if "human" in outs else [outs["robot"]])
    batch = coll(feats)
    print("\ncollated batch:", {k: (tuple(v.shape), str(v.dtype)) for k, v in batch.items()})
    check(tuple(batch["text"].shape) == (2, cfg.max_length) and tuple(batch["text_attention_mask"].shape) == (2, cfg.max_length), "text ids/mask (2, 512)")
    check(tuple(batch["images"].shape) == (2, 33, 288, 192, 3) and batch["images"].dtype == torch.uint8, "images (2,33,288,192,3) uint8")
    check(tuple(batch["action"].shape) == (2, 96, 32) and tuple(batch["state"].shape) == (2, 4, 64), "action (2,96,32) state (2,4,64)")
    hra = batch["has_real_action"]
    check(tuple(hra.shape) == (2,) and bool(hra[0]) and (bool(hra[1]) == ("human" not in outs)), f"has_real_action per sample = {hra.tolist()} (scalar per sample -> (B,) -> reshaped (B,1,1) in the loss)")
    tok = coll.tokenizer.tokenizer
    for i in range(2):
        print(f"  decoded prompt[{i}]: {tok.decode(batch['text'][i][batch['text_attention_mask'][i] > 0], skip_special_tokens=True)!r}")
    print("\nFAILURES:", FAIL)
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
