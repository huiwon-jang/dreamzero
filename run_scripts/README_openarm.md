# DreamZero on OpenArm (`openarm_wam_v1`) — adapter notes

huiwon 2026-09-16. Branch `huiwon/openarm` of the clone `/data/huiwon/dreamzero` (upstream ab790c1 + the three
earlier patches: `experiment.py` deepspeed-first import, `wan_flow_matching_action_tf.py:800` per-sample
`has_real_action` broadcast, `lerobot_sharded.py` `_lang_range_fallback`). Everything below is additive and
marked `★ huiwon 2026-09-16` in the source. Nothing was GPU-tested (see "Not verified").

Two runs, both Wan2.2-TI2V-5B full fine-tune, AR `max_chunk_size=4` (33 frames / 96 actions / 4 states per
sample), 50K steps, save every 1000 optimizer steps keep 5, auto-resume from the fixed output dir:

| variant | data config | subsets | global batch | plate | yaml / job |
|---|---|---|---|---|---|
| robot | `data=dreamzero/openarm_wan22` | 6 robot | 32 | pd8 x 4 gpu x GA1 | `run_scripts/k8s/huiwon_dz_openarm_wan22_b32_ar4_robot_50k_4gpu.yaml` / `huiwon-dz-oav1-wan22-b32-ar4-robot-50k-4gpu` |
| robothuman | `data=dreamzero/openarm_human_wan22` | 6 robot + 5 human (video-only) | 64 (32 robot + 32 human in expectation) | pd8 x 4 gpu x GA2 | `run_scripts/k8s/huiwon_dz_openarm_wan22_b64_ar4_robothuman_50k_4gpu.yaml` / `huiwon-dz-oav1-wan22-b64-ar4-rh-50k-4gpu` |

Outputs: `/data/rlwrld-unified-checkpoints/huiwon/dreamzero/dz_openarm_wam_v1_{robot,robothuman}_b32_ar4_50k/checkpoint-N`.
Logs: `/data/huiwon/logs/gr00t-vdm/dz-oav1-wan22-{b32-ar4-robot,b64-ar4-rh}-50k-4gpu-<ts>.{out,err}`.

Submit (from the mlxp repo; the yamls are copied to `yamls/` there — substitute `REPLACE_WITH_WANDB_KEY` in a local copy first):

```bash
kubectl apply -f yamls/huiwon_dz_openarm_wan22_b32_ar4_robot_50k_4gpu.yaml -n p-rlwrld
kubectl apply -f yamls/huiwon_dz_openarm_wan22_b64_ar4_robothuman_50k_4gpu.yaml -n p-rlwrld
```

## 1. Files

New:
- `groot/vla/configs/data/dreamzero/openarm_wan22.yaml` — robot-only data config (modality configs, transforms, mixture).
- `groot/vla/configs/data/dreamzero/openarm_human_wan22.yaml` — inherits the above (absolute hydra path `/data/dreamzero/openarm_wan22`; a relative path is re-resolved under `data/dreamzero/` and fails), overrides only `train_dataset.mixture_spec`.
- `groot/vla/configs/model/dreamzero/action_head/wan_flow_matching_action_tf_wan22_openarm.yaml` — wan22 head + `frame_seqlen: 54`, `target_video 288x192`.
- `run_scripts/train/dz_openarm_wan22_ar4_50k.sh` — launcher (`DZ_VARIANT=robot|robothuman`, `PLATE_PD/PLATE_GA`, hard gates).
- `run_scripts/train/dz_openarm_preflight_compose_check.py` — hydra compose gate (no torch); `compose_cfg()` is shared with the smoke.
- `run_scripts/train/dz_openarm_data_smoke.py` — CPU dataset+transform smoke (one-episode shards).
- `run_scripts/data/gen_openarm_relative_stats_dreamzero.py` — pooled `meta/relative_stats_dreamzero.json` generator.
- `run_scripts/k8s/huiwon_dz_openarm_wan22_{b32_ar4_robot,b64_ar4_robothuman}_50k_4gpu.yaml`.

Patched (all guarded so DROID/upstream behaviour is unchanged):
- `groot/vla/data/schema/embodiment_tags.py:351,357` — `OPENARM = "openarm"`, `OPENARM_HUMAN = "openarm_human"`.
- `groot/vla/configs/model/dreamzero/transform/base.yaml:46-47` — projector indices `openarm: 33`, `openarm_human: 34` (only the prompt branch and the `embodiment_id` tensor use them; `CausalWanModel` forces `embodiment_id=0` for its category-specific encoders, `wan_video_dit_action_casual_chunk.py:1365,1755`).
- `groot/vla/model/dreamzero/transform/dreamzero_cotrain.py:92` `_openarm_prompt()`; `:142-145,:168-171` prompt branches in `collate` (the upstream `else: raise ValueError` would otherwise kill the run); `:354` `_prepare_video` vertical concat for 2 views; `:570` video-only handling for `OPENARM_HUMAN`.
- `groot/vla/data/dataset/lerobot.py:139,193` `black_missing_video_keys` kwarg + injection of missing view keys; `:1011,1019` `_get_step_filter` looks lengths up by position (episode ids are not contiguous in `rlrwld_human_lerobot`, 7 dropped episodes); `:1079` third `info.json` layout for the video feature (human subsets have `info.video.fps` but no `info.video.channels`); `:2509` key-wise merge of per-dataset video metadata (fps 20/30 and banana's `ego_left/ego_right` alias keys differ within one tag).
- `groot/vla/data/dataset/lerobot_sharded.py:68,386` `get_all_video_paths` -> `None` for black keys; `:496` `get_shard` fills `np.zeros_like(reference view)`; `:870,1051` `get_state/get_action` use the fallback anchors; `:1172` `_lang_range_fallback` publishes `_fallback_anchors`; `:1183` `_apply_fallback_anchors`; `:1310` a trimmed (25-frame) window now also takes the fallback.

## 2. Data adapter

**Subsets / embodiment tags.** 6 robot subsets under tag `openarm`, 5 human subsets under `openarm_human`. Each is a
separate `ShardedLeRobotSubLangSingleActionChunkDatasetDROID` (DreamZero's LeRobot loader), so all `meta/*` files are
read per subset.

**Cameras (2 views).** DreamZero keeps cameras as separate video streams (`video.camera_ego_left`, `video.camera_ego_right`,
resolved through each subset's `modality.json`; banana's `observation.image.ego_*` paths resolve through its aliases)
until `DreamTransform._prepare_video`, which builds one canvas per sample. Upstream has a DROID 2x2 layout and a generic
2x2 grid (view0 top-left, view1 bottom-left, right column black — half the canvas wasted for 2 views). For OpenArm the
new branch stacks the two views vertically, left on top / right below (the WAM canvas convention):
per view `VideoCrop(0.95, random) -> VideoResize 144x192` (`image_resolution_height/width` in the data yaml), canvas
`288x192` (HxW, 3:2 — the WAM 384x256 geometry at 0.75 scale). WanVideoVAE38 (16x) -> latent 18x12 (even, no crop in the
dynamics loss), patch (1,2,2) -> 9x6 = **54 tokens/frame** (`frame_seqlen=54`; DROID used a 160x320 canvas = 50). The
action head's `target_video_height/width = 288/192` makes its GPU resize a no-op. Per-sample tensors after the
transform+collator: `images (33, 288, 192, 3) uint8`, `state (4, 64)` (28 real dims + zero pad, `state_mask`),
`action (96, 32)` (28 real + pad, `action_mask`), `has_real_action ()`, `embodiment_id ()`, `text` -> (512,) umT5 ids.

**Missing view = black frame.** DreamZero has no option for a camera that a dataset lacks — `get_all_video_paths`
KeyErrors on `modality.json`. With `dataset_kwargs.black_missing_video_keys: true` the loader registers the missing key
against a present view (so metadata resolution / `_check_integrity` / video path resolve) and `get_shard` caches zeros of
the reference view's shape for it; the human right-only subsets (`rlwrld_human_lerobot`, the three
`openarm_validation_v2_junhyeong/*`) therefore get a black top half; `anyh2r` has both views. The prompt says
"A black view means that camera is missing." Off by default -> upstream unchanged.

**State/action (28-dim).** `modality.json` order `neck_joints(2) | left_arm_joints(7) | right_arm_joints(7) |
left_hand_joints(6) | right_hand_joints(6)` = the 28-dim layout; `max_state_dim 64 / max_action_dim 32` pads.
`relative_action: true, relative_action_keys: [left_arm_joints, right_arm_joints]`: DreamZero's own
`_convert_to_relative_action` subtracts the state at the first frame of each 24-step chunk from that chunk's actions
(exactly `DroidJointGripperTransform`-style: arms relative to the anchor state, neck/hands absolute — no extra
transform needed). Normalization is `q99` (q01/q99 -> [-1,1], clipped) for every key:
- absolute keys (neck, hands, all state): `meta/stats.json` q01/q99 — the pooled robot stats that every subset carries;
- relative arm keys: `meta/relative_stats_dreamzero.json` — DreamZero wants GLOBAL per-dim quantiles of the relative
  deltas (its `_calculate_relative_stats_for_key` pools every anchor i and horizon h of `action[i+h]-state[i]`), which is
  a different object from the WAM per-timestep 24x7 `relative_stats.json` (per-horizon quantiles cannot be combined into
  a global quantile). `run_scripts/data/gen_openarm_relative_stats_dreamzero.py` streams the 6 robot subsets with
  exactly the upstream semantics, skips banana's zero-filled `left_arm_joints` (`meta/wam_action_groups.json`), and
  writes the SAME file into all 11 subsets (md5 `d8ed2ca11429710c6fb795d27767e300`, provenance next to it; pooled
  samples 3,945,768 left / 4,304,808 right). q01/q99 (rad): left `[-0.228..0.181, -0.106..0.070, -0.227..0.224,
  -0.252..0.288, -0.306..0.374, -0.186..0.264, -0.237..0.123]`, right `[-0.238..0.307, -0.138..0.146, -0.395..0.404,
  -0.368..0.435, -0.416..0.325, -0.314..0.280, -0.162..0.252]`.
  Identical files matter twice: (a) if the file is missing the loader computes per-subset stats and WRITES them into
  the dataset (for human subsets that would be all-zero stats); (b) `LeRobotMixtureDataset.update_metadata` re-sets every
  transform's stats to the per-tag MERGED metadata (`min_max` mixing across subsets), which is only a no-op because the
  inputs are identical. The launcher gates on the marker and on one md5 across subsets.

**Banana groups.** `banana_v21_openarm28` has real data only in `right_arm/right_hand`; neck/left_arm/left_hand are
zero-filled. DreamZero has no per-group action mask, so for banana samples those dims train toward the normalized value
of 0 (state 0 -> `2*(0-q01)/(q99-q01)-1`, relative arm delta 0 -> ~0). Accepted and stated; ~8% of robot frames.

**fps.** Subsets are 20 fps (teleop_v3, the four human_as_openarm28 except anyh2r) or 30 fps (jungwook, banana,
anyh2r). DreamZero's sampler is frame-based (video: 8 frames at stride 3 per chunk, 24 actions per chunk, `fps` kwarg
unused for decord) -> a chunk spans 0.8 s at 30 fps and 1.2 s at 20 fps. Accepted (mixed real-time horizon), no
resampling; the `fps` per-embodiment map only exists for yam.

**Language.** `annotation.human.task_description` -> `task_index` column -> `meta/tasks.jsonl` string (72 strings over the
root). Exactly like the DROID run there is no text cache: the collator tokenizes online with the umT5 tokenizer
(`/data/huiwon/checkpoints/umt5-xxl`, tokenizer files only) after prefixing `_openarm_prompt()`, and the action head runs
the umT5-XXL encoder every step (`encode_prompt`).

**Sampler / window consistency (pd>1 fix).** Upstream trains with `per_device_train_batch_size=1`; at pd>1 the collator
`np.stack`s samples, so every sample must have exactly `8*max_chunk_size+1` frames. Upstream's language-anchored sampler
returns a trimmed 25-frame window when the last chunk anchor is exactly `len-24`, and the earlier `_lang_range_fallback`
(clipped forward window `anchor+3k`) only re-anchored the video, while `get_state/get_action` kept their own (possibly
earlier) chunks. Now the fallback also publishes its chunk anchors (`_fallback_anchors[first_idx]`), `get_state/get_action`
consume them (`_apply_fallback_anchors`), and the trimmed-window case takes the fallback too. Frames/states/actions past the
episode end repeat the last frame/state/action (chunks fully past the end become "hold still"). Verified by value on every
anchor of two robot episodes in the smoke (0.6% of anchors hit the fallback in a 333-frame episode; short human episodes
(< 97 frames) always do).

## 3. Human = video-only

`OPENARM_HUMAN` samples: `DreamTransform.apply_single` (`dreamzero_cotrain.py:570`) sets `has_real_action = 0`, zeroes
`action`, `action_mask`, `state`, `state_mask`. The action loss is `mse * action_mask`, then multiplied by
`has_real_action.reshape(B,1,1)` (`wan_flow_matching_action_tf.py:800`, our earlier broadcast patch), so the action loss of
these samples is exactly 0; the dynamics (video) loss is computed for every sample regardless (`:789`) — that is what
trains on human video. `is_cotrain_instance` / `state_mask` are not consumed by the model; the zero state token is what
DreamZero itself does for its LAPA/DREAM instances. The zero-filled parquet placeholders never reach the model as "real"
values.

## 4. Mixing

`mixture_spec` elements with `distribute_weights: true` and `balance_dataset_weights: false`: inside an element the
weight is split frame-proportionally (`len_i / sum len`), elements are then normalized. Robot-only: 6 subsets, weights
0.151 / 0.092 / 0.075 / 0.035 / 0.105 / 0.042 (jungwook / bottle / cup / doll / snack / banana). Robot+human: two elements
of weight 1.0 -> robot 0.5 : human 0.5 (human split 0.106 rlwrld / 0.020 close / 0.040 box / 0.017 open / 0.318 anyh2r).
DreamZero does NOT support per-batch ratios: `ShardedLeRobotMixtureDataset` draws whole shards (~1e4 frames of ONE
subset) with probability `weight_i * shard_len / dataset_len` and yields 1000 samples per visited shard, so a per-device
micro-batch is homogeneous (all robot or all human) and 32:32 holds in expectation over steps, not per optimizer step
(WAM's `MIX_EXACT` is not reproducible). `dataset_shard_sampling_rate 0.1`, seed 42, schedule re-seeded on resume
(`base.py:559`).

## 5. Plates and memory

- robot: pd8 x 4 gpu x GA1 = 32. robothuman: pd8 x 4 gpu x GA2 = 64 — chosen over pd16 GA1 so the per-micro-batch
  footprint is identical to the robot plate (one plate to validate), and GA=2 on ZeRO-2 is safe on the pinned
  deepspeed 0.19.6 (launcher re-asserts; #7718/#8224). Fallback via spec.env: `PLATE_PD=4 PLATE_GA=2` (robot) /
  `PLATE_PD=4 PLATE_GA=4` (robothuman); the launcher refuses any plate whose product is not the expected global batch.
- Memory expectation, not measured: the earlier chunk-4 b32 DROID arms never reached a memory test (they died at the
  `has_real_action` broadcast bug); the ar1 arms were parked before a step completed. Per sample the sequence is small
  (33 frames -> 9 latent frames x 54 tokens = 486 video tokens + 96 action + 4 state tokens; DROID 160x320: 450) and
  the 5B full fine-tune is dominated by weights/optimizer: bf16 weights ~10 GB per GPU, ZeRO-2 CPU-offloaded Adam
  (`zero2_offload.json`), gradient checkpointing on (`use_gradient_checkpointing: true`). pd8 with 33x288x192 uint8
  video is ~13 MB per sample on the host side. Expect pd8 to fit on 141 GB H200; VAE/umT5/CLIP run every step in bf16
  and the umT5-XXL encoder (~11 GB bf16) is resident — the same as the DROID run.
- Checkpoints: 5B full FT + ZeRO-2 optimizer state ~100 GB each; keep 5 => ~0.5 TB per run.
- Dataloader: one-episode CPU cost in the debug pod (~1 core) was ~2.3 s/sample (decord decode + crop/resize/jitter
  on 2x33 frames); with 14 cores/GPU and `NW=4` workers this should stay ahead of a multi-second optimizer step, but it is
  the first thing to look at if s/it is far above the DROID ar1 rate (7.4 s/it at pd16).

## 6. LR / schedule (DreamZero defaults, deliberately NOT our min-lr rule)

`lr 1e-5` (upstream DROID full-FT value), `warmup_ratio 0.05` = 2500 steps of 50K, `lr_scheduler_type cosine` (HF
Trainer cosine decays to 0 — there is no min-lr floor; our WAM runs use `min_lr_ratio 0.1`, this is left as DreamZero
ships it), `weight_decay 1e-5`, AdamW `betas (0.95, 0.999)` `eps 1e-8` (`conf.yaml`), `max_grad_norm` HF default 1.0,
bf16 + tf32, seed 42. `save_steps` is an HF Trainer optimizer-step cadence (GA-invariant); the previous DROID run showed
"실효 save_steps 5000" because it was configured 5000, not because of GA.

## 7. Verification performed (no GPU)

- `python -m py_compile` on every patched/new .py; `bash -n` on the launcher; both k8s yamls `kubectl apply --dry-run=server` OK.
- `dz_openarm_preflight_compose_check.py` for both variants: FAILURES 0 (batch arithmetic GA 1 / 2, frame_seqlen 54 ==
  (2*144/32)*(192/32), target 288x192, relative keys, black-view flag, mixture paths, LR/schedule values).
- `dz_openarm_data_smoke.py` (robothuman): builds the real mixture (11 datasets, weights above); for a jungwook episode
  (333 frames, 30 fps), a banana episode, a rlwrld right-only human episode (20 fps) and an anyh2r episode: every anchor
  produced `images (33,288,192,3) uint8 / state (4,64) / action (96,32)`; video/state/action chunk anchors verified by value
  (normal path and fallback windows); relative right-arm rows == `action[anchor+h] - state[anchor]`; normalized actions in
  [-1,1] with arm clip fraction ~1.75% (q01/q99 semantics); human samples `has_real_action=False`, zero action/state,
  embodiment_id 34, top half of the canvas black for right-only subsets, both halves live for anyh2r; a mixed
  [robot, human] batch through `DefaultDataCollator` gives text ids/mask (2,512), images (2,33,288,192,3) uint8,
  `has_real_action = [True, False]`, and the decoded prompts carry the openarm prefixes.

## 8. Not verified (needs a GPU)

- Model forward/backward with `frame_seqlen=54` / 288x192 latents (the DiT is RoPE-based and `frame_seqlen` is a config
  value consumed consistently, but no one has run this canvas), memory at pd8, step time.
- The first-frame CLIP path (`encode_image`) at 288x192 — it is resolution-agnostic in code, untested here.
- ZeRO-2 GA=2 end-to-end on this clone (verified only by the deepspeed version gate).
- Resume from a checkpoint into a re-seeded shard schedule; wandb.
- The real `__iter__` (full 1e4-frame shard caching, background prefetch, per-worker schedule) — the smoke drives
  one-episode shards through `get_shard`/`get_step_data`/`transforms` directly.

## 9. Reproduce the checks

```bash
export DREAMZERO_ROOT=/data/huiwon/dreamzero OPENARM_DATA_ROOT=/data/huiwon/data/openarm_wam_v1 \
  WAN22_CKPT_DIR=/data/huiwon/checkpoints/Wan2.2-TI2V-5B IMAGE_ENCODER_DIR=/data/huiwon/checkpoints/Wan2.1-I2V-14B-480P \
  TOKENIZER_DIR=/data/huiwon/checkpoints/umt5-xxl OUTPUT_DIR=/tmp/x HF_HUB_OFFLINE=1
cd /data/huiwon/dreamzero
.venv/bin/python run_scripts/data/gen_openarm_relative_stats_dreamzero.py --dry-run      # pooled relative stats
DZ_VARIANT=robot      GLOBAL_BATCH_SIZE=32 .venv/bin/python run_scripts/train/dz_openarm_preflight_compose_check.py
DZ_VARIANT=robothuman GLOBAL_BATCH_SIZE=64 .venv/bin/python run_scripts/train/dz_openarm_preflight_compose_check.py
DZ_VARIANT=robothuman GLOBAL_BATCH_SIZE=64 .venv/bin/python run_scripts/train/dz_openarm_data_smoke.py   # ~10 min on 1 core
```
