#!/bin/bash
# ==============================================================================================
# DreamZero OpenArm full fine-tune — Wan2.2-TI2V-5B — MLXP single-node 4xH200 — AR max_chunk_size=4, 50K steps
#
# huiwon 2026-09-16. Derived from dz_droid_wan22_full_b32_200k.sh (itself a verbatim derivative of upstream
# scripts/train/droid_training_full_finetune_wan22.sh). Two variants, selected by DZ_VARIANT:
#   robot       data=dreamzero/openarm_wan22        6 robot subsets, global batch 32 (pd8 x 4 gpu x GA1)
#   robothuman  data=dreamzero/openarm_human_wan22  6 robot + 5 human (video-only) subsets, global batch 64
#                                                   (pd8 x 4 gpu x GA2 = 32 robot + 32 human IN EXPECTATION)
# Plate knobs (yaml spec.env or shell): PLATE_PD (per-device batch), PLATE_GA (grad accum). The effective batch
# PLATE_PD*NUM_GPUS*PLATE_GA must equal EXPECT_GLOBAL_BATCH (32 / 64) or the launcher refuses (fail-fast, no
# silent batch drift). OOM fallback: PLATE_PD=4 PLATE_GA=2 (robot) / PLATE_PD=4 PLATE_GA=4 (robothuman).
#
# Recipe = upstream DROID full-FT defaults except what OpenArm needs (all stated, see README_openarm.md):
#   num_frames=33 (=8*4+1) action_horizon=24 num_views=2 max_chunk_size=4 · action head
#   wan_flow_matching_action_tf_wan22_openarm (= wan22 + frame_seqlen 54 + target 288x192 for the vertically
#   stacked 2x(144x192) ego canvas) · transform dreamzero_cotrain · num_frame_per_block=2 num_action_per_block=24
#   num_state_per_block=1 · seed 42 · lr 1e-5 · warmup_ratio 0.05 (=2500 steps of 50K) · wd 1e-5 · cosine to 0
#   (DreamZero/HF default — NOT our WAM min_lr 0.1 rule; deliberately left as DreamZero ships it) · adam betas
#   (0.95, 0.999) from conf.yaml · bf16+tf32 · train_architecture=full · deepspeed zero2_offload · save every
#   1000 optimizer steps (HF Trainer save_steps counts optimizer steps, GA-invariant), keep 5.
# Auto-resume is implicit: experiment/base.py:644 get_checkpoint_path(OUTPUT_DIR) picks the highest
# checkpoint-N in the FIXED output dir and passes resume_from_checkpoint=True (a config.json there = finished).
# ==============================================================================================

export HYDRA_FULL_ERROR=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -n "$DREAMZERO_ROOT" ] && [ -d "$DREAMZERO_ROOT/groot" ]; then
    :
elif [ -d "$SCRIPT_REPO_ROOT/groot" ]; then
    DREAMZERO_ROOT="$SCRIPT_REPO_ROOT"
else
    DREAMZERO_ROOT="${DREAMZERO_ROOT:-/data/huiwon/dreamzero}"
fi
export DREAMZERO_ROOT
if [ ! -d "$DREAMZERO_ROOT/groot" ]; then
    echo "ERROR: No groot/ under $DREAMZERO_ROOT. Set DREAMZERO_ROOT to the dreamzero repo root that contains groot/."
    exit 1
fi

_fail() { echo "FATAL $*"; exit 1; }

# ============ USER CONFIGURATION ============
DZ_VARIANT=${DZ_VARIANT:-robot}
case "$DZ_VARIANT" in
    robot)      DATA_CFG=dreamzero/openarm_wan22;       _DEF_GA=1; _DEF_EXPECT=32; _DEF_OUT=dz_openarm_wam_v1_robot_b32_ar4_50k ;;
    robothuman) DATA_CFG=dreamzero/openarm_human_wan22; _DEF_GA=2; _DEF_EXPECT=64; _DEF_OUT=dz_openarm_wam_v1_robothuman_b32_ar4_50k ;;
    *) _fail "DZ_VARIANT must be robot|robothuman, got '$DZ_VARIANT'" ;;
esac
export DZ_VARIANT
export OPENARM_DATA_ROOT=${OPENARM_DATA_ROOT:-"/data/huiwon/data/openarm_wam_v1"}
export OUTPUT_DIR=${OUTPUT_DIR:-"/data/rlwrld-unified-checkpoints/huiwon/dreamzero/$_DEF_OUT"}

export NUM_GPUS=${NUM_GPUS:-4}
export MAX_CHUNK_SIZE=${MAX_CHUNK_SIZE:-4}
export NUM_FRAMES=${NUM_FRAMES:-$(( 8 * MAX_CHUNK_SIZE + 1 ))}
export PLATE_PD=${PLATE_PD:-8}
export PLATE_GA=${PLATE_GA:-$_DEF_GA}
export EXPECT_GLOBAL_BATCH=${EXPECT_GLOBAL_BATCH:-$_DEF_EXPECT}
export PER_DEVICE_BS=$PLATE_PD
export GLOBAL_BATCH_SIZE=$(( PLATE_PD * NUM_GPUS * PLATE_GA ))
export MAX_STEPS=${MAX_STEPS:-50000}
export SAVE_STEPS=${SAVE_STEPS:-1000}
export SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
export NW=${NW:-4}
export LR=${LR:-1e-5}
export WD=${WD:-1e-5}
export WARMUP_RATIO=${WARMUP_RATIO:-0.05}

export WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"/data/huiwon/checkpoints/Wan2.2-TI2V-5B"}
export IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"/data/huiwon/checkpoints/Wan2.1-I2V-14B-480P"}
export TOKENIZER_DIR=${TOKENIZER_DIR:-"/data/huiwon/checkpoints/umt5-xxl"}
export DEEPSPEED_CFG=${DEEPSPEED_CFG:-zero2_offload}
# =============================================

# ============ HARD GATES ============
[ "$GLOBAL_BATCH_SIZE" -eq "$EXPECT_GLOBAL_BATCH" ] \
    || _fail "plate pd${PLATE_PD} x ${NUM_GPUS} gpu x GA${PLATE_GA} = $GLOBAL_BATCH_SIZE != EXPECT_GLOBAL_BATCH=$EXPECT_GLOBAL_BATCH"
[ "$NUM_FRAMES" -eq $(( 8 * MAX_CHUNK_SIZE + 1 )) ] || _fail "NUM_FRAMES=$NUM_FRAMES != 8*MAX_CHUNK_SIZE+1 (lerobot_sharded.py sampler)"
[ "$SAVE_TOTAL_LIMIT" -ge 5 ] || _fail "save_total_limit must be >= 5 (experiment/base.py asserts it)"
echo "[batch] $DZ_VARIANT: global=$GLOBAL_BATCH_SIZE = ${NUM_GPUS} gpu x pd${PER_DEVICE_BS} x GA${PLATE_GA}; chunk=$MAX_CHUNK_SIZE frames=$NUM_FRAMES actions=$(( 24 * MAX_CHUNK_SIZE ))"

# ============ PYTHON ============
if [ -n "${DZ_PYTHON:-}" ] && [ -x "$DZ_PYTHON" ]; then
    PY="$DZ_PYTHON"
elif [ -x "$DREAMZERO_ROOT/.venv/bin/python" ]; then
    PY="$DREAMZERO_ROOT/.venv/bin/python"
elif [ -x "/usr/bin/python3.11" ]; then
    PY="/usr/bin/python3.11"
else
    PY="$(command -v python3)"
fi
echo "[python] $PY -> $("$PY" -V 2>&1)"
case "$("$PY" -V 2>&1)" in *" 3.11."*) ;; *) _fail "need python 3.11 (pyproject requires-python ~=3.11,<3.13), got $("$PY" -V 2>&1)";; esac
"$PY" -c "import torch, transformers, deepspeed, decord, groot.vla.experiment.experiment" \
    || _fail "dreamzero env incomplete for $PY"

# deepspeed ZeRO-2 + gradient accumulation > 1 requires >= 0.19.6 (#7718 / #8224)
if [ "$PLATE_GA" -gt 1 ]; then
    _DSV=$("$PY" -c "import importlib.metadata as m; print(m.version('deepspeed'))" 2>/dev/null)
    case "$_DSV" in
        0.19.6|0.19.[7-9]*|0.[2-9]*|[1-9]*) echo "[deepspeed] $_DSV — ZeRO-2 with GA=$PLATE_GA is safe" ;;
        *) _fail "GA=$PLATE_GA needs deepspeed>=0.19.6 (have ${_DSV:-?}; ZeRO-2 GA bug #7718)" ;;
    esac
fi

# dataset: every subset the chosen data config lists must be complete. relative_stats_dreamzero.json MUST
# pre-exist and be the pooled file from run_scripts/data/gen_openarm_relative_stats_dreamzero.py — otherwise
# lerobot.py:459-531 silently COMPUTES per-subset stats (zeros for human!) and WRITES them into the dataset.
ROBOT_SUBSETS="robot/openarm_ego_jungwook robot/openarm_teleop_v3/bottle robot/openarm_teleop_v3/cup robot/openarm_teleop_v3/doll robot/openarm_teleop_v3/snack robot/banana_v21_openarm28"
HUMAN_SUBSETS="human_as_openarm28/rlwrld_human_lerobot human_as_openarm28/openarm_validation_v2_junhyeong/close_air_fryer human_as_openarm28/openarm_validation_v2_junhyeong/left_hand_box_white_container human_as_openarm28/openarm_validation_v2_junhyeong/open_air_fryer human_as_openarm28/anyh2r"
SUBSETS="$ROBOT_SUBSETS"; [ "$DZ_VARIANT" = robothuman ] && SUBSETS="$ROBOT_SUBSETS $HUMAN_SUBSETS"
_REL_MD5=""
for s in $SUBSETS; do
    d="$OPENARM_DATA_ROOT/$s"
    for f in meta/info.json meta/modality.json meta/stats.json meta/episodes.jsonl meta/tasks.jsonl meta/relative_stats_dreamzero.json meta/relative_stats_dreamzero.provenance.json; do
        [ -s "$d/$f" ] || _fail "missing $d/$f"
    done
    [ -d "$d/videos" ] || _fail "missing $d/videos"
    grep -q '"left_arm_joints"' "$d/meta/relative_stats_dreamzero.json" && grep -q '"right_arm_joints"' "$d/meta/relative_stats_dreamzero.json" \
        || _fail "$d/meta/relative_stats_dreamzero.json lacks left_arm_joints/right_arm_joints"
    grep -q 'huiwon_openarm_relative_stats_dreamzero_v1' "$d/meta/relative_stats_dreamzero.provenance.json" \
        || _fail "$d relative stats are not the pooled generator output (provenance marker missing)"
    _m=$(md5sum "$d/meta/relative_stats_dreamzero.json" | cut -d' ' -f1)
    if [ -z "$_REL_MD5" ]; then _REL_MD5=$_m; elif [ "$_REL_MD5" != "$_m" ]; then _fail "relative_stats_dreamzero.json differs across subsets ($s) — must be the single pooled file"; fi
done
echo "[data] $DZ_VARIANT: $(echo $SUBSETS | wc -w) subsets OK under $OPENARM_DATA_ROOT (relative stats md5 $_REL_MD5)"

# every weight file the hydra command below references
for f in "$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
         "$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
         "$WAN22_CKPT_DIR/diffusion_pytorch_model.safetensors.index.json" \
         "$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
         "$TOKENIZER_DIR/tokenizer.json" \
         "$TOKENIZER_DIR/spiece.model"; do
    [ -s "$f" ] || _fail "missing/empty weight file $f (upstream would silently hf-download it)"
done
"$PY" - "$WAN22_CKPT_DIR" <<'PYEOF' || _fail "DiT shard check failed"
import json, os, sys
d = sys.argv[1]
with open(os.path.join(d, "diffusion_pytorch_model.safetensors.index.json")) as f:
    wm = json.load(f)["weight_map"]
missing = [s for s in sorted(set(wm.values())) if not os.path.isfile(os.path.join(d, s)) or os.path.getsize(os.path.join(d, s)) == 0]
assert not missing, f"empty/missing DiT shards: {missing}"
print("[dit] shards OK:", sorted(set(wm.values())))
PYEOF

EXPERIMENT_PY="$DREAMZERO_ROOT/groot/vla/experiment/experiment.py"
[ -f "$EXPERIMENT_PY" ] || _fail "Not found: $EXPERIMENT_PY"
[ -s "$DREAMZERO_ROOT/groot/vla/configs/deepspeed/$DEEPSPEED_CFG.json" ] || _fail "missing deepspeed cfg $DEEPSPEED_CFG.json"

# hydra composition gate (hydra+omegaconf only): every override below exists and lands where expected
"$PY" "$SCRIPT_DIR/dz_openarm_preflight_compose_check.py" || _fail "hydra compose preflight gate"

RUN_CMD=( "$PY" -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone "$EXPERIMENT_PY" )
cd "$DREAMZERO_ROOT" || _fail "cd $DREAMZERO_ROOT"
mkdir -p "$OUTPUT_DIR"

if [ -f "$OUTPUT_DIR/config.json" ]; then
    echo "[resume] WARNING: $OUTPUT_DIR/config.json exists -> the run is considered FINISHED and will exit(0)."
fi
_LAST=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
echo "[resume] latest checkpoint in OUTPUT_DIR: ${_LAST:-none (fresh run from step 0)}"

echo "[launch] $(date -u +%FT%TZ) DreamZero OpenArm ($DZ_VARIANT) full FT Wan2.2-TI2V-5B, b$GLOBAL_BATCH_SIZE (pd$PER_DEVICE_BS x ${NUM_GPUS}gpu x GA$PLATE_GA), chunk $MAX_CHUNK_SIZE, ${MAX_STEPS} steps, save every $SAVE_STEPS keep $SAVE_TOTAL_LIMIT -> $OUTPUT_DIR"

# NOTE: keep this override list in sync with dz_openarm_preflight_compose_check.py::overrides()
"${RUN_CMD[@]}" \
    report_to=wandb \
    data=$DATA_CFG \
    wandb_project=${WANDB_PROJECT:-dreamzero} \
    train_architecture=full \
    num_frames=$NUM_FRAMES \
    action_horizon=24 \
    num_views=2 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22_openarm \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=$LR \
    training_args.deepspeed="groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json" \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=$WARMUP_RATIO \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$PER_DEVICE_BS \
    global_batch_size=$GLOBAL_BATCH_SIZE \
    max_steps=$MAX_STEPS \
    weight_decay=$WD \
    save_total_limit=$SAVE_TOTAL_LIMIT \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=true \
    dataloader_num_workers=$NW \
    image_resolution_width=192 \
    image_resolution_height=144 \
    save_lora_only=false \
    max_chunk_size=$MAX_CHUNK_SIZE \
    save_strategy=steps \
    openarm_data_root=$OPENARM_DATA_ROOT \
    dit_version=$WAN22_CKPT_DIR \
    text_encoder_pretrained_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR
