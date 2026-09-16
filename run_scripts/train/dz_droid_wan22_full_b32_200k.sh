#!/bin/bash
# ==============================================================================================
# DreamZero DROID full fine-tune — Wan2.2-TI2V-5B — MLXP single-node 4xH200, effective batch 128, 200K steps
#
# Derived VERBATIM from scripts/train/droid_training_full_finetune_wan22.sh (HEAD ab790c19).
# The hydra recipe below is byte-identical to upstream except for the four values MLXP needs:
#   1. per_device_train_batch_size 1 -> $PER_DEVICE_BS (default 4) and global_batch_size -> 128
#      => gradient_accumulation_steps = 128 / (4 GPU * 4) = 8   (experiment/utils.py:104)
#   2. save_steps 1000 -> $SAVE_STEPS (default 5000): a 5B full fine-tune checkpoint carries
#      ZeRO-2 optimizer state (~100 GB); save_total_limit=10 => ~1 TB resident. 1000 would mean
#      200 checkpoint writes over the run; 5000 (40 writes) is our cluster convention.
#   3. the auto-download blocks are replaced by hard existence gates — a mis-typed path must NOT
#      silently start a 22 GB HF download inside a GPU job.
#   4. python selection honours $DZ_PYTHON (the uv venv interpreter). The upstream
#      /usr/bin/python3.11 probe is kept as a fallback but this image has only /usr/bin/python3.10,
#      which does NOT have the dreamzero deps.
#
# Everything else is upstream: num_frames=33 action_horizon=24 num_views=3, action head
# wan_flow_matching_action_tf_wan22, num_frame_per_block=2 num_action_per_block=24
# num_state_per_block=1, seed=42, lr 1e-5, warmup_ratio 0.05, wd 1e-5, bf16+tf32, 320x160,
# max_chunk_size=4, save_strategy=steps, save_total_limit=10, upload_checkpoints=false,
# train_architecture=full, deepspeed zero2_offload (CPU optimizer offload — what makes 5B fit).
# ==============================================================================================

export HYDRA_FULL_ERROR=1

# Repo root (same logic as droid_training_full_finetune_wan22.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -n "$DREAMZERO_ROOT" ] && [ -d "$DREAMZERO_ROOT/groot" ]; then
    :
elif [ -d "$SCRIPT_REPO_ROOT/groot" ]; then
    DREAMZERO_ROOT="$SCRIPT_REPO_ROOT"
else
    DREAMZERO_ROOT="${DREAMZERO_ROOT:-/data/huiwon/dreamzero}"
fi
if [ ! -d "$DREAMZERO_ROOT/groot" ]; then
    echo "ERROR: No groot/ under $DREAMZERO_ROOT. Set DREAMZERO_ROOT to the dreamzero repo root that contains groot/."
    exit 1
fi

# ============ USER CONFIGURATION ============
DROID_DATA_ROOT=${DROID_DATA_ROOT:-"/data/shared_dataset/DreamZero-DROID-Data"}
OUTPUT_DIR=${OUTPUT_DIR:-"/data/rlwrld-unified-checkpoints/huiwon/dreamzero/dreamzero_droid_wan22_full_b32_200k"}

NUM_GPUS=${NUM_GPUS:-4}
# ★ AR chunk 수 (huiwon 2026-09-13): DreamZero 한 샘플 = MAX_CHUNK_SIZE 개의 (video block, state, action)
#   연쇄. dataset 이 프레임 수를 8*chunk+1 로 잡으므로 NUM_FRAMES 도 같이 움직여야 한다
#   (lerobot_sharded.py:1159). chunk=1 이면 샘플 1개 = 청크 1개 = 우리 WAM DROID 샘플과 동형.
MAX_CHUNK_SIZE=${MAX_CHUNK_SIZE:-4}
NUM_FRAMES=${NUM_FRAMES:-$(( 8 * MAX_CHUNK_SIZE + 1 ))}
PER_DEVICE_BS=${PER_DEVICE_BS:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
MAX_STEPS=${MAX_STEPS:-200000}
SAVE_STEPS=${SAVE_STEPS:-5000}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10}
NW=${NW:-4}

WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"/data/huiwon/checkpoints/Wan2.2-TI2V-5B"}
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"/data/huiwon/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/data/huiwon/checkpoints/umt5-xxl"}
DEEPSPEED_CFG=${DEEPSPEED_CFG:-zero2_offload}
# =============================================

# ============ HARD GATES (replace upstream auto-download) ============
_fail() { echo "FATAL $*"; exit 1; }

# batch arithmetic must resolve to exactly GLOBAL_BATCH_SIZE with an integer accumulation
_PER_STEP=$(( PER_DEVICE_BS * NUM_GPUS ))
[ "$_PER_STEP" -gt 0 ] || _fail "PER_DEVICE_BS*NUM_GPUS = 0"
[ $(( GLOBAL_BATCH_SIZE % _PER_STEP )) -eq 0 ] \
    || _fail "global_batch_size=$GLOBAL_BATCH_SIZE not divisible by per_step=$_PER_STEP (experiment/utils.py:107 asserts this)"
_GA=$(( GLOBAL_BATCH_SIZE / _PER_STEP ))
[ "$GLOBAL_BATCH_SIZE" -eq "${EXPECT_GLOBAL_BATCH:-32}" ] || _fail "expected effective batch ${EXPECT_GLOBAL_BATCH:-32}, got $GLOBAL_BATCH_SIZE"
echo "[batch] global=$GLOBAL_BATCH_SIZE = ${NUM_GPUS} gpu x pd${PER_DEVICE_BS} x GA${_GA}"

# deepspeed ZeRO-2 + gradient accumulation > 1 requires >= 0.19.6
# (0.17.x/<0.18.3: microsoft/DeepSpeed#7718 keeps only the LAST micro-batch; 0.18.3-0.19.5: #8224 regression)
if [ "$_GA" -gt 1 ]; then
    _DSV=$("${DZ_PYTHON:-python3}" -c "import importlib.metadata as m; print(m.version('deepspeed'))" 2>/dev/null)
    case "$_DSV" in
        0.19.6|0.19.[7-9]*|0.[2-9]*|[1-9]*) echo "[deepspeed] $_DSV — ZeRO-2 with GA=$_GA is safe" ;;
        *) _fail "GA=$_GA needs deepspeed>=0.19.6 (have ${_DSV:-?}; ZeRO-2 GA bug #7718)" ;;
    esac
fi

[ -d "$DROID_DATA_ROOT/meta" ] || _fail "DROID dataset not found at $DROID_DATA_ROOT (expected data/ meta/ videos/)"
for f in "$DROID_DATA_ROOT/meta/stats.json" \
         "$DROID_DATA_ROOT/meta/relative_stats_dreamzero.json" \
         "$DROID_DATA_ROOT/meta/info.json" \
         "$DROID_DATA_ROOT/meta/modality.json"; do
    # relative_stats_dreamzero.json MUST pre-exist: lerobot.py:459-531 would otherwise COMPUTE and
    # WRITE it into the (shared, not ours) dataset root. meta/stats.json likewise (lerobot.py:420-457).
    [ -f "$f" ] || _fail "missing $f — the loader would try to WRITE it into the shared dataset root"
done
grep -q '"joint_position"' "$DROID_DATA_ROOT/meta/relative_stats_dreamzero.json" \
    || _fail "meta/relative_stats_dreamzero.json lacks joint_position (relative_action_keys)"

# every weight file the hydra command below references
for f in "$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth" \
         "$WAN22_CKPT_DIR/Wan2.2_VAE.pth" \
         "$WAN22_CKPT_DIR/diffusion_pytorch_model.safetensors.index.json" \
         "$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
         "$TOKENIZER_DIR/tokenizer.json" \
         "$TOKENIZER_DIR/spiece.model"; do
    [ -s "$f" ] || _fail "missing/empty weight file $f (upstream would silently hf-download it)"
done
# DiT shards named in the index must all be present
"${DZ_PYTHON:-python3}" - "$WAN22_CKPT_DIR" <<'PYEOF' || _fail "DiT shard check failed"
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "diffusion_pytorch_model.safetensors.index.json")
with open(idx) as f:
    wm = json.load(f)["weight_map"]
def _bad(s):
    f = os.path.join(d, s)
    return (not os.path.isfile(f)) or os.path.getsize(f) == 0
missing = [s for s in sorted(set(wm.values())) if _bad(s)]
assert not missing, f"empty/missing DiT shards: {missing}"
print("[dit] shards OK:", sorted(set(wm.values())))
PYEOF

EXPERIMENT_PY="$DREAMZERO_ROOT/groot/vla/experiment/experiment.py"
[ -f "$EXPERIMENT_PY" ] || _fail "Not found: $EXPERIMENT_PY"

# ============ PYTHON ============
if [ -n "${DZ_PYTHON:-}" ] && [ -x "$DZ_PYTHON" ]; then
    PY="$DZ_PYTHON"
elif [ -x "/usr/bin/python3.11" ]; then
    PY="/usr/bin/python3.11"
else
    PY="$(command -v python3)"
fi
echo "[python] $PY -> $("$PY" -V 2>&1)"
case "$("$PY" -V 2>&1)" in *" 3.11."*) ;; *) _fail "need python 3.11 (pyproject requires-python ~=3.11,<3.13), got $("$PY" -V 2>&1)";; esac
"$PY" -c "import torch, transformers, deepspeed, decord, groot.vla.experiment.experiment" \
    || _fail "dreamzero env incomplete for $PY"

RUN_CMD=( "$PY" -m torch.distributed.run --nproc_per_node "$NUM_GPUS" --standalone "$EXPERIMENT_PY" )
cd "$DREAMZERO_ROOT" || _fail "cd $DREAMZERO_ROOT"
mkdir -p "$OUTPUT_DIR"

# auto-resume is implicit: experiment/base.py:644 get_checkpoint_path($OUTPUT_DIR) picks the highest
# checkpoint-N and passes resume_from_checkpoint=True. A config.json in $OUTPUT_DIR means "finished"
# and the process exits 0 without training (utils.py:57).
if [ -f "$OUTPUT_DIR/config.json" ]; then
    echo "[resume] WARNING: $OUTPUT_DIR/config.json exists -> the run is considered FINISHED and will exit(0)."
fi
_LAST=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
echo "[resume] latest checkpoint in OUTPUT_DIR: ${_LAST:-none (fresh run from step 0)}"

echo "[launch] $(date -u +%FT%TZ) full fine-tune Wan2.2-TI2V-5B, b$GLOBAL_BATCH_SIZE (pd$PER_DEVICE_BS x ${NUM_GPUS}gpu x GA$_GA), ${MAX_STEPS} steps, save every $SAVE_STEPS -> $OUTPUT_DIR"

"${RUN_CMD[@]}" \
    report_to=wandb \
    data=dreamzero/droid_relative_wan22 \
    wandb_project=${WANDB_PROJECT:-dreamzero} \
    train_architecture=full \
    num_frames=$NUM_FRAMES \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/${DEEPSPEED_CFG}.json" \
    save_steps=$SAVE_STEPS \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$PER_DEVICE_BS \
    global_batch_size=$GLOBAL_BATCH_SIZE \
    max_steps=$MAX_STEPS \
    weight_decay=1e-5 \
    save_total_limit=$SAVE_TOTAL_LIMIT \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=true \
    dataloader_num_workers=$NW \
    image_resolution_width=320 \
    image_resolution_height=160 \
    save_lora_only=false \
    max_chunk_size=$MAX_CHUNK_SIZE \
    save_strategy=steps \
    droid_data_root=$DROID_DATA_ROOT \
    dit_version=$WAN22_CKPT_DIR \
    text_encoder_pretrained_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR
