"""Compose the exact hydra config the launcher will pass, WITHOUT importing torch.

Proves: every override key exists, the data/action-head/transform groups resolve,
and the batch/save/step values land where we expect.
"""
import os
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = os.environ.get("DREAMZERO_ROOT", "/data/huiwon/dreamzero")
CFG_DIR = os.path.join(ROOT, "groot/vla/configs")
DATA = os.environ["DROID_DATA_ROOT"]
W22 = os.environ["WAN22_CKPT_DIR"]
IENC = os.environ["IMAGE_ENCODER_DIR"]
TOK = os.environ["TOKENIZER_DIR"]
OUT = os.environ["OUTPUT_DIR"]
PD = int(os.environ.get("PER_DEVICE_BS", "4"))
GB = int(os.environ.get("GLOBAL_BATCH_SIZE", "128"))
MS = int(os.environ.get("MAX_STEPS", "200000"))
SS = int(os.environ.get("SAVE_STEPS", "5000"))
STL = int(os.environ.get("SAVE_TOTAL_LIMIT", "10"))
NW = int(os.environ.get("NW", "4"))
NG = int(os.environ.get("NUM_GPUS", "4"))
MCS = int(os.environ.get("MAX_CHUNK_SIZE", "4"))
NF = int(os.environ.get("NUM_FRAMES", str(8 * MCS + 1)))
DSC = os.environ.get("DEEPSPEED_CFG", "zero2_offload")

overrides = [
    "report_to=wandb",
    "data=dreamzero/droid_relative_wan22",
    "wandb_project=dreamzero",
    "train_architecture=full",
    f"num_frames={NF}",
    "action_horizon=24",
    "num_views=3",
    "model=dreamzero/vla",
    "model/dreamzero/action_head=wan_flow_matching_action_tf_wan22",
    "model/dreamzero/transform=dreamzero_cotrain",
    "num_frame_per_block=2",
    "num_action_per_block=24",
    "num_state_per_block=1",
    "seed=42",
    "training_args.learning_rate=1e-5",
    f"training_args.deepspeed=groot/vla/configs/deepspeed/{DSC}.json",
    f"save_steps={SS}",
    "training_args.warmup_ratio=0.05",
    f"output_dir={OUT}",
    f"per_device_train_batch_size={PD}",
    f"global_batch_size={GB}",
    f"max_steps={MS}",
    "weight_decay=1e-5",
    f"save_total_limit={STL}",
    "upload_checkpoints=false",
    "bf16=true",
    "tf32=true",
    "eval_bf16=true",
    "dataloader_pin_memory=true",
    f"dataloader_num_workers={NW}",
    "image_resolution_width=320",
    "image_resolution_height=160",
    "save_lora_only=false",
    f"max_chunk_size={MCS}",
    "save_strategy=steps",
    f"droid_data_root={DATA}",
    f"dit_version={W22}",
    f"text_encoder_pretrained_path={W22}/models_t5_umt5-xxl-enc-bf16.pth",
    f"image_encoder_pretrained_path={IENC}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    f"vae_pretrained_path={W22}/Wan2.2_VAE.pth",
    f"tokenizer_path={TOK}",
]

with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
    cfg = compose(config_name="conf", overrides=overrides)

ah = cfg.action_head_cfg.config
checks = [
    ("global_batch_size", cfg.global_batch_size, GB),
    ("per_device_train_batch_size", cfg.training_args.per_device_train_batch_size, PD),
    ("max_steps", cfg.training_args.max_steps, MS),
    ("save_steps", cfg.training_args.save_steps, SS),
    ("save_total_limit", cfg.training_args.save_total_limit, STL),
    ("training_args.learning_rate", cfg.training_args.learning_rate, 1e-05),
    ("training_args.warmup_ratio", cfg.training_args.warmup_ratio, 0.05),
    ("weight_decay", cfg.training_args.weight_decay, 1e-05),
    ("lr_scheduler_type", cfg.training_args.lr_scheduler_type, "cosine"),
    ("bf16", cfg.training_args.bf16, True),
    ("tf32", cfg.training_args.tf32, True),
    ("seed", cfg.training_args.seed, 42),
    ("deepspeed", cfg.training_args.deepspeed, f"groot/vla/configs/deepspeed/{DSC}.json"),
    ("dataloader_num_workers", cfg.training_args.dataloader_num_workers, NW),
    ("num_frames", cfg.num_frames, NF),
    ("action_horizon", cfg.action_horizon, 24),
    ("num_views", cfg.num_views, 3),
    ("train_architecture", ah.train_architecture, "full"),
    ("save_lora_only", cfg.save_lora_only, False),
    ("upload_checkpoints", cfg.upload_checkpoints, False),
    ("image_resolution_width", cfg.image_resolution_width, 320),
    ("image_resolution_height", cfg.image_resolution_height, 160),
    ("max_chunk_size", cfg.max_chunk_size, MCS),
    ("frame_seqlen", ah.diffusion_model_cfg.frame_seqlen, 50),
    ("dit.model_type", ah.diffusion_model_cfg.model_type, "ti2v"),
    ("dit.dim", ah.diffusion_model_cfg.dim, 3072),
    ("dit.in_dim", ah.diffusion_model_cfg.in_dim, 48),
    ("dit.num_layers", ah.diffusion_model_cfg.num_layers, 30),
    ("dit.num_frame_per_block", ah.diffusion_model_cfg.num_frame_per_block, 2),
    ("dit.num_action_per_block", ah.diffusion_model_cfg.num_action_per_block, 24),
    ("dit.num_state_per_block", ah.diffusion_model_cfg.num_state_per_block, 1),
    ("dit_path", ah.diffusion_model_cfg.diffusion_model_pretrained_path, W22),
    ("vae._target_", ah.vae_cfg._target_,
     "groot.vla.model.dreamzero.modules.wan_video_vae.WanVideoVAE38"),
    ("vae.z_dim", ah.vae_cfg.z_dim, 48),
    ("vae_path", ah.vae_cfg.vae_pretrained_path, f"{W22}/Wan2.2_VAE.pth"),
    ("t5_path", ah.text_encoder_cfg.text_encoder_pretrained_path,
     f"{W22}/models_t5_umt5-xxl-enc-bf16.pth"),
    ("clip_path", ah.image_encoder_cfg.image_encoder_pretrained_path,
     f"{IENC}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
    ("droid_data_root", cfg.droid_data_root, DATA),
    ("target_video_height", ah.target_video_height, 160),
    ("target_video_width", ah.target_video_width, 320),
    ("relative_action", cfg.relative_action, True),
    ("output_dir", cfg.training_args.output_dir, OUT),
]

bad = 0
for name, got, want in checks:
    ok = got == want
    bad += (not ok)
    print(("  OK  " if ok else "  FAIL") + f" {name} = {got!r}" + ("" if ok else f"  (want {want!r})"))

# grad accum arithmetic exactly as experiment/utils.py:104
world = NG
per_step = cfg.training_args.per_device_train_batch_size * world
assert cfg.global_batch_size % per_step == 0
ga = cfg.global_batch_size // per_step
print(f"\n  grad_accum = {cfg.global_batch_size} // ({cfg.training_args.per_device_train_batch_size} * {world}) = {ga}")
print(f"  effective batch = {cfg.training_args.per_device_train_batch_size} * {world} * {ga} = "
      f"{cfg.training_args.per_device_train_batch_size * world * ga}")
assert cfg.training_args.per_device_train_batch_size * world * ga == GB   # GB from env (b32 arms: 32; upstream example: 128)

# tokenizer path inside the transform subtree
tok = None
print(f"  transform tokenizer_path (cfg root) = {cfg.tokenizer_path}")

print("\nFAILURES:", bad)
sys.exit(1 if bad else 0)
