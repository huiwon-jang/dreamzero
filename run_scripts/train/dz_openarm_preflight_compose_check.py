"""huiwon 2026-09-16: compose the exact hydra config the OpenArm launcher passes, WITHOUT importing torch.

Usage: DZ_VARIANT=robot|robothuman OPENARM_DATA_ROOT=... WAN22_CKPT_DIR=... IMAGE_ENCODER_DIR=... TOKENIZER_DIR=...
       OUTPUT_DIR=... [PER_DEVICE_BS GLOBAL_BATCH_SIZE MAX_STEPS SAVE_STEPS SAVE_TOTAL_LIMIT NW NUM_GPUS
       MAX_CHUNK_SIZE NUM_FRAMES DEEPSPEED_CFG] python dz_openarm_preflight_compose_check.py

Proves: every override key exists, data/action-head/transform groups resolve, the mixture lists the expected
subsets, the OpenArm geometry (per-view 144x192, frame_seqlen 54, target 288x192), relative keys, the
black-view flag, and the batch/save/step values. `compose_cfg()` is reused by dz_openarm_data_smoke.py.
"""
import os
import sys

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = os.environ.get("DREAMZERO_ROOT", "/data/huiwon/dreamzero")
CFG_DIR = os.path.join(ROOT, "groot/vla/configs")

ROBOT_SUBSETS = [
    "robot/openarm_ego_jungwook",
    "robot/openarm_teleop_v3/bottle",
    "robot/openarm_teleop_v3/cup",
    "robot/openarm_teleop_v3/doll",
    "robot/openarm_teleop_v3/snack",
    "robot/banana_v21_openarm28",
]
HUMAN_SUBSETS = [
    "human_as_openarm28/rlwrld_human_lerobot",
    "human_as_openarm28/openarm_validation_v2_junhyeong/close_air_fryer",
    "human_as_openarm28/openarm_validation_v2_junhyeong/left_hand_box_white_container",
    "human_as_openarm28/openarm_validation_v2_junhyeong/open_air_fryer",
    "human_as_openarm28/anyh2r",
]


def env():
    e = {
        "variant": os.environ.get("DZ_VARIANT", "robot"),
        "data": os.environ["OPENARM_DATA_ROOT"],
        "w22": os.environ["WAN22_CKPT_DIR"],
        "ienc": os.environ["IMAGE_ENCODER_DIR"],
        "tok": os.environ["TOKENIZER_DIR"],
        "out": os.environ["OUTPUT_DIR"],
        "pd": int(os.environ.get("PER_DEVICE_BS", "8")),
        "gb": int(os.environ.get("GLOBAL_BATCH_SIZE", "32")),
        "ms": int(os.environ.get("MAX_STEPS", "50000")),
        "ss": int(os.environ.get("SAVE_STEPS", "1000")),
        "stl": int(os.environ.get("SAVE_TOTAL_LIMIT", "5")),
        "nw": int(os.environ.get("NW", "4")),
        "ng": int(os.environ.get("NUM_GPUS", "4")),
        "mcs": int(os.environ.get("MAX_CHUNK_SIZE", "4")),
        "dsc": os.environ.get("DEEPSPEED_CFG", "zero2_offload"),
        "lr": os.environ.get("LR", "1e-5"),
        "wd": os.environ.get("WD", "1e-5"),
        "warmup": os.environ.get("WARMUP_RATIO", "0.05"),
    }
    e["nf"] = int(os.environ.get("NUM_FRAMES", str(8 * e["mcs"] + 1)))
    assert e["variant"] in ("robot", "robothuman"), e["variant"]
    e["data_cfg"] = "dreamzero/openarm_wan22" if e["variant"] == "robot" else "dreamzero/openarm_human_wan22"
    return e


def overrides(e):
    return [
        "report_to=wandb",
        f"data={e['data_cfg']}",
        "wandb_project=dreamzero",
        "train_architecture=full",
        f"num_frames={e['nf']}",
        "action_horizon=24",
        "num_views=2",
        "model=dreamzero/vla",
        "model/dreamzero/action_head=wan_flow_matching_action_tf_wan22_openarm",
        "model/dreamzero/transform=dreamzero_cotrain",
        "num_frame_per_block=2",
        "num_action_per_block=24",
        "num_state_per_block=1",
        "seed=42",
        f"training_args.learning_rate={e['lr']}",
        f"training_args.deepspeed=groot/vla/configs/deepspeed/{e['dsc']}.json",
        f"save_steps={e['ss']}",
        f"training_args.warmup_ratio={e['warmup']}",
        f"output_dir={e['out']}",
        f"per_device_train_batch_size={e['pd']}",
        f"global_batch_size={e['gb']}",
        f"max_steps={e['ms']}",
        f"weight_decay={e['wd']}",
        f"save_total_limit={e['stl']}",
        "upload_checkpoints=false",
        "bf16=true",
        "tf32=true",
        "eval_bf16=true",
        "dataloader_pin_memory=true",
        f"dataloader_num_workers={e['nw']}",
        "image_resolution_width=192",
        "image_resolution_height=144",
        "save_lora_only=false",
        f"max_chunk_size={e['mcs']}",
        "save_strategy=steps",
        f"openarm_data_root={e['data']}",
        f"dit_version={e['w22']}",
        f"text_encoder_pretrained_path={e['w22']}/models_t5_umt5-xxl-enc-bf16.pth",
        f"image_encoder_pretrained_path={e['ienc']}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        f"vae_pretrained_path={e['w22']}/Wan2.2_VAE.pth",
        f"tokenizer_path={e['tok']}",
    ]


def compose_cfg(e=None):
    e = e or env()
    with initialize_config_dir(config_dir=CFG_DIR, version_base=None):
        cfg = compose(config_name="conf", overrides=overrides(e))
    return cfg, e


def main():
    cfg, e = compose_cfg()
    ah = cfg.action_head_cfg.config
    spec = OmegaConf.to_container(cfg.train_dataset.mixture_spec, resolve=True)
    paths = {tag: p for el in spec for tag, p in el["dataset_path"].items()}
    want_robot = [f"{e['data']}/{s}" for s in ROBOT_SUBSETS]
    want_human = [f"{e['data']}/{s}" for s in HUMAN_SUBSETS]
    checks = [
        ("global_batch_size", cfg.global_batch_size, e["gb"]),
        ("per_device_train_batch_size", cfg.training_args.per_device_train_batch_size, e["pd"]),
        ("max_steps", cfg.training_args.max_steps, e["ms"]),
        ("save_steps", cfg.training_args.save_steps, e["ss"]),
        ("save_total_limit", cfg.training_args.save_total_limit, e["stl"]),
        ("training_args.learning_rate", cfg.training_args.learning_rate, float(e["lr"])),
        ("training_args.warmup_ratio", cfg.training_args.warmup_ratio, float(e["warmup"])),
        ("weight_decay", cfg.training_args.weight_decay, float(e["wd"])),
        ("lr_scheduler_type", cfg.training_args.lr_scheduler_type, "cosine"),
        ("adam_beta1/2", (cfg.training_args.adam_beta1, cfg.training_args.adam_beta2), (0.95, 0.999)),
        ("bf16", cfg.training_args.bf16, True),
        ("seed", cfg.training_args.seed, 42),
        ("deepspeed", cfg.training_args.deepspeed, f"groot/vla/configs/deepspeed/{e['dsc']}.json"),
        ("dataloader_num_workers", cfg.training_args.dataloader_num_workers, e["nw"]),
        ("num_frames", cfg.num_frames, e["nf"]),
        ("num_frames == 8*chunk+1", cfg.num_frames, 8 * cfg.max_chunk_size + 1),
        ("action_horizon", cfg.action_horizon, 24),
        ("num_views", cfg.num_views, 2),
        ("max_chunk_size", cfg.max_chunk_size, e["mcs"]),
        ("train_architecture", ah.train_architecture, "full"),
        ("image_resolution (W,H) per view", (cfg.image_resolution_width, cfg.image_resolution_height), (192, 144)),
        ("frame_seqlen", ah.diffusion_model_cfg.frame_seqlen, 54),
        ("frame_seqlen == (2H/16/2)*(W/16/2)", ah.diffusion_model_cfg.frame_seqlen,
         (2 * cfg.image_resolution_height // 32) * (cfg.image_resolution_width // 32)),
        ("target_video (H,W)", (ah.target_video_height, ah.target_video_width), (288, 192)),
        ("dit.model_type", ah.diffusion_model_cfg.model_type, "ti2v"),
        ("dit.dim/layers/in", (ah.diffusion_model_cfg.dim, ah.diffusion_model_cfg.num_layers, ah.diffusion_model_cfg.in_dim), (3072, 30, 48)),
        ("dit.num_frame_per_block", ah.diffusion_model_cfg.num_frame_per_block, 2),
        ("dit.num_action_per_block", ah.diffusion_model_cfg.num_action_per_block, 24),
        ("dit.num_state_per_block", ah.diffusion_model_cfg.num_state_per_block, 1),
        ("vae._target_", ah.vae_cfg._target_, "groot.vla.model.dreamzero.modules.wan_video_vae.WanVideoVAE38"),
        ("max_state_dim/max_action_dim", (cfg.max_state_dim, cfg.max_action_dim), (64, 32)),
        ("relative_action", cfg.relative_action, True),
        ("relative_action_keys", list(cfg.relative_action_keys), ["left_arm_joints", "right_arm_joints"]),
        ("black_missing_video_keys", cfg.train_dataset.dataset_kwargs.black_missing_video_keys, True),
        ("video keys", list(cfg.modality_configs.openarm.video.modality_keys), ["video.camera_ego_left", "video.camera_ego_right"]),
        ("state keys", list(cfg.modality_configs.openarm.state.modality_keys),
         ["state.neck_joints", "state.left_arm_joints", "state.right_arm_joints", "state.left_hand_joints", "state.right_hand_joints"]),
        ("action delta_indices", list(cfg.modality_configs.openarm.action.delta_indices), list(range(24))),
        ("language key", list(cfg.modality_configs.openarm.language.modality_keys), ["annotation.human.task_description"]),
        ("embodiment idx openarm/openarm_human", (cfg.embodiment_tag_to_projector_index.openarm, cfg.embodiment_tag_to_projector_index.openarm_human), (33, 34)),
        ("mixture robot paths", paths.get("openarm"), want_robot),
        ("mixture human paths", paths.get("openarm_human"), want_human if e["variant"] == "robothuman" else None),
        ("mixture weights (per element)", [el["dataset_weight"] for el in spec], [1.0] * (2 if e["variant"] == "robothuman" else 1)),
        ("distribute_weights", [el["distribute_weights"] for el in spec], [True] * len(spec)),
        ("balance_dataset_weights", cfg.train_dataset.mixture_kwargs.balance_dataset_weights, False),
        ("tokenizer_path", cfg.tokenizer_path, e["tok"]),
        ("output_dir", cfg.training_args.output_dir, e["out"]),
        ("dit_path", ah.diffusion_model_cfg.diffusion_model_pretrained_path, e["w22"]),
    ]
    bad = 0
    for name, got, want in checks:
        ok = got == want
        bad += (not ok)
        print(("  OK  " if ok else "  FAIL") + f" {name} = {got!r}" + ("" if ok else f"  (want {want!r})"))
    per_step = cfg.training_args.per_device_train_batch_size * e["ng"]
    assert cfg.global_batch_size % per_step == 0, (cfg.global_batch_size, per_step)
    ga = cfg.global_batch_size // per_step
    print(f"\n  grad_accum = {cfg.global_batch_size} // (pd{cfg.training_args.per_device_train_batch_size} * {e['ng']} gpu) = {ga}")
    print(f"  per sample: {cfg.num_frames} frames x canvas {2*cfg.image_resolution_height}x{cfg.image_resolution_width}, "
          f"{cfg.max_chunk_size} chunks x 24 = {24*cfg.max_chunk_size} actions (28 real dims, padded to {cfg.max_action_dim}), "
          f"{cfg.max_chunk_size} states (padded to {cfg.max_state_dim})")
    print(f"  warmup steps = {int(cfg.training_args.warmup_ratio * cfg.training_args.max_steps)} of {cfg.training_args.max_steps} (cosine to 0)")
    print("\nFAILURES:", bad)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
