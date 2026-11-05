"""Experimental Mask R-CNN with the SAM2 Hiera-L image trunk as backbone."""

_base_ = [
    "mmdet::_base_/models/mask-rcnn_r50_fpn.py",
    "../_base_/uiis10k.py",
    "mmdet::_base_/schedules/schedule_2x.py",
    "mmdet::_base_/default_runtime.py",
]

custom_imports = dict(imports=["uwsam.extensions.sam2"], allow_failed_imports=False)

crop_size = (512, 512)
sam2_checkpoint = "pretrain/sam2_hiera_large.pt"

model = dict(
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        batch_augments=[dict(type="BatchFixedSizePad", size=crop_size, pad_mask=True)],
    ),
    backbone=dict(
        _delete_=True,
        type="SAM2Backbone",
        checkpoint=sam2_checkpoint,
        embed_dim=144,
        num_heads=2,
        stages=(2, 6, 36, 4),
        global_att_blocks=(23, 33, 43),
        window_pos_embed_bkg_spatial_size=(7, 7),
        window_spec=(8, 4, 16, 8),
        peft_config=dict(
            peft_type="LORA",
            r=16,
            target_modules=["qkv"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
        ),
    ),
    neck=dict(in_channels=[144, 288, 576, 1152], out_channels=256, num_outs=5),
    roi_head=dict(
        bbox_head=dict(num_classes=10),
        mask_head=dict(num_classes=10),
    ),
)
