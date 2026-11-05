"""Optional Mask R-CNN backbone interface for experimentation."""

## ---------------------- DEFAULT_SETTING ----------------------
_base_ = [
    "mmdet::_base_/models/mask-rcnn_r50_fpn.py",
    "../_base_/uiis10k.py",
    "mmdet::_base_/schedules/schedule_2x.py",
    "mmdet::_base_/default_runtime.py",
]

default_scope = "mmdet"
custom_imports = dict(imports=["uwsam"], allow_failed_imports=False)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        max_keep_ckpts=1,
        save_best=["coco/segm_mAP"],
        rule="greater",
        save_last=True,
    ),
)
## ---------------------- MODEL_SETTING -------------------------


batch_augments = [dict(type="BatchFixedSizePad", size=(512, 512), pad_mask=True)]
norm_cfg = dict(type="LN2d", requires_grad=True)
model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
    backbone=dict(
        _delete_=True,
        type="EfficientSAMEncoder",
        img_size=512,
        patch_embed_dim=384,
        num_heads=6,
        out_indices=-1,
        neck_dims=[256, 256],
        init_cfg=dict(
            type="Pretrained",
            prefix="image_encoder.",
            checkpoint=r"pretrain/efficient_sam_vits.pt",
        ),
    ),
    neck=dict(
        _delete_=True,
        type="SimpleFPN",
        backbone_channel=256,
        in_channels=[64, 128, 256, 256],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg,
    ),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=dict(
            type="Shared4Conv1FCBBoxHead",
            conv_out_channels=256,
            norm_cfg=norm_cfg,
            num_classes=10,
        ),
        mask_head=dict(norm_cfg=norm_cfg, num_classes=10),
    ),
)

## ---------------------- OPTIM_SETTING -------------------------

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=24, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(type="LinearLR", start_factor=0.067, by_epoch=False, begin=0, end=500),
    dict(
        type="MultiStepLR",
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[22, 24],
        gamma=0.1,
    ),
]

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.00004),
    # clip_grad=dict(max_norm=10, norm_type=2)
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=64)
