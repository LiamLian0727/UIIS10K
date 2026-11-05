_base_ = [
    "./_base_/model.py",
    "./_base_/uiis10k.py",
    "./_base_/runtime.py",
]

custom_imports = dict(imports=["uwsam"], allow_failed_imports=False)

default_scope = "mmdet"
work_dir = "work_dirs/uwsam_teacher"

## ---------------------- MODEL ----------------------

crop_size = (512, 512)
num_classes = 10

batch_augments = [
    dict(
        type="BatchFixedSizePad",
        size=crop_size,
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=False,
    )
]

data_preprocessor = dict(
    type="DetDataPreprocessor",
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    bgr_to_rgb=True,
    pad_mask=True,
    pad_size_divisor=32,
    batch_augments=batch_augments,
)

norm_cfg = dict(type="LN2d", requires_grad=True)

# sam base model
sam_pretrain_name = "pretrain/sam-vit-huge"
sam_pretrain_ckpt_path = "pretrain/sam-vit-huge/pytorch_model.bin"

model = dict(
    data_preprocessor=data_preprocessor,
    decoder_freeze=False,
    shared_image_embedding=dict(extra_config=dict(image_size=crop_size[0])),
    backbone=dict(
        _delete_=True,
        img_size=crop_size[0],
        type="SAMImageEncoder",
        hf_pretrain_name=sam_pretrain_name,
        init_cfg=dict(type="Pretrained", checkpoint=sam_pretrain_ckpt_path),
        peft_config=dict(
            peft_type="LORA",
            r=16,
            target_modules=["qkv"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
        ),
    ),
    neck=dict(
        feature_aggregator=dict(
            _delete_=True,
            type="ChannelAttentionAggregator",
            in_channels=256,
            hidden_channels=64,
            out_channels=256,
        ),
        feature_spliter=dict(
            _delete_=True,
            type="SimpleFPN",
            backbone_channel=256,
            in_channels=[64, 128, 256, 256],
            out_channels=256,
            num_outs=5,
            norm_cfg=norm_cfg,
        ),
    ),
    roi_head=dict(
        bbox_head=dict(
            type="Shared4Conv1FCBBoxHead",
            conv_out_channels=256,
            norm_cfg=norm_cfg,
            num_classes=10,
        ),
    ),
    train_cfg=dict(rcnn=dict(mask_size=crop_size)),
)

## ---------------------- Optim ----------------------

max_epochs = 24
base_lr = 0.0002

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=3)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

find_unused_parameters = True

# learning rate
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=50),
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[15, 21],
        gamma=0.1,
    ),
]

optim_wrapper = dict(
    type="AmpOptimWrapper",
    dtype="float16",
    optimizer=dict(
        type="AdamW",
        lr=base_lr,
        weight_decay=0.05,
    ),
)
