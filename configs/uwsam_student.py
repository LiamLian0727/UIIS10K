_base_ = [
    "./_base_/model.py",
    "./_base_/uiis10k.py",
    "./_base_/runtime.py",
]

custom_imports = dict(imports=["uwsam"], allow_failed_imports=False)

default_scope = "mmdet"
work_dir = "work_dirs/uwsam_student"

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

model = dict(
    init_cfg=dict(type="Pretrained", checkpoint="pretrain/uwsam_teacher.pth"),
    data_preprocessor=data_preprocessor,
    decoder_freeze=False,
    encoder_freeze=False,
    shared_image_embedding=dict(
        extra_config=dict(image_size=crop_size[0]),
        init_cfg=None,
    ),
    backbone=dict(
        _delete_=True,
        type="EfficientSAMEncoder",
        patch_embed_dim=384,
        img_size=crop_size,
        num_heads=6,
        out_indices=[2, 5, 8, 11],
        neck_dims=[256, 256],
        init_cfg=dict(
            type="Pretrained",
            prefix="image_encoder.",
            checkpoint=r"pretrain/efficient_sam_vits.pt",
        ),
    ),
    teacher=dict(
        img_size=crop_size[0],
        type="SAMImageEncoder",
        hf_pretrain_name="pretrain/sam-vit-huge",
        init_cfg=None,
        out_indices=[2, 5, 8, 11],
        peft_config=dict(
            peft_type="LORA",
            r=16,
            target_modules=["qkv"],
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
        ),
    ),
    distiller=dict(
        type="MGUKDDistiller",
        nums=4,
        connector=dict(
            type="MaskGATConnector",
            student_channels=256,
            teacher_channels=256,
            gat_config=dict(num_layers=2, v2=False, heads=4),
            top_k=11,
            mask_ratio=0.65,
            mask_on_channel=False,
            loss=dict(type="FeatureReconstructionLoss", loss_weight=2e-5),
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
            reg_class_agnostic=False,
            conv_out_channels=256,
            norm_cfg=norm_cfg,
            num_classes=num_classes,
        ),
        mask_head=dict(mask_decoder=dict(init_cfg=None)),
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
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=250),
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
    constructor="LayerDecayOptimizerConstructor",
    paramwise_cfg={
        "decay_rate": 0.7,
        "decay_type": "layer_wise",
        "num_layers": 12,
    },
    optimizer=dict(
        type="AdamW",
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    clip_grad=dict(max_norm=10, norm_type=2),
)
