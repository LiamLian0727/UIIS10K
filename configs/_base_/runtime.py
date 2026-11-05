_base_ = ["mmdet::_base_/default_runtime.py"]

default_hooks = dict(
    logger=dict(interval=20),
    checkpoint=dict(
        interval=3,
        max_keep_ckpts=1,
        save_best=["coco/bbox_mAP", "coco/segm_mAP"],
        rule="greater",
        save_last=True,
    ),
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
