"""Student evaluation without the training-only teacher and graph distiller."""

_base_ = ["./uwsam_student.py"]
model = dict(
    init_cfg=None,
    teacher=None,
    distiller=None,
    backbone=dict(init_cfg=None),
    shared_image_embedding=dict(init_cfg=None),
    roi_head=dict(mask_head=dict(mask_decoder=dict(init_cfg=None))),
)
