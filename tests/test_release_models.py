"""Exercise the release configs and a small real UWSAM on CPU, without downloads."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmdet.structures.mask import BitmapMasks
from mmdet.utils import register_all_modules
from mmengine.config import Config
from mmengine.structures import InstanceData
from transformers import SamConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("config_path", sorted((ROOT / "configs").rglob("*.py")))
def test_config_inheritance_and_custom_imports(config_path):
    cfg = Config.fromfile(str(config_path))
    if "custom_imports" in cfg and "model" in cfg and "backbone" in cfg.model:
        assert MODELS.get(cfg.model.type) is not None
        assert MODELS.get(cfg.model.backbone.type) is not None


@pytest.fixture(scope="module")
def tiny_student_config(tmp_path_factory):
    register_all_modules(init_default_scope=True)
    config_dir = tmp_path_factory.mktemp("sam-vit-huge")
    sam = SamConfig(
        vision_config=dict(
            image_size=64,
            hidden_size=24,
            num_hidden_layers=4,
            num_attention_heads=3,
            mlp_dim=48,
            output_channels=256,
        ),
        prompt_encoder_config=dict(image_size=64, image_embedding_size=4),
    )
    sam.save_pretrained(config_dir)
    cfg = Config.fromfile(str(ROOT / "configs/uwsam_student_inference.py"))
    model_cfg = cfg.model
    model_cfg.backbone.update(
        img_size=64, patch_embed_dim=24, num_heads=3, depth=4, out_indices=[0, 1, 2, 3]
    )
    model_cfg.shared_image_embedding.hf_pretrain_name = str(config_dir)
    model_cfg.roi_head.mask_head.mask_decoder.hf_pretrain_name = str(config_dir)
    model_cfg.data_preprocessor.batch_augments[0].size = (64, 64)
    model_cfg.train_cfg.rcnn.mask_size = (64, 64)
    model_cfg.train_cfg.rpn.sampler.num = 32
    model_cfg.train_cfg.rcnn.sampler.num = 16
    model_cfg.train_cfg.rpn_proposal.update(nms_pre=32, max_per_img=16)
    model_cfg.test_cfg.rpn.update(nms_pre=32, max_per_img=16)
    model_cfg.test_cfg.rcnn.update(max_per_img=4, score_thr=0.0)
    return model_cfg


@pytest.fixture(scope="module")
def tiny_student(tiny_student_config):
    model = MODELS.build(deepcopy(tiny_student_config))
    model.init_weights()
    return model


def make_sample(empty=False):
    sample = DetDataSample(
        metainfo=dict(
            img_shape=(64, 64),
            ori_shape=(64, 64),
            pad_shape=(64, 64),
            batch_input_shape=(64, 64),
            scale_factor=(1.0, 1.0),
            img_id=0,
        )
    )
    mask = np.zeros((0 if empty else 1, 64, 64), dtype=np.uint8)
    if not empty:
        mask[:, 16:48, 16:48] = 1
    sample.gt_instances = InstanceData(
        bboxes=torch.empty((0, 4))
        if empty
        else torch.tensor([[16.0, 16.0, 48.0, 48.0]]),
        labels=torch.empty(0, dtype=torch.long) if empty else torch.tensor([0]),
        masks=BitmapMasks(mask, 64, 64),
    )
    return sample


@pytest.mark.parametrize("empty", [False, True])
def test_student_training_loss_and_backward(tiny_student, empty):
    model = tiny_student.train()
    model.zero_grad(set_to_none=True)
    losses = model(torch.randn(1, 3, 64, 64), [make_sample(empty)], mode="loss")
    total, _ = model.parse_losses(losses)
    assert torch.isfinite(total)
    total.backward()
    assert model.backbone.patch_embed.proj.weight.grad is not None


def test_student_predicts_a_batch_without_teacher(tiny_student):
    model = tiny_student.eval()
    with torch.no_grad():
        results = model(
            torch.randn(2, 3, 64, 64), [make_sample(), make_sample()], mode="predict"
        )
    assert model.teacher is None
    assert len(results) == 2
    assert all(result.pred_instances.masks.shape[-2:] == (64, 64) for result in results)


def test_student_initialization_and_distillation_flow(
    tiny_student, tiny_student_config, tmp_path
):
    from tools.convert_teacher_checkpoint import convert_state_dict

    backbone_state = {
        f"image_encoder.{key}": value.clone()
        for key, value in tiny_student.backbone.state_dict().items()
    }
    backbone_state["image_encoder.patch_embed.proj.weight"].fill_(0.125)
    backbone_file = tmp_path / "efficient_sam.pt"
    torch.save({"state_dict": backbone_state}, backbone_file)

    teacher_state = {
        key: value.clone() for key, value in tiny_student.state_dict().items()
    }
    teacher_state["backbone.patch_embed.proj.weight"].fill_(0.25)
    teacher_file = tmp_path / "teacher.pth"
    torch.save({"state_dict": convert_state_dict(teacher_state)}, teacher_file)

    config = deepcopy(tiny_student_config)
    config.init_cfg = dict(type="Pretrained", checkpoint=str(teacher_file))
    config.backbone.init_cfg = dict(
        type="Pretrained", prefix="image_encoder.", checkpoint=str(backbone_file)
    )
    config.teacher = deepcopy(config.backbone)
    config.teacher.init_cfg = None
    config.distiller = dict(
        type="MGUKDDistiller",
        nums=4,
        connector=dict(
            type="MaskGATConnector",
            student_channels=256,
            teacher_channels=256,
            gat_config=dict(num_layers=2, heads=4),
            top_k=3,
            mask_ratio=0.65,
            loss=dict(type="FeatureReconstructionLoss", loss_weight=2e-5),
        ),
    )
    model = MODELS.build(config)
    model.init_weights()
    assert torch.all(model.backbone.patch_embed.proj.weight == 0.125)
    assert torch.all(model.teacher.patch_embed.proj.weight == 0.25)
    model.train()
    losses = model(torch.randn(1, 3, 64, 64), [make_sample()], mode="loss")
    assert all(f"loss_connector_{index}" in losses for index in range(4))
    total, _ = model.parse_losses(losses)
    assert torch.isfinite(total)
    total.backward()
    assert (
        model.distiller.connectors[0].generation.gat.convs[0].lin.weight.grad
        is not None
    )
    assert not model.teacher.training
    assert all(parameter.grad is None for parameter in model.teacher.parameters())


def test_student_tensor_mode_uses_eupg_features(tiny_student):
    with torch.no_grad():
        outputs = tiny_student.eval()(
            torch.randn(1, 3, 64, 64), [make_sample()], mode="tensor"
        )
    assert len(outputs) == 3
    assert outputs[0].ndim == 2
    assert outputs[2].ndim == 4
