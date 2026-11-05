"""Focused checks for UWSAM feature distillation and mask routing."""

from types import SimpleNamespace

import torch
from torch import nn

from uwsam.components import EUPGMaskHead
from uwsam.detector import UWSAM
from uwsam.distillation import MGUKDDistiller
from uwsam.graph import build_knn_edges


def test_knn_edges_match_pairwise_cosine():
    torch.manual_seed(7)
    features = torch.randn(2, 9, 4)
    k = 3
    similarities = torch.cosine_similarity(
        features.unsqueeze(2), features.unsqueeze(1), dim=3
    )
    neighbors = similarities.topk(k, dim=2).indices
    source = torch.arange(9).repeat_interleave(k).view(9, k).repeat(2, 1, 1)
    expected = torch.stack((source, neighbors), dim=3).view(2, -1, 2).transpose(1, 2)

    assert torch.equal(build_knn_edges(features, k=k), expected)


def test_distillation_losses_backpropagate_to_student_and_gat():
    torch.manual_seed(11)
    distiller = MGUKDDistiller(
        nums=2,
        connector=dict(
            type="MaskGATConnector",
            student_channels=4,
            teacher_channels=4,
            gat_config=dict(num_layers=2, heads=1),
            top_k=3,
            mask_ratio=0.5,
            loss=dict(type="FeatureReconstructionLoss", loss_weight=0.01),
        ),
    )
    student = tuple(torch.randn(2, 4, 3, 3, requires_grad=True) for _ in range(2))
    teacher = tuple(torch.randn(2, 4, 3, 3) for _ in range(2))

    losses = distiller(student, teacher)
    assert set(losses) == {"loss_connector_0", "loss_connector_1"}
    sum(losses.values()).backward()
    assert all(
        feature.grad is not None and feature.grad.abs().sum() > 0 for feature in student
    )
    gat_parameters = distiller.connectors[0].generation.gat.parameters()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in gat_parameters)


def test_frozen_teacher_stays_in_eval_mode():
    # The train method only needs registered modules; model construction is
    # covered by the full forward tests.
    model = UWSAM.__new__(UWSAM)
    nn.Module.__init__(model)
    model.teacher = nn.Sequential(nn.BatchNorm2d(2))
    model.backbone = nn.Linear(2, 2)
    model.frozen_modules = [model.teacher]

    model.train()
    assert model.training
    assert model.backbone.training
    assert not model.teacher.training
    assert not model.teacher[0].training


def test_mask_head_routes_roi_features_to_source_images():
    class RecordingDecoder(nn.Module):
        def forward(self, **kwargs):
            self.image_embeddings = kwargs["image_embeddings"].detach().clone()
            count = self.image_embeddings.shape[0]
            return torch.ones(count, 1, 2, 2), torch.zeros(count, 1), None

    head = EUPGMaskHead.__new__(EUPGMaskHead)
    nn.Module.__init__(head)
    head.per_pointset_point = 1
    head.with_sincos = False
    head.multimask_output = False
    head.attention_similarity = None
    head.target_embedding = None
    head.output_attentions = None
    head.point_emb = nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 4)
    )
    head.no_mask_embed = nn.Embedding(1, 4)
    head.mask_decoder = RecordingDecoder()

    roi_features = torch.randn(3, 4, 2, 2)
    images = torch.arange(3, dtype=torch.float).view(3, 1, 1, 1).expand(3, 4, 2, 2)
    positions = torch.zeros_like(images)
    masks, iou = head(
        roi_features,
        image_embeddings=images,
        image_positional_embeddings=positions,
        roi_img_ids=torch.tensor([0, 0, 2]),
    )
    assert masks.shape == (3, 1, 2, 2)
    assert iou.shape == (3, 1)
    assert torch.equal(
        head.mask_decoder.image_embeddings[:, 0, 0, 0], torch.tensor([0.0, 0.0, 2.0])
    )


def test_mask_prediction_honors_rescale_flag():
    head = EUPGMaskHead.__new__(EUPGMaskHead)
    nn.Module.__init__(head)
    mask_logits = torch.ones(1, 1, 2, 2)
    boxes = torch.tensor([[0.0, 0.0, 3.0, 4.0]])
    meta = dict(
        scale_factor=(0.5, 0.5),
        ori_shape=(8, 6),
        img_shape=(4, 3),
        batch_input_shape=(4, 4),
    )
    config = SimpleNamespace(mask_thr_binary=0.5)

    scaled_masks = head._predict_by_feat_single(
        mask_logits, boxes.clone(), torch.tensor([0]), meta, config, rescale=False
    )
    original_boxes = boxes.clone()
    original_masks = head._predict_by_feat_single(
        mask_logits, original_boxes, torch.tensor([0]), meta, config, rescale=True
    )

    assert scaled_masks.shape == (1, 4, 3)
    assert original_masks.shape == (1, 8, 6)
    assert torch.equal(original_boxes, boxes * 2)
