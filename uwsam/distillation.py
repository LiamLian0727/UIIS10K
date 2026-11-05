from typing import Dict, Optional

import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.model import BaseModule

from .graph import ImageGraphAttention, build_knn_edges


@MODELS.register_module()
class MGUKDDistiller(BaseModule):
    """Apply one mask graph connector to each selected feature level."""

    def __init__(
        self,
        connector: Dict,
        nums=1,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)
        self.connectors = nn.ModuleList([MODELS.build(connector) for _ in range(nums)])

    def forward(self, student: tuple, teacher: tuple) -> dict:
        loss = dict()
        assert len(student) == len(teacher) == len(self.connectors), (
            "The number of connectors should be the same as the number of student and teacher."
        )
        for i, connector in enumerate(self.connectors):
            loss[f"loss_connector_{i}"] = connector(student[i], teacher[i])
        return loss


@MODELS.register_module()
class MaskGATConnector(BaseModule):
    """Reconstruct teacher features from masked student tokens using GAT.

    Args:
        student_channels(int): Number of channels in the student's feature map.
        teacher_channels(int): Number of channels in the teacher's feature map.
        gat_config (Dict, optional): The config for the GAT model, including the following keys:
            - in_channels(int): Number of input channels.
            - hidden_channels(int): Number of hidden channels.
            - num_layers(int): Number of GAT layers.
            - out_channels(int): Number of output channels.
            - v2(bool): Whether to use the GATv2 model. Defaults to False.
        top_k: number of neighbors in GAT, the nearest k nodes will be connected
        mask_ratio (float, optional): masked ratio. Defaults to 0.65
        mask_on_channel (bool, optional): Whether to mask on channel or mask on token. Defaults to False.
        init_cfg (Optional[Dict], optional): The weight initialized config for
            :class:`BaseModule`. Defaults to None.
    """

    def __init__(
        self,
        student_channels: int,
        teacher_channels: int,
        gat_config: Dict = None,
        top_k: int = 7,
        mask_ratio: float = 0.65,
        mask_on_channel: bool = False,
        loss: Dict = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)

        self.top_k = top_k
        self.mask_ratio = mask_ratio
        self.mask_on_channel = mask_on_channel
        self.loss = MODELS.build(
            loss if loss is not None else dict(type="FeatureReconstructionLoss")
        )

        # align the student and teacher feature dimensions
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(
                student_channels, teacher_channels, kernel_size=1, stride=1, padding=0
            )
        else:
            self.align = None

        default_gat_config = dict(
            in_channels=teacher_channels,
            hidden_channels=teacher_channels,
            num_layers=2,
            out_channels=teacher_channels,
            v2=False,
        )
        if gat_config is not None:
            gat_config = gat_config.copy()
            gat_config.pop("type", None)
            default_gat_config.update(gat_config)

        self.conv = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.generation = ImageGraphAttention(**default_gat_config)

    def forward(self, student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
        """Forward computation.

        Args:
            student (torch.Tensor): student feature map, shape (B, C, H, W).
            teacher (torch.Tensor): teacher feature map, shape (B, C, H, W).
        """
        return self.forward_train(student, teacher)

    def forward_train(
        self, feature_s: torch.Tensor, feature_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Train computation of MaskGATConnector.
        Args:
            feature (torch.Tensor): The input feature map, shape (B, C, H, W).
        """
        if self.align is not None:
            feature_s = self.align(feature_s)

        B, C, H, W = feature_s.shape
        # B, C, H, W -> B, C, HW -> B, HW, C
        embedding = feature_s.view(B, C, -1).permute(0, 2, 1)
        edge_index = build_knn_edges(embedding, k=self.top_k)

        if not self.mask_on_channel:
            mat = torch.rand((B, 1, H, W), device=feature_s.device)
        else:
            mat = torch.rand((B, C, 1, 1), device=feature_s.device)

        mat = torch.where(mat > 1 - self.mask_ratio, 0.0, 1.0)

        masked_embedding = torch.mul(feature_s, mat).view(B, C, -1).permute(0, 2, 1)
        new_fea = self.generation(masked_embedding, edge_index)
        # B, HW, C -> B, C, HW -> B, C, H, W
        new_fea = new_fea.permute(0, 2, 1).view(B, C, H, W).contiguous()
        kd_loss = self.loss(new_fea, feature_t)
        return kd_loss


@MODELS.register_module()
class FeatureReconstructionLoss(BaseModule):
    """Sum squared feature reconstruction error, normalized by batch size."""

    def __init__(self, loss_weight: float = 0.0002) -> None:
        super().__init__()
        self.alpha_mgd = loss_weight
        self.loss_mse = nn.MSELoss(reduction="sum")

    def forward(
        self, student_features: torch.Tensor, teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward function.

        Args:
            student_features(torch.Tensor): Bs*C*H*W, student's feature map
            teacher_features(torch.Tensor): Bs*C*H*W, teacher's feature map

        Return:
            torch.Tensor: The calculated loss value.
        """
        assert student_features.shape == teacher_features.shape
        loss = (
            self._reconstruction_error(student_features, teacher_features)
            * self.alpha_mgd
        )

        return loss

    def _reconstruction_error(
        self, student_features: torch.Tensor, teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """Get MSE distance of student_features and teacher_features.

        Args:
            student_features(torch.Tensor): Bs*C*H*W, student's feature map
            teacher_features(torch.Tensor): Bs*C*H*W, teacher's feature map

        Return:
            torch.Tensor: The calculated mse distance value.
        """
        N, C, H, W = teacher_features.shape
        dis_loss = self.loss_mse(student_features, teacher_features) / N

        return dis_loss
