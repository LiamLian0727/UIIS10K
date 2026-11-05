"""UWSAM teacher and student detector."""

import copy
from typing import Tuple

import torch
from mmdet.models import MaskRCNN
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from torch import Tensor


@MODELS.register_module()
class UWSAM(MaskRCNN):
    """Mask R-CNN with SAM image features and EUPG mask prompting.

    During student training, a frozen teacher supplies feature maps to MG-UKD.
    The module attributes retain their checkpoint key paths.
    """

    def __init__(
        self,
        shared_image_embedding,
        teacher=None,
        distiller=None,
        decoder_freeze=True,
        encoder_freeze=True,
        *args,
        **kwargs,
    ):
        peft_config = kwargs.get("backbone", {}).get("peft_config")
        super().__init__(*args, **kwargs)
        self.shared_image_embedding = MODELS.build(shared_image_embedding)
        self.frozen_modules = []
        self.teacher = None
        if teacher is not None:
            self.teacher = MODELS.build(teacher).eval()
            self.frozen_modules += [self.teacher]
            assert distiller is not None, "distiller is required when teacher is set"
            self.distiller = MODELS.build(distiller)

        self.decoder_freeze = decoder_freeze

        if (peft_config is None) and encoder_freeze:
            self.frozen_modules += [self.backbone]
        if self.decoder_freeze:
            self.frozen_modules += [
                self.shared_image_embedding,
                self.roi_head.mask_head.mask_decoder,
                self.roi_head.mask_head.no_mask_embed,
            ]
        self._freeze_modules(self.frozen_modules)

    def _freeze_modules(self, module_list):
        for module in module_list:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        for module in self.frozen_modules:
            module.eval()
        return self

    def _image_positional_embeddings(self, size):
        target_device = self.shared_image_embedding.shared_image_embedding.positional_embedding.device
        target_dtype = self.shared_image_embedding.shared_image_embedding.positional_embedding.dtype
        grid = torch.ones((size, size), device=target_device, dtype=target_dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size
        x_embed = x_embed / size

        positional_embedding = self.shared_image_embedding(
            torch.stack([x_embed, y_embed], dim=-1)
        )
        return positional_embedding.permute(2, 0, 1).unsqueeze(0)

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        vision_outputs = self.backbone(batch_inputs)
        if not isinstance(vision_outputs, tuple):
            raise TypeError("Image encoder must return a tuple of feature maps")
        image_embeddings = vision_outputs[-1]
        vision_hidden_states = vision_outputs

        image_positional_embeddings = self._image_positional_embeddings(
            size=image_embeddings.shape[-1]
        )
        # repeat with batch size
        batch_size = image_embeddings.shape[0]
        image_positional_embeddings = image_positional_embeddings.repeat(
            batch_size, 1, 1, 1
        )
        x = self.neck(vision_hidden_states)
        return x, image_embeddings, image_positional_embeddings, vision_hidden_states

    def _forward(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> tuple:
        """Return raw detection and mask tensors through the EUPG interface."""
        features, image_embeddings, positions, _ = self.extract_feat(batch_inputs)
        if batch_data_samples[0].get("proposals", None) is None:
            proposals = self.rpn_head.predict(
                features, batch_data_samples, rescale=False
            )
        else:
            proposals = [sample.proposals for sample in batch_data_samples]
        return self.roi_head(
            features,
            proposals,
            batch_data_samples,
            image_embeddings=image_embeddings,
            image_positional_embeddings=positions,
        )

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> dict:
        x, image_embeddings, image_positional_embeddings, vision_hidden_states = (
            self.extract_feat(batch_inputs)
        )

        losses = dict()
        if self.teacher is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher(batch_inputs)
                if not isinstance(teacher_outputs, tuple):
                    raise TypeError(
                        "Teacher backbone must return a tuple of feature maps"
                    )
            kd_loss = self.distiller(vision_hidden_states, teacher_outputs)
            losses.update(kd_loss)

        # RPN forward and loss
        proposal_cfg = self.train_cfg.get("rpn_proposal", self.test_cfg.rpn)
        rpn_data_samples = copy.deepcopy(batch_data_samples)
        # set cat_id of gt_labels to 0 in RPN
        for data_sample in rpn_data_samples:
            data_sample.gt_instances.labels = torch.zeros_like(
                data_sample.gt_instances.labels
            )

        rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
            x, rpn_data_samples, proposal_cfg=proposal_cfg
        )
        # avoid get same name with roi_head loss
        keys = rpn_losses.keys()
        for key in list(keys):
            if "loss" in key and "rpn" not in key:
                rpn_losses[f"rpn_{key}"] = rpn_losses.pop(key)
        losses.update(rpn_losses)

        roi_losses = self.roi_head.loss(
            x,
            rpn_results_list,
            batch_data_samples,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )
        losses.update(roi_losses)

        return losses

    def predict(
        self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True
    ) -> SampleList:
        x, image_embeddings, image_positional_embeddings, _ = self.extract_feat(
            batch_inputs
        )

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get("proposals", None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False
            )
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x,
            rpn_results_list,
            batch_data_samples,
            rescale=rescale,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list
        )
        return batch_data_samples
