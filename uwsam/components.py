"""EUPG prompting and SAM decoder components."""

import copy
from typing import List, Optional, Tuple

import einops
import torch
from mmcv.cnn import build_activation_layer
from mmdet.models import FCNMaskHead, SinePositionalEncoding, StandardRoIHead
from mmdet.models.task_modules import SamplingResult
from mmdet.models.utils import empty_instances, unpack_gt_instances
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from mmengine import ConfigDict
from mmengine.model import BaseModule
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import SamConfig
from transformers.models.sam.modeling_sam import (
    SamMaskDecoder,
    SamPositionalEmbedding,
    SamPromptEncoder,
)


@MODELS.register_module()
class EUPGNeck(BaseModule):
    """Aggregate encoder features and build the detection feature pyramid."""

    def __init__(
        self,
        feature_aggregator=None,
        feature_spliter=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        if feature_aggregator is not None:
            self.feature_aggregator = MODELS.build(feature_aggregator)
        if feature_spliter is not None:
            self.feature_spliter = MODELS.build(feature_spliter)

    def forward(self, inputs):
        if hasattr(self, "feature_aggregator"):
            x = self.feature_aggregator(inputs)
        else:
            x = inputs
        if hasattr(self, "feature_spliter"):
            x = self.feature_spliter(x)
        else:
            x = (x,)
        return x


@MODELS.register_module()
class EUPGRoIHead(StandardRoIHead):
    """Predict bounding boxes and route RoI features to the EUPG mask head."""

    def __init__(self, with_extra_pe=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if with_extra_pe:
            out_channels = self.bbox_roi_extractor.out_channels
            positional_encoding = dict(
                num_feats=out_channels // 2,
                normalize=True,
            )
            self.extra_pe = SinePositionalEncoding(**positional_encoding)

    def forward(
        self,
        x,
        rpn_results_list,
        batch_data_samples=None,
        image_embeddings=None,
        image_positional_embeddings=None,
    ):
        """Expose the same raw forward interface as a standard RoI head."""
        if hasattr(self, "extra_pe"):
            batch_size, _, height, width = x[0].shape
            mask = torch.zeros(
                (batch_size, height, width), device=x[0].device, dtype=torch.bool
            )
            positions = self.extra_pe(mask)
            x = tuple(
                feature
                + F.interpolate(
                    positions,
                    size=feature.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for feature in x
            )
        rois = bbox2roi([result.bboxes for result in rpn_results_list])
        bbox_results = self._bbox_forward(x, rois)
        mask_rois = rois[:100]
        if len(mask_rois):
            masks = self._mask_forward(
                x,
                mask_rois,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )["mask_preds"]
        else:
            height, width = image_embeddings.shape[-2:]
            masks = image_embeddings.new_empty((0, 1, height * 4, width * 4))
        return bbox_results["cls_score"], bbox_results["bbox_pred"], masks

    def _mask_forward(
        self,
        x: Tuple[Tensor],
        rois: Tensor = None,
        pos_inds: Optional[Tensor] = None,
        bbox_feats: Optional[Tensor] = None,
        image_embeddings=None,
        image_positional_embeddings=None,
    ) -> dict:
        assert (rois is not None) ^ (pos_inds is not None and bbox_feats is not None)
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[: self.mask_roi_extractor.num_inputs], rois
            )
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_preds, iou_predictions = self.mask_head(
            mask_feats,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            roi_img_ids=rois[:, 0] if rois is not None else None,
        )
        mask_results = dict(
            mask_preds=mask_preds,
            mask_feats=mask_feats,
            iou_predictions=iou_predictions,
        )
        return mask_results

    def mask_loss(
        self,
        x: Tuple[Tensor],
        sampling_results: List[SamplingResult],
        bbox_feats: Tensor,
        batch_gt_instances: InstanceList,
        image_embeddings=None,
        image_positional_embeddings=None,
    ) -> dict:
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_priors for res in sampling_results])
            if len(pos_rois) == 0:
                return dict(loss_mask=dict(loss_mask=0 * x[0].sum()))
            mask_results = self._mask_forward(
                x,
                pos_rois,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_priors.shape[0], device=device, dtype=torch.uint8
                    )
                )
                pos_inds.append(
                    torch.zeros(
                        res.neg_priors.shape[0], device=device, dtype=torch.uint8
                    )
                )
            pos_inds = torch.cat(pos_inds)

            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats
            )

        mask_loss_and_target = self.mask_head.loss_and_target(
            mask_preds=mask_results["mask_preds"],
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=self.train_cfg,
        )

        mask_results.update(loss_mask=mask_loss_and_target["loss_mask"])
        return mask_results

    def loss(
        self,
        x: Tuple[Tensor],
        rpn_results_list: InstanceList,
        batch_data_samples: List[DetDataSample],
        # extra inputs
        image_embeddings=None,
        image_positional_embeddings=None,
    ) -> dict:

        assert len(rpn_results_list) == len(batch_data_samples)

        outputs = unpack_gt_instances(batch_data_samples)
        batch_gt_instances, batch_gt_instances_ignore, _ = outputs

        if hasattr(self, "extra_pe"):
            bs, _, h, w = x[0].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feats_pe = self.extra_pe(mask_pe)
            outputs = []
            for i in range(len(x)):
                output = x[i] + F.interpolate(
                    img_feats_pe,
                    size=x[i].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                outputs.append(output)
            x = tuple(outputs)

        # assign gts and sample proposals
        num_imgs = len(batch_data_samples)
        sampling_results = []
        for i in range(num_imgs):
            # rename rpn_results.bboxes to rpn_results.priors
            rpn_results = rpn_results_list[i]
            rpn_results.priors = rpn_results.pop("bboxes")

            assign_result = self.bbox_assigner.assign(
                rpn_results, batch_gt_instances[i], batch_gt_instances_ignore[i]
            )
            sampling_result = self.bbox_sampler.sample(
                assign_result,
                rpn_results,
                batch_gt_instances[i],
                feats=[lvl_feat[i][None] for lvl_feat in x],
            )
            sampling_results.append(sampling_result)

        losses = dict()
        # bbox head loss
        if self.with_bbox:
            bbox_results = self.bbox_loss(x, sampling_results)
            losses.update(bbox_results["loss_bbox"])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self.mask_loss(
                x,
                sampling_results,
                bbox_results["bbox_feats"],
                batch_gt_instances,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )
            losses.update(mask_results["loss_mask"])

        return losses

    def predict_mask(
        self,
        x: Tuple[Tensor],
        batch_img_metas: List[dict],
        results_list: InstanceList,
        rescale: bool = False,
        image_embeddings=None,
        image_positional_embeddings=None,
    ) -> InstanceList:

        # don't need to consider aug_test.
        bboxes = [res.bboxes for res in results_list]
        mask_rois = bbox2roi(bboxes)
        if mask_rois.shape[0] == 0:
            results_list = empty_instances(
                batch_img_metas,
                mask_rois.device,
                task_type="mask",
                instance_results=results_list,
                mask_thr_binary=self.test_cfg.mask_thr_binary,
            )
            return results_list

        mask_results = self._mask_forward(
            x,
            mask_rois,
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
        )

        mask_preds = mask_results["mask_preds"]
        # split batch mask prediction back to each image
        num_mask_rois_per_img = [len(res) for res in results_list]
        mask_preds = mask_preds.split(num_mask_rois_per_img, 0)

        results_list = self.mask_head.predict_by_feat(
            mask_preds=mask_preds,
            results_list=results_list,
            batch_img_metas=batch_img_metas,
            rcnn_test_cfg=self.test_cfg,
            rescale=rescale,
        )
        return results_list

    def predict(
        self,
        x: Tuple[Tensor],
        rpn_results_list: InstanceList,
        batch_data_samples: SampleList,
        rescale: bool = False,
        # extra inputs
        image_embeddings=None,
        image_positional_embeddings=None,
    ) -> InstanceList:
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]

        if hasattr(self, "extra_pe"):
            bs, _, h, w = x[0].shape
            mask_pe = torch.zeros((bs, h, w), device=x[0].device, dtype=torch.bool)
            img_feats_pe = self.extra_pe(mask_pe)
            outputs = []
            for i in range(len(x)):
                output = x[i] + F.interpolate(
                    img_feats_pe,
                    size=x[i].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                outputs.append(output)
            x = tuple(outputs)

        # If it has the mask branch, the bbox branch does not need
        # to be scaled to the original image scale, because the mask
        # branch will scale both bbox and mask at the same time.
        bbox_rescale = rescale if not self.with_mask else False
        results_list = self.predict_bbox(
            x,
            batch_img_metas,
            rpn_results_list,
            rcnn_test_cfg=self.test_cfg,
            rescale=bbox_rescale,
        )

        if self.with_mask:
            results_list = self.predict_mask(
                x,
                batch_img_metas,
                results_list,
                rescale=rescale,
                image_embeddings=image_embeddings,
                image_positional_embeddings=image_positional_embeddings,
            )
        return results_list


@MODELS.register_module()
class EUPGMaskHead(FCNMaskHead, BaseModule):
    """Turn RoI features into SAM sparse prompts and predict masks."""

    def __init__(
        self,
        mask_decoder,
        in_channels,
        roi_feat_size=14,
        per_pointset_point=5,
        with_sincos=True,
        multimask_output=False,
        attention_similarity=None,
        target_embedding=None,
        output_attentions=None,
        class_agnostic=False,
        loss_mask: ConfigType = None,
        init_cfg=None,
        *args,
        **kwargs,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)

        self.in_channels = in_channels
        self.roi_feat_size = roi_feat_size
        self.per_pointset_point = per_pointset_point
        self.with_sincos = with_sincos
        self.multimask_output = multimask_output
        self.attention_similarity = attention_similarity
        self.target_embedding = target_embedding
        self.output_attentions = output_attentions

        self.mask_decoder = MODELS.build(mask_decoder)

        prompt_encoder = dict(
            type="SAMPromptEncoder",
            hf_pretrain_name=copy.deepcopy(mask_decoder.get("hf_pretrain_name")),
            init_cfg=copy.deepcopy(mask_decoder.get("init_cfg")),
        )
        prompt_encoder = MODELS.build(prompt_encoder)
        prompt_encoder.init_weights()
        self.no_mask_embed = prompt_encoder.prompt_encoder.no_mask_embed

        if with_sincos:
            num_sincos = 2
        else:
            num_sincos = 1
        self.point_emb = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_channels * roi_feat_size**2 // 4, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels * num_sincos * per_pointset_point),
        )

        if loss_mask is None:
            loss_mask = dict(type="CrossEntropyLoss", use_mask=True, loss_weight=1.0)
        self.loss_mask = MODELS.build(loss_mask)
        self.class_agnostic = class_agnostic

    def init_weights(self) -> None:
        BaseModule.init_weights(self)

    def forward(
        self,
        x,
        image_embeddings,
        image_positional_embeddings,
        roi_img_ids=None,
    ):
        img_bs = image_embeddings.shape[0]
        roi_bs = x.shape[0]
        image_embedding_size = image_embeddings.shape[-2:]

        point_embeddings = self.point_emb(x)
        point_embeddings = einops.rearrange(
            point_embeddings, "b (n c) -> b n c", n=self.per_pointset_point
        )
        if self.with_sincos:
            point_embeddings = (
                torch.sin(point_embeddings[..., ::2]) + point_embeddings[..., 1::2]
            )

        # (B * N_set), N_point, C
        sparse_embeddings = point_embeddings.unsqueeze(1)
        num_roi_per_image = torch.bincount(roi_img_ids.long())
        # deal with the case that there is no roi in an image
        num_roi_per_image = torch.cat(
            [
                num_roi_per_image,
                torch.zeros(
                    img_bs - len(num_roi_per_image),
                    device=num_roi_per_image.device,
                    dtype=num_roi_per_image.dtype,
                ),
            ]
        )

        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            roi_bs, -1, image_embedding_size[0], image_embedding_size[1]
        )
        # get image embeddings with num_roi_per_image
        image_embeddings = image_embeddings.repeat_interleave(num_roi_per_image, dim=0)
        image_positional_embeddings = image_positional_embeddings.repeat_interleave(
            num_roi_per_image, dim=0
        )

        low_res_masks, iou_predictions, mask_decoder_attentions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_positional_embeddings=image_positional_embeddings,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
            attention_similarity=self.attention_similarity,
            target_embedding=self.target_embedding,
            output_attentions=self.output_attentions,
        )
        h, w = low_res_masks.shape[-2:]
        low_res_masks = low_res_masks.reshape(roi_bs, -1, h, w)
        iou_predictions = iou_predictions.reshape(roi_bs, -1)
        return low_res_masks, iou_predictions

    def get_targets(
        self,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        rcnn_train_cfg: ConfigDict,
    ) -> Tensor:
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        gt_masks = [res.masks for res in batch_gt_instances]
        mask_targets_list = []
        mask_size = rcnn_train_cfg.mask_size
        device = pos_proposals[0].device
        for pos_gt_inds, gt_mask in zip(pos_assigned_gt_inds, gt_masks):
            if len(pos_gt_inds) == 0:
                mask_targets = torch.zeros(
                    (0,) + mask_size, device=device, dtype=torch.float32
                )
            else:
                mask_targets = gt_mask[pos_gt_inds.cpu()].to_tensor(
                    dtype=torch.float32, device=device
                )
            mask_targets_list.append(mask_targets)
        mask_targets = torch.cat(mask_targets_list)
        return mask_targets

    def loss_and_target(
        self,
        mask_preds: Tensor,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        rcnn_train_cfg: ConfigDict,
    ) -> dict:
        mask_targets = self.get_targets(
            sampling_results=sampling_results,
            batch_gt_instances=batch_gt_instances,
            rcnn_train_cfg=rcnn_train_cfg,
        )

        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        # resize to mask_targets size
        mask_preds = F.interpolate(
            mask_preds,
            size=mask_targets.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        loss = dict()
        if mask_preds.size(0) == 0:
            loss_mask = mask_preds.sum()
        else:
            if self.class_agnostic:
                loss_mask = self.loss_mask(
                    mask_preds, mask_targets, torch.zeros_like(pos_labels)
                )
            else:
                loss_mask = self.loss_mask(mask_preds, mask_targets, pos_labels)
        loss["loss_mask"] = loss_mask
        return dict(loss_mask=loss, mask_targets=mask_targets)

    def _predict_by_feat_single(
        self,
        mask_preds: Tensor,
        bboxes: Tensor,
        labels: Tensor,
        img_meta: dict,
        rcnn_test_cfg: ConfigDict,
        rescale: bool = False,
        activate_map: bool = False,
    ) -> Tensor:
        scale_factor = bboxes.new_tensor(img_meta["scale_factor"]).repeat((1, 2))
        if not activate_map:
            mask_preds = mask_preds.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_preds = bboxes.new_tensor(mask_preds)

        if rescale:  # in-placed rescale the bboxes
            bboxes /= scale_factor
        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = F.interpolate(
            mask_preds,
            size=img_meta["batch_input_shape"],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        img_h, img_w = img_meta["img_shape"][:2]
        im_mask = im_mask[:, :img_h, :img_w]
        output_shape = img_meta["ori_shape"][:2] if rescale else (img_h, img_w)
        if im_mask.shape[-2:] != output_shape:
            im_mask = F.interpolate(
                im_mask.unsqueeze(1),
                size=output_shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        if threshold >= 0:
            im_mask = im_mask >= threshold
        else:
            im_mask = (im_mask * 255).to(dtype=torch.uint8)
        return im_mask


@MODELS.register_module()
class ChannelAttentionAggregator(BaseModule):
    """Apply channel attention to the final encoder feature map."""

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=256,
        act_cfg=None,
        inplace: bool = True,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        if hidden_channels is None:
            hidden_channels = in_channels // 4

        act_cfg = dict(type="ReLU") if act_cfg is None else act_cfg.copy()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        if act_cfg["type"] not in ["Tanh", "PReLU", "Sigmoid", "HSigmoid", "Swish"]:
            act_cfg.setdefault("inplace", inplace)
        self.act = build_activation_layer(act_cfg)
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            x = inputs[-1]
        avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
        return x * self.Sigmoid(avg_out + max_out).view(x.size(0), x.size(1), 1, 1)


@MODELS.register_module()
class SAMPositionalEmbedding(SamPositionalEmbedding, BaseModule):
    """SAM random Fourier positional embedding with checkpoint key layout."""

    def __init__(
        self,
        hf_pretrain_name,
        extra_config=None,
        init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).vision_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.shared_image_embedding = SamPositionalEmbedding(sam_config)

    def forward(self, *args, **kwargs):
        return self.shared_image_embedding(*args, **kwargs)


@MODELS.register_module()
class SAMPromptEncoder(SamPromptEncoder, BaseModule):
    """SAM prompt encoder used to initialize the no-mask token."""

    def __init__(
        self,
        hf_pretrain_name,
        extra_config=None,
        init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).prompt_encoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.prompt_encoder = SamPromptEncoder(sam_config, shared_patch_embedding=None)

    def forward(self, *args, **kwargs):
        return self.prompt_encoder(*args, **kwargs)


@MODELS.register_module()
class SAMMaskDecoder(SamMaskDecoder, BaseModule):
    """SAM mask decoder wrapper preserving checkpoint key layout."""

    def __init__(
        self,
        hf_pretrain_name,
        extra_config=None,
        init_cfg=None,
    ):
        BaseModule.__init__(self, init_cfg=init_cfg)
        sam_config = SamConfig.from_pretrained(hf_pretrain_name).mask_decoder_config
        if extra_config is not None:
            sam_config.update(extra_config)
        self.mask_decoder = SamMaskDecoder(sam_config)

    def forward(self, *args, **kwargs):
        return self.mask_decoder(*args, **kwargs)
