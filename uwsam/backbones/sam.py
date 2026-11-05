"""SAM image encoder backed by MMPretrain ViTSAM."""

from mmdet.registry import MODELS
from mmengine.dist import is_main_process
from mmengine.model import BaseModule
from peft import get_peft_config, get_peft_model


@MODELS.register_module()
class SAMImageEncoder(BaseModule):
    """Build a SAM ViT encoder and expose selected MMPretrain features."""

    def __init__(
        self,
        hf_pretrain_name,
        img_size=1024,
        out_indices=-1,
        peft_config=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        vision_encoder_cfg = dict(
            type="mmpretrain.ViTSAM",
            arch=hf_pretrain_name.split("-")[-1].split("_")[-1],
            img_size=img_size,
            patch_size=16,
            out_indices=out_indices,
            out_channels=256,
            use_abs_pos=True,
            use_rel_pos=True,
            window_size=14,
        )
        vision_encoder = MODELS.build(vision_encoder_cfg)
        # load checkpoint
        if init_cfg is not None:
            from mmengine.runner.checkpoint import load_checkpoint

            load_checkpoint(
                vision_encoder,
                init_cfg.get("checkpoint"),
                map_location="cpu",
                revise_keys=[
                    (r"^module\.", ""),
                    (r"^vision_encoder\.", ""),
                    (r".layer_norm1.", ".ln1."),
                    (r".layer_norm2.", ".ln2."),
                    (r".mlp.lin1.", ".ffn.layers.0.0."),
                    (r".mlp.lin2.", ".ffn.layers.1."),
                    (r"neck.conv1.", "channel_reduction.0."),
                    (r"neck.ln1.", "channel_reduction.1."),
                    (r"neck.conv2.", "channel_reduction.2."),
                    (r"neck.ln2.", "channel_reduction.3."),
                ],
            )

        if peft_config is not None and isinstance(peft_config, dict):
            config = {
                "peft_type": "LORA",
                "r": 16,
                "target_modules": ["qkv"],
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "bias": "none",
                "inference_mode": False,
            }
            config.update(peft_config)
            peft_config = get_peft_config(config)
            self.vision_encoder = get_peft_model(vision_encoder, peft_config)
            if is_main_process():
                self.vision_encoder.print_trainable_parameters()
        else:
            self.vision_encoder = vision_encoder
        self.vision_encoder.is_init = True

    def init_weights(self):
        # Checkpoint weights are loaded in __init__ or by the detector init_cfg.
        pass

    def forward(self, *args, **kwargs):
        return self.vision_encoder(*args, **kwargs)
