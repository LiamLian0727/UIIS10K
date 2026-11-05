"""SAM2 Hiera image trunk as a standard MMDetection backbone."""

from mmdet.registry import MODELS
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import CheckpointLoader

from .hieradet import Hiera


@MODELS.register_module()
class SAM2Backbone(BaseModule):
    """Return Hiera's four spatial features for a standard FPN.

    ``checkpoint`` is an official SAM2 checkpoint with a ``model`` key, or an
    MMEngine checkpoint with a ``state_dict`` key. LoRA is applied after the
    pretrained trunk weights have been loaded.
    """

    def __init__(self, checkpoint=None, peft_config=None, **hiera_kwargs):
        super().__init__()
        self.trunk = Hiera(return_interm_layers=True, **hiera_kwargs)

        if checkpoint is not None:
            saved = CheckpointLoader.load_checkpoint(checkpoint, map_location="cpu")
            weights = saved.get("state_dict", saved.get("model", saved))
            prefix = "image_encoder.trunk."
            trunk_weights = {
                name[len(prefix) :]: value
                for name, value in weights.items()
                if name.startswith(prefix)
            }
            if not trunk_weights:
                raise ValueError(f"No {prefix} weights found in {checkpoint}")
            self.trunk.load_state_dict(trunk_weights, strict=False)

        if peft_config is not None:
            from peft import get_peft_config, get_peft_model

            self.trunk = get_peft_model(self.trunk, get_peft_config(peft_config))

    def forward(self, x):
        return tuple(self.trunk(x))
