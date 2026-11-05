"""UWSAM model registration.

Importing this package registers the teacher, student, and their active modules
with MMDetection.
"""

from .backbones.efficient_sam import EfficientSAMEncoder
from .backbones.sam import SAMImageEncoder
from .components import (
    ChannelAttentionAggregator,
    EUPGMaskHead,
    EUPGNeck,
    EUPGRoIHead,
    SAMMaskDecoder,
    SAMPositionalEmbedding,
    SAMPromptEncoder,
)
from .detector import UWSAM
from .distillation import FeatureReconstructionLoss, MaskGATConnector, MGUKDDistiller
from .layers import LN2d
from .necks import SimpleFPN
from .optim import LayerDecayOptimizerConstructor

__all__ = [
    "UWSAM",
    "EUPGNeck",
    "EUPGRoIHead",
    "EUPGMaskHead",
    "ChannelAttentionAggregator",
    "SAMPositionalEmbedding",
    "SAMPromptEncoder",
    "SAMMaskDecoder",
    "SAMImageEncoder",
    "EfficientSAMEncoder",
    "MGUKDDistiller",
    "MaskGATConnector",
    "FeatureReconstructionLoss",
    "LN2d",
    "SimpleFPN",
    "LayerDecayOptimizerConstructor",
]
