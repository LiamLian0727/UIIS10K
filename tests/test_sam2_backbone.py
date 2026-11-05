"""CPU checks for the optional SAM2 Hiera Mask R-CNN backbone."""

import torch

from uwsam.extensions.sam2 import SAM2Backbone

TINY_HIERA = dict(
    embed_dim=16,
    num_heads=1,
    stages=(1, 1, 1, 1),
    global_att_blocks=(),
    window_pos_embed_bkg_spatial_size=(2, 2),
    window_spec=(2, 2, 2, 2),
)


def test_hiera_returns_four_fpn_feature_levels():
    model = SAM2Backbone(**TINY_HIERA).eval()

    with torch.no_grad():
        features = model(torch.randn(1, 3, 32, 32))

    assert isinstance(features, tuple)
    assert [tuple(feature.shape) for feature in features] == [
        (1, 16, 8, 8),
        (1, 32, 4, 4),
        (1, 64, 2, 2),
        (1, 128, 1, 1),
    ]


def test_official_model_checkpoint_loads_trunk_before_lora(tmp_path):
    original = SAM2Backbone(**TINY_HIERA)
    with torch.no_grad():
        original.trunk.patch_embed.proj.weight.fill_(0.125)

    checkpoint = tmp_path / "sam2_hiera_tiny.pt"
    torch.save(
        {
            "model": {
                f"image_encoder.trunk.{name}": value.clone()
                for name, value in original.trunk.state_dict().items()
            }
        },
        checkpoint,
    )

    loaded = SAM2Backbone(
        checkpoint=str(checkpoint),
        peft_config=dict(
            peft_type="LORA",
            r=2,
            target_modules=["qkv"],
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            inference_mode=False,
        ),
        **TINY_HIERA,
    )

    patch_weight = next(
        parameter
        for name, parameter in loaded.trunk.named_parameters()
        if name.endswith("patch_embed.proj.weight")
    )
    assert torch.equal(patch_weight, original.trunk.patch_embed.proj.weight)
    assert any("lora_" in name for name, _ in loaded.trunk.named_parameters())
