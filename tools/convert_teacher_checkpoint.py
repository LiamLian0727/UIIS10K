"""Prepare a trained UWSAM teacher checkpoint for student distillation."""

import argparse
from pathlib import Path

import torch


def convert_state_dict(state_dict):
    """Move teacher backbone weights under the student's ``teacher`` module."""
    converted = {}
    backbone_count = 0
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            new_key = "teacher." + key[len("backbone.") :]
            backbone_count += 1
        else:
            new_key = key
        if new_key in converted:
            raise ValueError(f"Duplicate key after teacher conversion: {new_key}")
        converted[new_key] = value
    if backbone_count == 0:
        raise ValueError("Checkpoint has no backbone.* keys to convert")
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Trained teacher MMEngine checkpoint")
    parser.add_argument(
        "output", type=Path, nargs="?", default=Path("pretrain/uwsam_teacher.pth")
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state_dict = convert_state_dict(checkpoint["state_dict"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict}, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
