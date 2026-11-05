"""Export a trained student without distillation-only teacher weights."""

import argparse
from pathlib import Path

import torch


def student_state_dict(state_dict):
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(("teacher.", "distiller."))
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Trained student MMEngine checkpoint")
    parser.add_argument("output", type=Path, help="Student-only inference checkpoint")
    args = parser.parse_args()

    checkpoint = torch.load(args.input, map_location="cpu", weights_only=False)
    state_dict = student_state_dict(checkpoint["state_dict"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": state_dict}, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
