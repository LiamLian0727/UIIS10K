"""Download the public weights used by the UWSAM configs.

Examples:
    python pretrain/download_pretrained.py sam
    python pretrain/download_pretrained.py efficient-sam-vits
    python pretrain/download_pretrained.py sam2-hiera-l
"""

import argparse
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

PRETRAIN_DIR = Path(__file__).resolve().parent
EFFICIENT_SAM_URL = (
    "https://github.com/yformer/EfficientSAM/raw/main/weights/efficient_sam_vits.pt.zip"
)
SAM2_HIERA_L_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
)


def download_file(url, destination):
    if destination.exists():
        print(f"Already present: {destination}")
        return
    temporary = destination.with_name(destination.name + ".part")
    with urllib.request.urlopen(url) as source, temporary.open("wb") as output:
        shutil.copyfileobj(source, output)
    temporary.replace(destination)
    print(f"Downloaded {destination}")


def download_sam(output_dir):
    from huggingface_hub import hf_hub_download

    destination = output_dir / "sam-vit-huge"
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "pytorch_model.bin"):
        hf_hub_download("facebook/sam-vit-huge", filename, local_dir=destination)
    print(f"Downloaded SAM ViT-H into {destination}")


def download_efficient_sam(output_dir):
    import torch

    destination = output_dir / "efficient_sam_vits.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Already present: {destination}")
        return
    archive = destination.with_suffix(".pt.zip")
    download_file(EFFICIENT_SAM_URL, archive)
    with zipfile.ZipFile(archive) as zipped:
        name = next(
            name
            for name in zipped.namelist()
            if Path(name).name == "efficient_sam_vits.pt"
        )
        checkpoint = torch.load(
            io.BytesIO(zipped.read(name)), map_location="cpu", weights_only=False
        )
    state_dict = checkpoint.get("model", checkpoint)
    temporary = destination.with_name(destination.name + ".part")
    torch.save({"state_dict": state_dict}, temporary)
    temporary.replace(destination)
    archive.unlink()
    print(f"Prepared {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("sam", "efficient-sam-vits", "sam2-hiera-l"))
    parser.add_argument("--output-dir", type=Path, default=PRETRAIN_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "sam":
        download_sam(args.output_dir)
    elif args.model == "efficient-sam-vits":
        download_efficient_sam(args.output_dir)
    else:
        download_file(SAM2_HIERA_L_URL, args.output_dir / "sam2_hiera_large.pt")


if __name__ == "__main__":
    main()
