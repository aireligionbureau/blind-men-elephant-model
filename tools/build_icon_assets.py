from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


PNG_SIZES = {
    "favicon-32.png": 32,
    "apple-touch-icon.png": 180,
    "icon-192.png": 192,
    "icon-512.png": 512,
}
ICO_SIZES = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build web and Windows icons from the model mark."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("homepage/assets"),
    )
    args = parser.parse_args()

    source = args.source.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as opened:
        original = opened.convert("RGB")
    icon_master = _tight_square_crop(original)

    source_copy = output_dir / "blind-men-elephant-model-mark.png"
    if source != source_copy.resolve():
        shutil.copy2(source, source_copy)

    master_path = output_dir / "blind-men-elephant-model-icon.png"
    master = icon_master.resize((1024, 1024), Image.Resampling.LANCZOS)
    master.save(master_path, format="PNG", optimize=True)

    generated: dict[str, list[int]] = {master_path.name: [1024, 1024]}
    for filename, size in PNG_SIZES.items():
        target = output_dir / filename
        master.resize((size, size), Image.Resampling.LANCZOS).save(
            target,
            format="PNG",
            optimize=True,
        )
        generated[filename] = [size, size]

    ico_path = output_dir / "blind-men-elephant-model.ico"
    master.save(ico_path, format="ICO", sizes=ICO_SIZES)
    generated[ico_path.name] = [size[0] for size in ICO_SIZES]
    print(json.dumps(generated, ensure_ascii=False, indent=2))


def _tight_square_crop(image: Image.Image) -> Image.Image:
    grayscale = image.convert("L")
    foreground = grayscale.point(lambda value: 255 if value > 14 else 0)
    bbox = foreground.getbbox()
    if bbox is None:
        return image.copy()

    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    side = min(
        min(image.size),
        max(width, height) * 1.10,
    )
    center_x = (left + right) / 2
    center_y = (top + bottom) / 2
    crop_left = round(center_x - side / 2)
    crop_top = round(center_y - side / 2)
    crop_left = max(0, min(crop_left, round(image.width - side)))
    crop_top = max(0, min(crop_top, round(image.height - side)))
    crop_side = round(side)
    return image.crop(
        (
            crop_left,
            crop_top,
            crop_left + crop_side,
            crop_top + crop_side,
        )
    )


if __name__ == "__main__":
    main()
