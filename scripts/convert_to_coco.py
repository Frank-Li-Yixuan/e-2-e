#!/usr/bin/env python
"""
Convert known datasets to COCO with categories:
  1: face, 2: license_plate

This script provides thin adapters or guidance; many datasets require license acceptance
and custom parsers. For each dataset, pass the relevant arguments. If an adapter is not
implemented, the script will print actionable instructions.

Usage examples:
  # WIDER FACE (needs an external converter or custom adapter)
  python scripts/convert_to_coco.py wider_face --images <dir> --wider_annotations <dir> --output data/wider_coco.json

  # CCPD (plates)
  python scripts/convert_to_coco.py ccpd --images <dir> --output data/ccpd_coco.json

  # UFPR-ALPR / SSIG-SegPlate / CrowdHuman
  python scripts/convert_to_coco.py ufpr_alpr --images <dir> --ann_dir <dir> --output data/ufpr_coco.json

"""
import argparse
import json
import os
from typing import Dict, Any, List

CAT_TARGET = [
    {"id": 1, "name": "face"},
    {"id": 2, "name": "license_plate"},
]


def wider_face_to_coco(images: str, wider_annotations: str, output: str) -> None:
    print("[INFO] WIDER FACE conversion not fully implemented in this script.")
    print("Please use an existing converter (e.g., wider-face to COCO) or adapt here.")
    print("Expected mapping: face -> category_id=1.")
    print(f"Suggested output path: {output}")


def ccpd_to_coco(images: str, output: str) -> None:
    from pathlib import Path
    try:
        from PIL import Image
    except Exception:
        print("[WARN] PIL not installed; cannot compute image sizes. Proceeding with None sizes.")
        Image = None  # type: ignore
    imgs = []
    anns = []
    next_img_id = 1
    next_ann_id = 1
    img_files = [p for p in Path(images).rglob("*.jpg")]
    for p in img_files:
        w = h = None
        if Image is not None:
            try:
                w, h = Image.open(str(p)).size
            except Exception:
                pass
        imgs.append({"id": next_img_id, "file_name": str(p.relative_to(images)).replace("\\", "/"), "width": w, "height": h})
        # CCPD labels are encoded in filename; simplistic bounding box extraction not provided here.
        # Many community scripts exist to decode CCPD metadata into plate boxes. Please integrate one here.
        next_img_id += 1
    coco = {"images": imgs, "annotations": anns, "categories": CAT_TARGET}
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"[INFO] CCPD COCO (images only, no boxes) written to {output}. Fill annotations via decoder.")


def generic_with_ann_dir(images: str, ann_dir: str, output: str, cat_id: int) -> None:
    from pathlib import Path
    try:
        from PIL import Image
    except Exception:
        print("[WARN] PIL not installed; cannot compute image sizes. Proceeding with None sizes.")
        Image = None  # type: ignore
    imgs = []
    anns = []
    next_img_id = 1
    next_ann_id = 1
    for p in Path(images).rglob("*.jpg"):
        w = h = None
        if Image is not None:
            try:
                w, h = Image.open(str(p)).size
            except Exception:
                pass
        stem = p.stem
        txt = Path(ann_dir) / f"{stem}.txt"
        imgs.append({"id": next_img_id, "file_name": str(p.relative_to(images)).replace("\\", "/"), "width": w, "height": h})
        if txt.exists():
            try:
                with open(txt, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            x, y, w1, h1 = map(float, parts[:4])
                            anns.append({
                                "id": next_ann_id, "image_id": next_img_id, "category_id": cat_id,
                                "bbox": [x, y, w1, h1], "area": w1 * h1, "iscrowd": 0,
                            })
                            next_ann_id += 1
            except Exception:
                pass
        next_img_id += 1
    coco = {"images": imgs, "annotations": anns, "categories": CAT_TARGET}
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"[INFO] COCO written to {output} with {len(imgs)} images and {len(anns)} annotations")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="dataset", required=True)

    s1 = sub.add_parser("wider_face")
    s1.add_argument("--images", required=True)
    s1.add_argument("--wider_annotations", required=True)
    s1.add_argument("--output", required=True)

    s2 = sub.add_parser("ccpd")
    s2.add_argument("--images", required=True)
    s2.add_argument("--output", required=True)

    s3 = sub.add_parser("ufpr_alpr")
    s3.add_argument("--images", required=True)
    s3.add_argument("--ann_dir", required=True)
    s3.add_argument("--output", required=True)

    s4 = sub.add_parser("ssig_segplate")
    s4.add_argument("--images", required=True)
    s4.add_argument("--ann_dir", required=True)
    s4.add_argument("--output", required=True)

    s5 = sub.add_parser("crowdhuman")
    s5.add_argument("--images", required=True)
    s5.add_argument("--ann_dir", required=True)
    s5.add_argument("--output", required=True)

    args = ap.parse_args()
    if args.dataset == "wider_face":
        wider_face_to_coco(args.images, args.wider_annotations, args.output)
    elif args.dataset == "ccpd":
        ccpd_to_coco(args.images, args.output)
    elif args.dataset in ("ufpr_alpr", "ssig_segplate", "crowdhuman"):
        # cat id heuristic: ufpr/ssig plates => 2, crowdhuman possibly faces (1) if head boxes
        cat = 2 if args.dataset in ("ufpr_alpr", "ssig_segplate") else 1
        generic_with_ann_dir(args.images, args.ann_dir, args.output, cat_id=cat)


if __name__ == "__main__":
    main()
