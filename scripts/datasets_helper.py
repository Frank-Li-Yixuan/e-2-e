#!/usr/bin/env python
"""
Dataset helper for downloading/preparing datasets and mapping to COCO with categories:
  1: face, 2: license_plate

This script prints instructions and provides stubs to convert known datasets into COCO.
Actual downloads require accepting licenses and manual steps per dataset.

Usage:
  python scripts/datasets_helper.py --list
  python scripts/datasets_helper.py --howto wider_face
  python scripts/datasets_helper.py --howto ccpd

After preparing per-dataset COCO files, use ensure_coco_mapping.py to merge.
"""
import argparse

HOWTO = {
    "wider_face": "Download WIDER FACE images and annotations; convert to COCO (faces->category_id=1). Use community converters or write a small adapter; then run ensure_coco_mapping.py.",
    "ccpd": "Clone https://github.com/detectRecog/CCPD; parse file names/labels to build COCO for license_plate (category_id=2). Many community scripts exist.",
    "ufpr_alpr": "Download UFPR-ALPR from the GitHub; annotations vary; map plates to category_id=2 and export COCO.",
    "ssig_segplate": "Download SSIG-SegPlate; use masks/boxes to form COCO with plates as category_id=2.",
    "crowdhuman": "Download CrowdHuman; if using people as 'face/person' proxy, decide mapping. Typically skip or map heads/faces to category_id=1 via head boxes.",
    "ijbc": "IJB-C requires access; if obtained, derive face boxes and export to COCO with category_id=1.",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--howto", choices=list(HOWTO.keys()))
    args = ap.parse_args()
    if args.list:
        print("Datasets:")
        for k in HOWTO:
            print(" -", k)
        return
    print(HOWTO[args.howto])


if __name__ == "__main__":
    main()
