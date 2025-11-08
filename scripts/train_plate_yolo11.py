#!/usr/bin/env python
import argparse, os
from ultralytics import YOLO

# Simple YOLOv11 training launcher for license plates
# Expects data YAML pointing to outputs/datasets/plate_yolo11

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_yaml', default='data/plate_yolo11.yaml')
    ap.add_argument('--model', default='yolo11n.pt', help='Base model, e.g., yolo11n.pt/yolo11s.pt')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--device', default='0')
    ap.add_argument('--project', default='runs/detect')
    ap.add_argument('--name', default='plate_yolo11')
    args = ap.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    # Optionally copy best weight to models/
    try:
        best = os.path.join(args.project, args.name, 'weights', 'best.pt')
        if os.path.isfile(best):
            os.makedirs('models', exist_ok=True)
            dst = os.path.join('models', 'plate_yolo11_best.pt')
            import shutil
            shutil.copy2(best, dst)
            print('Saved best weight to', dst)
    except Exception as e:
        print('[WARN] Could not copy best weight:', e)

if __name__ == '__main__':
    main()
