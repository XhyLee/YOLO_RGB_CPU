"""Preview the trained RGB detector on a camera or an image with per-class thresholds."""
import argparse
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def render(model, frame, size, thresholds):
    result = model.predict(frame, imgsz=size, conf=min(thresholds.values()),
                           device='cpu', verbose=False)[0]
    names = result.names
    boxes = result.boxes
    keep = boxes.conf < 0  # Boolean mask on the same device as the prediction.
    for class_id, name in names.items():
        keep |= (boxes.cls == class_id) & (boxes.conf >= thresholds[name])
    result = result[keep]
    return result.plot(), {name: int((result.boxes.cls == i).sum()) for i, name in names.items()}


def save_jpeg(cv2, path, frame):
    ok, data = cv2.imencode('.jpg', frame)
    if not ok:
        raise RuntimeError(f'Cannot encode image: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, default=ROOT / 'models/best.pt')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--camera', type=int, default=0, help='Camera index, defaults to 0')
    source.add_argument('--image', type=Path, help='Run on a local image and save an annotated JPEG')
    parser.add_argument('--output', type=Path, help='Annotated JPEG path for --image')
    parser.add_argument('--imgsz', type=int, choices=(320, 416, 640), default=640)
    parser.add_argument('--fire-conf', type=float, default=0.25)
    parser.add_argument('--smoke-conf', type=float, default=0.30)
    args = parser.parse_args()
    thresholds = {'fire': args.fire_conf, 'smoke': args.smoke_conf}
    if not all(0 < conf <= 1 for conf in thresholds.values()):
        parser.error('Confidence thresholds must be greater than 0 and at most 1.')
    if args.output and (not args.image or args.output.suffix.lower() not in {'.jpg', '.jpeg'}):
        parser.error('--output requires --image and a .jpg or .jpeg extension.')
    if not args.weights.is_file():
        parser.error(f'Weights not found: {args.weights}')

    import cv2
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(args.weights)
    if set(model.names.values()) != set(thresholds):
        parser.error(f'Expected fire and smoke classes, found: {model.names}')
    if args.image:
        frame = cv2.imdecode(np.frombuffer(args.image.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            parser.error(f'Cannot read image: {args.image}')
        preview, counts = render(model, frame, args.imgsz, thresholds)
        output = args.output or ROOT / 'runs' / 'preview' / f'{args.image.stem}_preview.jpg'
        if args.image.resolve() == output.resolve():
            parser.error('Input and output paths must differ.')
        save_jpeg(cv2, output, preview)
        print(f'{counts} -> {output}')
        return

    window = 'RGB fire and smoke | Q: quit | S: save raw frame'
    camera = cv2.VideoCapture(args.camera)
    try:
        if not camera.isOpened():
            raise RuntimeError('Camera could not open. Check permissions or use --camera 1.')
        print(f'imgsz={args.imgsz}; fire={args.fire_conf:g}; smoke={args.smoke_conf:g}')
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError('No frame available from the camera.')
            preview, _ = render(model, frame, args.imgsz, thresholds)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord('s'):
                filename = datetime.now().strftime('camera_%Y%m%d_%H%M%S_%f.jpg')
                target = ROOT / 'records' / 'field_candidates' / filename
                save_jpeg(cv2, target, frame)
                print(f'Saved: {target}', flush=True)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
