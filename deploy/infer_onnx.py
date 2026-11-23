"""YOLO11 raw ONNX detection on an image or camera, without PyTorch."""
from pathlib import Path
import argparse
import time

NAMES = ('fire', 'smoke')

def prepare(image, size):
    import numpy as np
    from PIL import Image
    w, h = image.size
    ratio = min(size / w, size / h)
    nw, nh = max(1, round(w * ratio)), max(1, round(h * ratio))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = Image.new('RGB', (size, size), (114, 114, 114))
    canvas.paste(image.resize((nw, nh), Image.Resampling.BILINEAR), (left, top))
    tensor = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(tensor), (nw / w, nh / h, left, top)

def decode(output, transform, original_size, conf, iou, max_det=300):
    import numpy as np
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 4 + len(NAMES):
        raise ValueError(f'Expected YOLO11 raw [1,6,N] for fire/smoke, got {output.shape}')
    pred = output[0].T
    scores = pred[:, 4:].max(axis=1)
    classes = pred[:, 4:].argmax(axis=1)
    keep = np.isfinite(pred).all(axis=1) & (scores >= conf) & (pred[:, 2] > 0) & (pred[:, 3] > 0)
    pred, scores, classes = pred[keep], scores[keep], classes[keep]
    boxes = np.concatenate((pred[:, :2] - pred[:, 2:4] / 2,
                            pred[:, :2] + pred[:, 2:4] / 2), axis=1)
    chosen = []
    for cls in np.unique(classes):
        idx = np.where(classes == cls)[0]
        order = idx[np.argsort(scores[idx])[::-1]]
        while len(order):
            current = int(order[0]); chosen.append(current)
            rest = order[1:]
            if not len(rest): break
            lt = np.maximum(boxes[current, :2], boxes[rest, :2])
            rb = np.minimum(boxes[current, 2:], boxes[rest, 2:])
            wh = np.maximum(0, rb - lt)
            intersection = wh[:, 0] * wh[:, 1]
            area_current = np.prod(boxes[current, 2:] - boxes[current, :2])
            area_rest = np.prod(boxes[rest, 2:] - boxes[rest, :2], axis=1)
            overlap = intersection / np.maximum(area_current + area_rest - intersection, 1e-9)
            order = rest[overlap <= iou]
    chosen = sorted(chosen, key=lambda k: float(scores[k]), reverse=True)[:max_det]
    sx, sy, left, top = transform
    width, height = original_size
    detections = []
    for k in chosen:
        b = boxes[k].astype(float).copy()
        b[[0, 2]] = np.clip((b[[0, 2]] - left) / sx, 0, width)
        b[[1, 3]] = np.clip((b[[1, 3]] - top) / sy, 0, height)
        if b[2] > b[0] and b[3] > b[1]:
            detections.append((int(classes[k]), float(scores[k]), tuple(b)))
    return detections

def annotate(image, detections):
    from PIL import ImageDraw
    result = image.copy(); draw = ImageDraw.Draw(result)
    for cls, score, box in detections:
        color = '#ff4c32' if cls == 0 else '#00c875'
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0], max(0, box[1] - 12)), f'{NAMES[cls]} {score:.2f}', fill=color)
    return result

def main():
    p = argparse.ArgumentParser(description='RGB fire/smoke YOLO11 ONNX CPU inference; no PyTorch needed.')
    p.add_argument('--model', type=Path, default=Path(__file__).parent/'rgb_yolo11s_320.onnx')
    p.add_argument('--source', required=True, help='Image path or camera index, e.g. 0')
    p.add_argument('--output', type=Path, help='Annotated image, or .avi for camera recording')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument('--show', action='store_true', help='Camera GUI window; requires OpenCV with GUI and desktop session')
    p.add_argument('--max-frames', type=int, default=300, help='Maximum camera frames; 0 means unlimited')
    a = p.parse_args()
    if not a.model.is_file():p.error('ONNX model missing. Export it on the training PC first.')
    if not 0 <= a.conf <= 1 or not 0 <= a.iou <= 1 or a.threads < 1 or a.max_frames < 0:
        p.error('Invalid thresholds, thread count, or frame limit.')
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    opts = ort.SessionOptions(); opts.intra_op_num_threads = a.threads; opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(a.model), sess_options=opts, providers=['CPUExecutionProvider'])
    input_info = session.get_inputs()[0]
    if (len(input_info.shape) != 4 or input_info.shape[:2] != [1, 3]
            or not isinstance(input_info.shape[2], int) or input_info.shape[2] != input_info.shape[3]
            or input_info.type != 'tensor(float)'):
        raise ValueError(f'Expected static FP32 [1,3,S,S], got {input_info.shape}, {input_info.type}')
    size = input_info.shape[2]
    if a.output: a.output.parent.mkdir(parents=True, exist_ok=True)
    def detect(image):
        tensor, transform = prepare(image, size)
        start = time.perf_counter()
        output = session.run(None, {input_info.name: tensor})[0]
        elapsed = time.perf_counter() - start
        return decode(output, transform, image.size, a.conf, a.iou), elapsed
    # Exclude one-time initialization from reported model timing.
    session.run(None, {input_info.name: np.zeros((1, 3, size, size), dtype=np.float32)})
    if not a.source.isdecimal():
        with Image.open(a.source) as src: image = src.convert('RGB')
        start = time.perf_counter(); detections, model_s = detect(image)
        total_s = time.perf_counter() - start
        result = annotate(image, detections)
        target = a.output or Path('prediction.jpg'); target.parent.mkdir(parents=True, exist_ok=True)
        result.save(target)
        for cls, score, box in detections:print(NAMES[cls], round(score, 4), [round(x, 1) for x in box])
        print(f'Saved {target}; model={model_s*1000:.1f} ms; preprocessing+model+NMS={total_s*1000:.1f} ms')
        return
    import cv2
    if a.output and a.output.suffix.lower() != '.avi':p.error('Camera output must end in .avi')
    capture = cv2.VideoCapture(int(a.source)); writer = None
    if not capture.isOpened():raise RuntimeError('Cannot open camera. Check camera index and OS permissions.')
    count = 0; model_times = []; start = time.perf_counter()
    try:
        while not a.max_frames or count < a.max_frames:
            ok, frame = capture.read()
            if not ok:break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detections, model_s = detect(image); model_times.append(model_s)
            result = cv2.cvtColor(np.array(annotate(image, detections)), cv2.COLOR_RGB2BGR)
            if a.output:
                if writer is None:
                    # Playback rate is fixed; measured inference rate is printed separately.
                    writer = cv2.VideoWriter(str(a.output), cv2.VideoWriter_fourcc(*'MJPG'), 10,
                                             (result.shape[1], result.shape[0]))
                    if not writer.isOpened():raise RuntimeError('Cannot create AVI output.')
                writer.write(result)
            count += 1
            if count % 10 == 0:
                print(f'frames={count}, measured pipeline FPS={count/(time.perf_counter()-start):.2f}', flush=True)
            if a.show:
                cv2.imshow('RGB fire / smoke', result)
                if cv2.waitKey(1) & 0xff == ord('q'):break
    finally:
        capture.release()
        if writer is not None:writer.release()
        if a.show:cv2.destroyAllWindows()
    if not count:raise RuntimeError('No frames received from camera.')
    print(f'frames={count}; measured pipeline FPS={count/(time.perf_counter()-start):.2f}; '
          f'mean model time={1000*sum(model_times)/len(model_times):.1f} ms')

if __name__ == '__main__':
    main()
