"""Export static, FP32, batch=1 ONNX; NMS stays outside the graph."""
from pathlib import Path
import argparse
import json
import shutil

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description='Export the trained RGB model to ONNX for CPU inference.')
    p.add_argument('--weights',type=Path,default=ROOT/'models/best.pt')
    p.add_argument('--imgsz',type=int,choices=[320,416,512,640],default=320)
    a=p.parse_args()
    if not a.weights.is_file():p.error('Model missing. Complete full training first.')
    from ultralytics import YOLO
    import onnx
    import onnxruntime as ort
    import numpy as np
    model=YOLO(str(a.weights.resolve()))
    exported=Path(model.export(format='onnx',imgsz=a.imgsz,batch=1,device='cpu',
                             half=False,dynamic=False,simplify=False,opset=12,nms=False))
    target=ROOT/'deploy'/f'rgb_yolo11s_{a.imgsz}.onnx'
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(exported,target)
    onnx.checker.check_model(onnx.load(str(target)))
    options=ort.SessionOptions();options.intra_op_num_threads=4
    session=ort.InferenceSession(str(target),sess_options=options,providers=['CPUExecutionProvider'])
    inp=session.get_inputs()[0]
    if inp.shape!=[1,3,a.imgsz,a.imgsz]:raise ValueError(f'Unexpected input shape: {inp.shape}')
    result=session.run(None,{inp.name:np.zeros(inp.shape,dtype=np.float32)})
    if not all(np.isfinite(x).all() for x in result):raise ValueError('Non-finite ONNX output')
    meta={'model':target.name,'names':{'0':'fire','1':'smoke'},'input':{'shape':inp.shape,'dtype':'float32','color':'RGB','normalization':'divide by 255','resize':'letterbox, pad value 114'},'output_shapes':[list(x.shape) for x in result],'postprocessing':'YOLO11 raw detect output: xywh boxes and class scores; confidence filter, class-aware NMS, undo letterbox','nms_embedded':False,'board_benchmarked':False,'validated_on_images':False}
    target.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Exported and ONNX CPU shape-check passed:',target)
    print('Next: evaluate this ONNX on val at the same imgsz. Board speed and accuracy remain untested.')

if __name__=='__main__':
    main()
