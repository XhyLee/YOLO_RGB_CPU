"""Evaluate PT or ONNX RGB detector on val or the held-out test set."""
from pathlib import Path
import argparse
from datetime import datetime

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description='Evaluate the RGB detector on CPU.')
    p.add_argument('--weights',type=Path,default=ROOT/'models/best.pt')
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--imgsz',type=int,default=640)
    a=p.parse_args()
    if not a.weights.is_file():p.error('Model missing. Complete full training first, or set --weights.')
    import torch
    import yaml
    from ultralytics import YOLO
    torch.set_num_threads(8)
    data=ROOT/'data'
    config={'path':data.as_posix(),'train':'images/train','val':'images/val','test':'images/test','names':{0:'fire',1:'smoke'}}
    (data/'data.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    model=YOLO(str(a.weights.resolve()),task='detect')
    metrics=model.val(data=str(data/'data.yaml'),split=a.split,device='cpu',imgsz=a.imgsz,
                      batch=1,workers=0,half=False,plots=True,project=str(ROOT/'runs'),
                      name=f'eval_{a.split}_{a.imgsz}_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    print('mAP50:',metrics.box.map50,'mAP50-95:',metrics.box.map)
    print('Per-image pipeline timing (ms):',metrics.speed)
    print('PC timing is not Orange Pi timing.')

if __name__=='__main__':
    main()
