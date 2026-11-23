"""CPU-only RGB training. Run this file from the project folder in VS Code."""
from pathlib import Path
import argparse
import json
import os
import shutil
from datetime import datetime

ROOT = Path(__file__).resolve().parent

def configure_cpu_threads(model, threads):
    # select_device() resets PyTorch's thread count inside Ultralytics.
    # Apply the requested value through callbacks after device setup, including validation.
    import torch
    def apply_threads(runner):
        torch.set_num_threads(threads)
    def report_threads(trainer):
        apply_threads(trainer)
        print(f'CPU compute threads: {torch.get_num_threads()} '
              f'(logical CPUs: {os.cpu_count()}); device={trainer.device}', flush=True)
    model.add_callback('on_train_start', report_threads)
    model.add_callback('on_train_epoch_start', apply_threads)
    model.add_callback('on_val_start', apply_threads)

def main():
    parser = argparse.ArgumentParser(description='Train RGB fire/smoke YOLO11s on CPU.')
    parser.add_argument('--quick', action='store_true', help='1 epoch, 320px, 10%% of training images; only a setup check')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--resume', type=Path, help='Path to an interrupted run/weights/last.pt')
    args = parser.parse_args()
    if min(args.epochs,args.imgsz,args.batch,args.threads)<=0:parser.error('Numeric arguments must be positive.')
    if args.quick and args.resume:parser.error('--quick and --resume cannot be combined.')
    if args.resume and not args.resume.is_file():parser.error('Resume checkpoint does not exist.')
    os.environ['OMP_NUM_THREADS']=str(args.threads)
    os.environ['MKL_NUM_THREADS']=str(args.threads)
    import torch
    import yaml
    from ultralytics import YOLO
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(min(args.threads,4))
    os.chdir(ROOT)
    dataset=ROOT/'data'
    config={'path':dataset.as_posix(),'train':'images/train','val':'images/val','test':'images/test','names':{0:'fire',1:'smoke'}}
    (dataset/'data.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    if args.resume:
        model=YOLO(str(args.resume.resolve()))
        configure_cpu_threads(model,args.threads)
        model.train(resume=True,device='cpu',workers=0,amp=False,cache=False)
    else:
        model=YOLO('yolo11s.pt')
        configure_cpu_threads(model,args.threads)
        run_name=('quick_' if args.quick else 'train_')+datetime.now().strftime('%Y%m%d_%H%M%S')
        model.train(data=str(dataset/'data.yaml'),device='cpu',
                    epochs=1 if args.quick else args.epochs,
                    imgsz=320 if args.quick else args.imgsz,
                    batch=2 if args.quick else args.batch,
                    fraction=0.1 if args.quick else 1.0,
                    workers=0,cache=False,amp=False,patience=15,seed=42,
                    project=str(ROOT/'runs'),name=run_name,exist_ok=False,plots=True)
    run_dir=Path(model.trainer.save_dir)
    print('\nRun directory:',run_dir)
    if args.quick:
        print('Setup check completed. This checkpoint is NOT the final model. Next: python train.py')
    else:
        (ROOT/'models').mkdir(exist_ok=True)
        best=run_dir/'weights/best.pt'
        if not best.is_file():raise FileNotFoundError(f'Training completed without best.pt: {best}')
        shutil.copy2(best,ROOT/'models/best.pt')
        last=run_dir/'weights/last.pt'
        if last.is_file():shutil.copy2(last,ROOT/'models/last.pt')
        details={'run_directory':str(run_dir),'best_weights':str(ROOT/'models/best.pt'),'model_family':'YOLO11s','quick':False}
        (ROOT/'models/training_run.json').write_text(json.dumps(details,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Final weights:',ROOT/'models/best.pt')

if __name__=='__main__':
    main()
