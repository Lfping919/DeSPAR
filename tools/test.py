import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import yaml
import time
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F

import py_sod_metrics
from thop import profile

from models import DeSPARNetwork
from datasets import build_dataloader

'''
python tools/test.py --config configs/stage1_dgl.yaml
python tools/test.py --config configs/stage2_dsr.yaml
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Industrial Evaluation Pipeline for DeSPAR")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--save_mask', action='store_true', help='If added, will save prediction masks to disk')
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    class Config:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)
        def set_flag(self, flag):
            self.save_mask = flag
            
    config_obj = Config(cfg)
    config_obj.set_flag(args.save_mask)
    return config_obj

def format_ascii_table(data, headers):
    col0_width = max([len(str(item[0])) for item in data] + [len(headers[0])]) + 2
    col1_width = max([len(str(item[1])) for item in data] + [len(headers[1])]) + 2
    
    separator = "+" + "-" * (col0_width + 2) + "+" + "-" * (col1_width + 2) + "+"
    
    lines = [separator]
    # Header
    lines.append(f"| {headers[0].ljust(col0_width)} | {headers[1].rjust(col1_width)} |")
    lines.append(separator)
    # Data rows
    for row in data:
        lines.append(f"| {str(row[0]).ljust(col0_width)} | {str(row[1]).rjust(col1_width)} |")
    lines.append(separator)
    
    return "\n".join(lines)

def main():
    cfg = parse_args()
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ds_root = os.path.join(ROOT_DIR, cfg.data_root, cfg.dataset)
    cls_path = os.path.join(ds_root, 'annotations', f'{cfg.dataset}_test_cls.npy')
    center_path = os.path.join(ROOT_DIR, cfg.weights_dir, f'prompt_centers_{cfg.dataset}.npy')
    best_weight = os.path.join(ROOT_DIR, cfg.weights_dir, f'stage{cfg.stage}_{cfg.dataset}', f'despar_stage{cfg.stage}_best.pth')

    
    if cfg.save_mask:
        save_dir = os.path.join(ROOT_DIR, 'predictions', f'stage{cfg.stage}_{cfg.dataset}')
        os.makedirs(save_dir, exist_ok=True)
    
    print("-" * 60)
    print(f"[INFO] Starting DeSPAR Stage {cfg.stage} Evaluation on {cfg.dataset}")
    print(f"[INFO] Loading weights from: {best_weight}")
    print("-" * 60)
    
    test_loader = build_dataloader(
        data_root=ds_root, mode='test', 
        cls_path=cls_path if cfg.stage == 2 else None,
        img_size=cfg.img_size
    )

    is_stage2 = (cfg.stage == 2)
    model = DeSPARNetwork(
        use_semantic_prompt=is_stage2, 
        class_num=cfg.class_num,
        prompt_init_path=center_path if is_stage2 else None
    ).to(device)
    
    checkpoint = torch.load(best_weight, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    FM = py_sod_metrics.Fmeasure()
    SM = py_sod_metrics.Smeasure()
    EM = py_sod_metrics.Emeasure()
    MAE = py_sod_metrics.MAE()
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []


    # 1. 计算 MACs, FLOPs 和 Params
    print("[INFO] Profiling Model Complexity (MACs/FLOPs/Params)...")
    dummy_img = torch.randn(1, 3, cfg.img_size, cfg.img_size).to(device)
    dummy_dep = torch.randn(1, 1, cfg.img_size, cfg.img_size).to(device)
    dummy_lbl = torch.tensor([0]).to(device) if is_stage2 else None
    
    with torch.no_grad():
        if is_stage2:
            macs, params = profile(model, inputs=(dummy_img, dummy_dep, dummy_lbl), verbose=False)
        else:
            macs, params = profile(model, inputs=(dummy_img, dummy_dep), verbose=False)
        
        flops = macs * 2.0


    # 2. GPU 预热与显存状态重置
    print("[INFO] Warming up GPU...")
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device) # 清空显存峰值记录器
        
    with torch.no_grad():
        for _ in range(10):
            model(dummy_img, dummy_dep, batch_labels=dummy_lbl)
    if device.type == 'cuda':
        torch.cuda.synchronize()


    # 3. 正式推理与评测
    print("[INFO] Running Inference & Evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            imgs = batch['image'].to(device)
            depths = batch['depth'].to(device)
            gts = batch['gt'] 
            orig_h, orig_w = gts.shape[2], gts.shape[3]
            
            labels = batch['label'].to(device) if is_stage2 else None
            
            if device.type == 'cuda':
                starter.record()
                
            preds, _, _ = model(imgs, depths, batch_labels=labels)
            
            if device.type == 'cuda':
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings.append(curr_time)
            
            preds = F.interpolate(preds, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            pred_npy = preds.sigmoid().squeeze().cpu().numpy()
            
            pred_npy = (pred_npy - pred_npy.min()) / (pred_npy.max() - pred_npy.min() + 1e-8)
            pred_img_npy = (pred_npy * 255).astype(np.uint8)
            
            gt_npy = gts.squeeze().cpu().numpy()
            gt_img_npy = (gt_npy * 255).astype(np.uint8)
            
            FM.step(pred=pred_img_npy, gt=gt_img_npy)
            SM.step(pred=pred_img_npy, gt=gt_img_npy)
            EM.step(pred=pred_img_npy, gt=gt_img_npy)
            MAE.step(pred=pred_img_npy, gt=gt_img_npy)
            
            if cfg.save_mask:
                file_name = batch['file_name'][0]
                if file_name.endswith('.jpg'):
                    file_name = file_name.replace('.jpg', '.png')
                Image.fromarray(pred_img_npy, mode='L').save(os.path.join(save_dir, file_name))


    # 4. 汇总
    avg_time_ms = np.mean(timings[1:]) if device.type == 'cuda' else 0
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0
    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0
    
    sm = SM.get_results()['sm']
    fm = FM.get_results()['fm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']
    

    results_data = [
        ["==== Accuracy Metrics ====", "==== Value ===="],
        ["S-measure (S_alpha)", f"{sm:.4f}"],
        ["F-measure Max", f"{fm['curve'].max():.4f}"],
        ["F-measure Mean", f"{fm['curve'].mean():.4f}"],
        ["F-measure Adp", f"{fm['adp']:.4f}"],
        ["E-measure Max", f"{em['curve'].max():.4f}"],
        ["E-measure Mean", f"{em['curve'].mean():.4f}"],
        ["E-measure Adp", f"{em['adp']:.4f}"],
        ["MAE (M)", f"{mae:.4f}"],
        ["==== Efficiency Metrics ====", "==== Value ===="],
        ["Params (M)", f"{params / 1e6:.4f}"],
        ["FLOPs (G)", f"{macs / 1e9:.3f}"], 
        ["Peak VRAM (MB)", f"{peak_vram:.2f}"],
        ["Inference FPS", f"{fps:.2f}"]
    ]
    
    table_str = format_ascii_table(results_data, headers=["Metric", "Score"])
    
    print("\n" + table_str + "\n")
    
    report_file = os.path.join(ROOT_DIR, cfg.weights_dir, f'stage{cfg.stage}_{cfg.dataset}', 'evaluation_report.txt')
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(f"Dataset: {cfg.dataset} | Stage: {cfg.stage} | Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(table_str + "\n")
        
    print(f"[INFO] Evaluation report saved to: {report_file}")
    if cfg.save_mask:
        print(f"[INFO] Prediction masks saved to: {save_dir}")

if __name__ == '__main__':
    main()
    