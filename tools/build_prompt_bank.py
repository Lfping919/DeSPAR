import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import DeSPARNetwork
from datasets import build_dataloader

'''
python tools/build_prompt_bank.py
'''

def parse_args():
    data_name = "ORSI-4199" # "ORSSD" "EORSSD"
    parser = argparse.ArgumentParser(description="Build Semantic Prompt Bank for Stage 2")
    parser.add_argument('--stage1_weight', type=str, default=os.path.join(ROOT_DIR, f'weights/stage1_{data_name}/despar_stage1_best.pth'), 
                        help='Path to trained Stage 1 weights')

    parser.add_argument('--data_root', type=str, default=os.path.join(ROOT_DIR, f'data/{data_name}'), help='Dataset root path')
    parser.add_argument('--cls_path', type=str, default=os.path.join(ROOT_DIR, f'data/{data_name}/annotations/{data_name}_train_cls.npy'), 
                        help='Path to class labels')
    parser.add_argument('--class_num', type=int, default=11, help='Number of categories in the dataset')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')

    parser.add_argument('--out_path', type=str, default=os.path.join(ROOT_DIR, f'weights/prompt_centers_{data_name}.npy'), 
                        help='Output path for the semantic centers')
    return parser.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"[Step 1] Loading Stage 1 Model from {args.stage1_weight} ...")
    model = DeSPARNetwork(use_semantic_prompt=False).to(device)
    
    checkpoint = torch.load(args.stage1_weight, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    print(f"[Step 2] Building DataLoader (Offline extraction, No Augmentation) ...")
    dataloader = build_dataloader(
        data_root=args.data_root,
        cls_path=args.cls_path,
        mode='train',
        apply_aug=False,
        batch_size=args.batch_size,
        shuffle=False
    )

    print(f"[Step 3] Extracting high-level features (x4) ...")
    all_feats = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Extracting"):
        imgs = batch['image'].to(device)
        depths = batch['depth'].to(device)
        labels = batch['label'].numpy() # [B]
        
        _, _, x4 = model(imgs, depths)  # x4: [B, 512, 11, 11]
        
        x4_gap = F.adaptive_avg_pool2d(x4, (1, 1)).flatten(1) # [B, 512]
        
        x4_norm = F.normalize(x4_gap, p=2, dim=1)
        
        all_feats.append(x4_norm.cpu().numpy())
        all_labels.extend(labels)
        
    all_feats = np.concatenate(all_feats, axis=0) # [N, 512]
    all_labels = np.array(all_labels)

    print(f"[Step 4] Calculating class centers (Mean) for {args.class_num} classes ...")
    centers = []
    for k in range(args.class_num):
        cls_mask = (all_labels == k)
        if not np.any(cls_mask):
            print(f"⚠️ Warning: No samples found for class {k}!")
            cls_center = np.random.randn(512) * 0.01 
        else:
            cls_samples = all_feats[cls_mask] # [N_k, 512]
            cls_center = np.mean(cls_samples, axis=0) # 按列取均值
        centers.append(cls_center)
        
    centers = np.array(centers) # [K, 512]
    
    print(f"[Step 5] Saving Semantic Prompt Bank to {args.out_path} ...")
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    np.save(args.out_path, centers)
    
    print(f"Finished! Extracted features shape: {centers.shape} (Expected: [{args.class_num}, 512])")

if __name__ == '__main__':
    main()
