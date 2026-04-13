import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import argparse
import yaml
import logging
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from models import DeSPARNetwork
from datasets import build_dataloader
from utils import get_param_groups, cosine_scheduler, SODLoss, PromptDiversityLoss, CosineContrastLoss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger(log_dir, stage):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'stage{stage}_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logger = logging.getLogger('DeSPAR_Train')
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] <%(filename)s:%(lineno)d> %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


'''
python tools/train.py --config configs/stage1_dgl.yaml
python tools/train.py --config configs/stage2_dsr.yaml
'''
def parse_config():
    parser = argparse.ArgumentParser(description="Unified Training Pipeline for DeSPAR")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    args = parser.parse_args()
    
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
        
    class Config:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)
    return Config(cfg)


def train_one_epoch(model, train_loader, optimizer, device, losses_dict, cfg, epoch):
    model.train()
    running_loss = 0.0
    is_stage2 = (cfg.stage == 2)
    
    sod_loss = losses_dict['sod']
    contrast_loss = losses_dict.get('contrast', None)
    div_loss = losses_dict.get('div', None)

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{cfg.epochs + cfg.warmup_epochs}] Training")
    for batch in pbar:
        imgs = batch['image'].to(device)
        depths = batch['depth'].to(device)
        gts = batch['gt'].to(device)
        labels = batch['label'].to(device) if is_stage2 else None
        
        optimizer.zero_grad()
        preds, _, x4 = model(imgs, depths, batch_labels=labels)
        
        loss = sod_loss(preds, gts)
        
        if is_stage2:
            x4_gap = F.normalize(F.adaptive_avg_pool2d(x4.detach(), (1, 1)).flatten(1), p=2, dim=1)
            l_con = contrast_loss(x4_gap, labels, model.prompt_bank.prompts)
            l_div = div_loss(model.prompt_bank.prompts)
            loss = loss + cfg.lc * l_con + cfg.ld * l_div
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
    return running_loss / len(train_loader)

def validate(model, val_loader, device, cfg):
    model.eval()
    total_mae = 0.0
    is_stage2 = (cfg.stage == 2)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            imgs = batch['image'].to(device)
            depths = batch['depth'].to(device)
            gts = batch['gt'].to(device)
            labels = batch['label'].to(device) if is_stage2 else None
            
            preds, _, _ = model(imgs, depths, batch_labels=labels)
            preds = F.interpolate(preds, size=(gts.shape[2], gts.shape[3]), mode='bilinear', align_corners=False)
            
            preds_prob = preds.sigmoid()
            batch_mae = torch.abs(preds_prob - gts).mean().item()
            total_mae += batch_mae
            
    return total_mae / len(val_loader)


def main():
    set_seed(42)
    
    cfg = parse_config()
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ds_root = os.path.join(ROOT_DIR, cfg.data_root, cfg.dataset)
    cls_path_train = os.path.join(ds_root, 'annotations', f'{cfg.dataset}_train_cls.npy')
    cls_path_test = os.path.join(ds_root, 'annotations', f'{cfg.dataset}_test_cls.npy')
    center_path = os.path.join(ROOT_DIR, cfg.weights_dir, f'prompt_centers_{cfg.dataset}.npy')
    stage1_weight = os.path.join(ROOT_DIR, cfg.weights_dir, f'stage1_{cfg.dataset}/despar_stage1_best.pth')
    save_dir = os.path.join(ROOT_DIR, cfg.weights_dir, f'stage{cfg.stage}_{cfg.dataset}')
    
    logger = setup_logger(save_dir, cfg.stage)
    writer = SummaryWriter(os.path.join(save_dir, 'tb_logs'))
    
    logger.info("--------------------------------------------------")
    logger.info(f"Starting DeSPAR Stage {cfg.stage} Training")
    logger.info(f"Dataset: {cfg.dataset} | Batch Size: {cfg.batch_size} | Epochs: {cfg.epochs + cfg.warmup_epochs}")
    logger.info("--------------------------------------------------")
    
    train_loader = build_dataloader(
        data_root=ds_root, mode='train', 
        cls_path=cls_path_train if cfg.stage == 2 else None,
        img_size=cfg.img_size, batch_size=cfg.batch_size
    )
    val_loader = build_dataloader(
        data_root=ds_root, mode='test', 
        cls_path=cls_path_test if cfg.stage == 2 else None,
        img_size=cfg.img_size
    )

    is_stage2 = (cfg.stage == 2)
    model = DeSPARNetwork(
        use_semantic_prompt=is_stage2, 
        class_num=cfg.class_num,
        prompt_init_path=center_path if is_stage2 else None
    ).to(device)
    
    if is_stage2:
        logger.info(f"Loading Stage 1 weights from: {stage1_weight}")
        checkpoint = torch.load(stage1_weight, map_location=device)
        model.load_state_dict(checkpoint, strict=False)

    pretrained_params, newly_added_params = get_param_groups(model)
    optimizer = torch.optim.AdamW([
        {"params": pretrained_params, "lr": cfg.base_lr},
        {"params": newly_added_params, "lr": cfg.base_lr * 2.0}
    ], weight_decay=1e-2)
    
    lr_schedule = cosine_scheduler(
        base_lr=cfg.base_lr, final_lr=cfg.final_lr, 
        total_epochs=cfg.epochs + cfg.warmup_epochs, warmup_epochs=cfg.warmup_epochs
    )

    losses_dict = {'sod': SODLoss()}
    if is_stage2:
        losses_dict['contrast'] = CosineContrastLoss(margin=0.5)
        losses_dict['div'] = PromptDiversityLoss(margin=0.5)

    best_val_mae = float('inf')
    
    for epoch in range(1, cfg.epochs + cfg.warmup_epochs + 1):
        if cfg.stage == 1:
            optimizer.param_groups[0]['lr'] = lr_schedule[epoch - 1]
            optimizer.param_groups[1]['lr'] = lr_schedule[epoch - 1] * 2.0
        else:
            optimizer.param_groups[0]['lr'] = lr_schedule[epoch - 1] * 0.1 
            optimizer.param_groups[1]['lr'] = lr_schedule[epoch - 1]
        
        train_loss = train_one_epoch(model, train_loader, optimizer, device, losses_dict, cfg, epoch)
        val_mae = validate(model, val_loader, device, cfg)
        
        logger.info(f"[Epoch {epoch}/{cfg.epochs + cfg.warmup_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_mae:.4f}")
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Metric/Val_MAE', val_mae, epoch)
        writer.add_scalar('LR/Backbone', optimizer.param_groups[0]['lr'], epoch)
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_path = os.path.join(save_dir, f'despar_stage{cfg.stage}_best.pth')
            torch.save(model.state_dict(), best_path)
            logger.info(f"[*] New Best Val Loss! Model saved to {best_path}")

    writer.close()
    logger.info("Training Completed Successfully.")

if __name__ == '__main__':
    main()

