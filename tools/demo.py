import sys
import os
import argparse
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
DA_DIR = os.path.join(ROOT_DIR, 'Depth-Anything-V2')
sys.path.append(DA_DIR)

from models import DeSPARNetwork
try:
    from depth_anything_v2.dpt import DepthAnythingV2
except ImportError:
    print("[ERROR] Cannot import DepthAnythingV2. Please ensure the 'Depth-Anything-V2' folder is in the project root.")
    sys.exit(1)

CLASS_MAP = {
    'stadium': 0,
    'aircraft': 1,
    'road': 2,
    'oil_tank': 3,
    'car': 4,
    'urban_landmark': 5,
    'ship': 6,
    'river': 7,
    'rural_building': 8,
    'lake': 9,
    'bridge': 10,
}

'''
python tools/demo.py --img 'assets/examples/stadium_IC_24.jpg' --label stadium
python tools/demo.py --img 'assets/examples/2012.jpg' --label aircraft
python tools/demo.py --img 'assets/examples/2013.jpg' --label aircraft
python tools/demo.py --img 'assets/examples/2198.jpg' --label road
python tools/demo.py --img 'assets/examples/storage tank_IC_2.jpg' --label oil_tank
python tools/demo.py --img 'assets/examples/cars_MSO_ (41).jpg' --label car
python tools/demo.py --img 'assets/examples/urban landmarks_CSO_5.jpg' --label urban_landmark
python tools/demo.py --img 'assets/examples/2229.jpg' --label ship
python tools/demo.py --img 'assets/examples/rivers_CB_1.jpg' --label river
python tools/demo.py --img 'assets/examples/rural buildings_CB_  (8).jpg' --label rural_building
python tools/demo.py --img 'assets/examples/lake_IC_17.jpg' --label lake
python tools/demo.py --img 'assets/examples/bridges_MSO_ (2).jpg' --label bridge
'''

def parse_args():
    parser = argparse.ArgumentParser(description="DeSPAR End-to-End Inference Demo")
    
    parser.add_argument('--img', type=str, required=True, help='Path to the input RGB image')
    parser.add_argument('--out_dir', type=str, default='outputs', help='Directory to save results')
    parser.add_argument('--label', type=str, default=None, help=f'(Optional) Class name. Available: {list(CLASS_MAP.keys())}')
    
    parser.add_argument('--weight_s1', type=str, default='weights/stage1_ORSI-4199/despar_stage1_best.pth')
    parser.add_argument('--weight_s2', type=str, default='weights/stage2_ORSI-4199/despar_stage2_best.pth')
    parser.add_argument('--center', type=str, default='weights/prompt_centers_ORSI-4199.npy')
    parser.add_argument('--class_num', type=int, default=len(CLASS_MAP))
    parser.add_argument('--img_size', type=int, default=352)
    
    parser.add_argument('--da_encoder', type=str, default='vitb', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--da_weight', type=str, default='Depth-Anything-V2/checkpoints/depth_anything_v2_vitb.pth')
    
    return parser.parse_args()

def prepare_tensors(rgb_pil, depth_pil, size):
    orig_w, orig_h = rgb_pil.size
    
    rgb_resized = rgb_pil.resize((size, size), Image.BILINEAR)
    depth_resized = depth_pil.resize((size, size), Image.BILINEAR)
    
    transform_rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_depth = transforms.Compose([
        transforms.ToTensor()
    ])
    
    img_tensor = transform_rgb(rgb_resized).unsqueeze(0)
    depth_tensor = transform_depth(depth_resized).unsqueeze(0)
    
    return img_tensor, depth_tensor, (orig_w, orig_h)

def overlay_mask_on_rgb(rgb_pil, mask_np, color=(255, 0, 0), alpha=0.6):
    rgb_np = np.array(rgb_pil).astype(np.float32)
    color_mask = np.zeros_like(rgb_np)
    color_mask[:, :, 0], color_mask[:, :, 1], color_mask[:, :, 2] = color
    
    mask_3d = np.expand_dims(mask_np, axis=2)
    blended = rgb_np * (1 - mask_3d * alpha) + color_mask * (mask_3d * alpha)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("-" * 60)
    print("[INFO] DeSPAR End-to-End Inference Engine")
    print(f"[INFO] Input Image: {args.img}")
    
    if not os.path.exists(args.img):
        print(f"[ERROR] Cannot find input image at {args.img}")
        sys.exit(1)


    # 1. 动态判断使用哪个阶段的模型
    label_id = None
    is_stage2 = False
    
    if args.label is not None:
        args.label = args.label.lower()
        if args.label not in CLASS_MAP:
            print(f"[ERROR] Unknown label: '{args.label}'. Supported: {list(CLASS_MAP.keys())}")
            sys.exit(1)
        label_id = CLASS_MAP[args.label]
        is_stage2 = True
        active_weight = args.weight_s2
        print(f"[INFO] Label provided: '{args.label}' -> Activating Stage 2 Model.")
    else:
        active_weight = args.weight_s1
        print("[INFO] No label provided -> Activating Stage 1 (Base) Model.")
    print("-" * 60)


    # 2. 生成深度图 
    print("[INFO] Initializing Depth-Anything-V2 module...")
    da_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    depth_anything = DepthAnythingV2(**da_configs[args.da_encoder])
    
    da_weight_path = os.path.join(ROOT_DIR, args.da_weight)
    if not os.path.exists(da_weight_path):
        print(f"[ERROR] Cannot find DepthAnything weight at {da_weight_path}")
        sys.exit(1)
        
    depth_anything.load_state_dict(torch.load(da_weight_path, map_location='cpu'))
    depth_anything = depth_anything.to(device).eval()

    print("[INFO] Estimating depth...")
    raw_image_cv = cv2.imread(args.img)
    with torch.no_grad():
        depth_np = depth_anything.infer_image(raw_image_cv, input_size=518)
    
    depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8) * 255.0
    depth_uint8 = depth_norm.astype(np.uint8)
    
    base_name = os.path.basename(args.img).split('.')[0]
    depth_save_path = os.path.join(args.out_dir, 'depths',  f"{base_name}.png")
    os.makedirs(os.path.dirname(depth_save_path), exist_ok=True)
    cv2.imwrite(depth_save_path, depth_uint8)
    print(f"[INFO] Depth map generated and saved to: {depth_save_path}")

    # 清理显存 (将 DA2 移出以腾出空间给 DeSPAR)
    del depth_anything
    torch.cuda.empty_cache()


    # 3. 准备 DeSPAR 输入
    rgb_pil = Image.fromarray(cv2.cvtColor(raw_image_cv, cv2.COLOR_BGR2RGB))
    depth_pil = Image.fromarray(depth_uint8, mode='L')
    
    img_tensor, depth_tensor, (orig_w, orig_h) = prepare_tensors(rgb_pil, depth_pil, args.img_size)
    img_tensor, depth_tensor = img_tensor.to(device), depth_tensor.to(device)


    # 4. DeSPAR 推理
    print(f"[INFO] Initializing DeSPAR Network...")
    model = DeSPARNetwork(
        use_semantic_prompt=is_stage2, 
        class_num=args.class_num,
        prompt_init_path=os.path.join(ROOT_DIR, args.center) if is_stage2 else None
    ).to(device)
    
    weight_path = os.path.join(ROOT_DIR, active_weight)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    print("[INFO] Running Segmentation...")
    start_time = time.perf_counter()
    with torch.no_grad():
        lbl_tensor = torch.tensor([label_id], dtype=torch.long).to(device) if is_stage2 else None
        preds, _, _ = model(img_tensor, depth_tensor, batch_labels=lbl_tensor)
        
        preds = F.interpolate(preds, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        pred_mask = preds.sigmoid().squeeze().cpu().numpy()
        
    infer_time = (time.perf_counter() - start_time) * 1000
    print(f"[INFO] Inference successful! Time cost: {infer_time:.2f} ms")


    # 5. 结果保存
    pred_mask_norm = (pred_mask - pred_mask.min()) / (pred_mask.max() - pred_mask.min() + 1e-8)
    
    # 保存原始黑白 Mask
    mask_save_path = os.path.join(args.out_dir, 'masks', f"{base_name}.png")
    os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
    Image.fromarray((pred_mask_norm * 255).astype(np.uint8), mode='L').save(mask_save_path)
    
    # 保存半透明红膜叠加图
    blended_save_path = os.path.join(args.out_dir, 'blendeds', f"{base_name}.jpg")
    os.makedirs(os.path.dirname(blended_save_path), exist_ok=True)
    blended_img = overlay_mask_on_rgb(rgb_pil, pred_mask_norm, color=(255, 50, 50), alpha=0.6)
    blended_img.save(blended_save_path)
    
    print(f"[INFO] Raw Mask saved to: {mask_save_path}")
    print(f"[INFO] Blended Visualization saved to: {blended_save_path}")
    print("-" * 60)
    print("[INFO] Process Complete.")

if __name__ == '__main__':
    main()