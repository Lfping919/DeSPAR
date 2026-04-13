import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
import torch.nn.functional as F
from models import DeSPARNetwork


# [Hack] 暴力替换 ONNX 不支持的 pixel_unshuffle 算子
def custom_pixel_unshuffle(x, downscale_factor):
    """用最基础的 view 和 permute 重构 pixel_unshuffle 数学逻辑"""
    b, c, h, w = x.shape
    r = downscale_factor
    x = x.view(b, c, h // r, r, w // r, r)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(b, c * (r ** 2), h // r, w // r)
    return x

# Monkey Patch 掉包：强行覆盖 PyTorch 底层函数
F.pixel_unshuffle = custom_pixel_unshuffle
torch.pixel_unshuffle = custom_pixel_unshuffle
if hasattr(torch.nn, 'PixelUnshuffle'):
    torch.nn.PixelUnshuffle.forward = lambda self, x: custom_pixel_unshuffle(x, self.downscale_factor)
# =====================================================================

'''
python tools/export_onnx.py --stage 2 --opset 11 
'''
def parse_args():
    parser = argparse.ArgumentParser(description="Export DeSPAR to ONNX format")
    
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2], help='Model stage to export')
    parser.add_argument('--dataset', type=str, default='ORSI-4199', help='Dataset name for dynamic routing')
    parser.add_argument('--class_num', type=int, default=11, help='Number of classes')
    
    parser.add_argument('--weight', type=str, default=None, help='Path to weight (Auto-inferred if None)')
    parser.add_argument('--center', type=str, default='weights/prompt_centers_ORSI-4199.npy')
    parser.add_argument('--out_file', type=str, default=None, help='Output ONNX path (Auto-inferred if None)')
    
    parser.add_argument('--img_size', type=int, default=352, help='Input image size')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version (11 or 12 is recommended)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cpu') 
    
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    if args.weight is None:
        args.weight = os.path.join(ROOT_DIR, f'weights/stage{args.stage}_{args.dataset}/despar_stage{args.stage}_best.pth')
        
    if args.out_file is None:
        weight_dir = os.path.dirname(args.weight)
        args.out_file = os.path.join(weight_dir, f'despar_stage{args.stage}.onnx')
    
    print("-" * 60)
    print(f"[INFO] Exporting DeSPAR Stage {args.stage} to ONNX...")
    print(f"[INFO] Target Dataset : {args.dataset}")
    print(f"[INFO] Using Weight   : {args.weight}")
    print(f"[INFO] Output Path    : {args.out_file}")
    

    # 1. 实例化模型并加载权重
    is_stage2 = (args.stage == 2)
    model = DeSPARNetwork(
        use_semantic_prompt=is_stage2, 
        class_num=args.class_num,
        prompt_init_path=os.path.join(ROOT_DIR, args.center) if is_stage2 and args.center else None
    ).to(device)

    checkpoint = torch.load(args.weight, map_location=device)
    if 'module.' in list(checkpoint.keys())[0]:
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    # 2. 构造 Dummy Input 
    print(f"[INFO] Generating dummy inputs (Size: {args.img_size}x{args.img_size})...")
    dummy_rgb = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    dummy_depth = torch.randn(1, 1, args.img_size, args.img_size, device=device)
    
    dynamic_axes_config = {
        'rgb': {0: 'batch_size'}, 
        'depth': {0: 'batch_size'},
        'sal_pred': {0: 'batch_size'},
        'edge_pred': {0: 'batch_size'},
        'feat_out': {0: 'batch_size'}
    }

    if is_stage2:
        dummy_label = torch.tensor([0], dtype=torch.long, device=device)
        export_inputs = (dummy_rgb, dummy_depth, dummy_label)
        input_names = ['rgb', 'depth', 'label']
        dynamic_axes_config['label'] = {0: 'batch_size'}
    else:
        export_inputs = (dummy_rgb, dummy_depth)
        input_names = ['rgb', 'depth']

    output_names = ['sal_pred', 'edge_pred', 'feat_out']


    # 3. 执行 ONNX 导出
    print("[INFO] Tracing computational graph and exporting to ONNX...")
    output_dir = os.path.dirname(args.out_file)
    if output_dir: 
        os.makedirs(output_dir, exist_ok=True)
        
    try:
        torch.onnx.export(
            model,
            export_inputs,
            args.out_file,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_config
        )
        print(f"[SUCCESS] ONNX model successfully exported to: {args.out_file}")
    except Exception as e:
        print(f"[ERROR] ONNX export failed. Error details:\n{e}")

    print("-" * 60)

if __name__ == '__main__':
    main()