import sys
import os
import argparse
import torch
import numpy as np
import onnx
import onnxruntime as ort

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from models import DeSPARNetwork

'''
python tools/verify_onnx.py --stage 1
'''

def parse_args():
    parser = argparse.ArgumentParser(description="Verify ONNX model structure and precision equivalence.")
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2], help='Model stage to verify (1 or 2)')
    parser.add_argument('--dataset', type=str, default='ORSI-4199', help='Dataset name')
    parser.add_argument('--class_num', type=int, default=11, help='Number of classes')
    return parser.parse_args()

def check_onnx_structure(onnx_path):
    """使用 ONNX 官方工具检查图结构是否合法"""
    print(f"[*] Checking ONNX structure: {onnx_path}")
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("[+] ONNX structure validation passed.")
    except Exception as e:
        print("[-] ONNX structure validation failed!")
        print(e)
        sys.exit(1)

def test_precision_equivalence(args, pth_path, onnx_path, center_path):
    """对比 PyTorch 和 ONNXRuntime 的输出绝对误差"""
    print("[*] Testing precision equivalence between PyTorch and ONNXRuntime...")
    
    # 1. 构造测试数据
    img_size = 352
    dummy_rgb = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    dummy_depth = np.random.randn(1, 1, img_size, img_size).astype(np.float32)
    dummy_label = np.array([0], dtype=np.int64)
    
    is_stage2 = (args.stage == 2)
    
    # 2. PyTorch 推理
    device = torch.device('cpu')
    model = DeSPARNetwork(
        use_semantic_prompt=is_stage2, 
        class_num=args.class_num,
        prompt_init_path=center_path if is_stage2 else None
    ).to(device)
    
    model.load_state_dict(torch.load(pth_path, map_location=device), strict=True)
    model.eval()
    
    with torch.no_grad():
        if is_stage2:
            pt_sal, _, _ = model(torch.from_numpy(dummy_rgb), torch.from_numpy(dummy_depth), batch_labels=torch.from_numpy(dummy_label))
        else:
            pt_sal, _, _ = model(torch.from_numpy(dummy_rgb), torch.from_numpy(dummy_depth))
    pt_sal_np = pt_sal.numpy()

    # 3. ONNXRuntime 推理
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {
        'rgb': dummy_rgb,
        'depth': dummy_depth
    }
    if is_stage2:
        ort_inputs['label'] = dummy_label
        
    ort_outs = ort_session.run(['sal_pred', 'edge_pred', 'feat_out'], ort_inputs)
    ort_sal_np = ort_outs[0]

    # 4. 误差对比
    try:
        np.testing.assert_allclose(pt_sal_np, ort_sal_np, rtol=1e-03, atol=1e-04)
        max_diff = np.max(np.abs(pt_sal_np - ort_sal_np))
        print(f"[+] Precision test passed! Maximum absolute difference: {max_diff:.6e}")
    except AssertionError as e:
        print("[-] Precision test failed. Outputs do not match.")
        print(e)
        sys.exit(1)

def main():
    args = parse_args()
    
    weight_dir = os.path.join(ROOT_DIR, f'weights/stage{args.stage}_{args.dataset}')
    pth_path = os.path.join(weight_dir, f'despar_stage{args.stage}_best.pth')
    onnx_path = os.path.join(weight_dir, f'despar_stage{args.stage}.onnx')
    center_path = os.path.join(ROOT_DIR, f'weights/prompt_centers_{args.dataset}.npy')
    
    print("-" * 50)
    print(f"DeSPAR ONNX Verification (Stage {args.stage})")
    print("-" * 50)
    
    if not os.path.exists(onnx_path):
        print(f"[-] Error: ONNX file not found at {onnx_path}")
        sys.exit(1)
    if not os.path.exists(pth_path):
        print(f"[-] Error: PyTorch weight not found at {pth_path}")
        sys.exit(1)
        
    check_onnx_structure(onnx_path)
    print("-" * 30)
    test_precision_equivalence(args, pth_path, onnx_path, center_path)
    print("-" * 50)

if __name__ == '__main__':
    main()