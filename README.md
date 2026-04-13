<div align="center">

# DeSPAR

[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-EE4C2C.svg?style=flat&logo=PyTorch)](https://pytorch.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Deployment-blue.svg)](https://onnx.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
<br>
[**English**](README_en.md) | **简体中文**

</div>

> **本项目为 IEEE TGRS 见刊论文 "[DeSPAR: Depth-Guided Semantic-Prompted Adaptive Refinement for ORSI Salient Object Detection](https://ieeexplore.ieee.org/document/11367015)" 的官方 PyTorch 实现与工程部署仓库。**


## 📖 项目简介

在光学遥感图像（ORSI）显著目标检测中，目前面临两大核心挑战：
1. **空间感知受限：** 仅依赖 RGB 模态在极端成像条件下（如图 a 和图 b），难以敏锐感知目标的几何凸起与空间布局，导致提取的目标空间结构表征不够鲁棒。
2. **语义特征混淆：** 遥感图像中目标与复杂背景的特征高度耦合，单模态方法极易发生类别混淆（如图 c 中，MRBINet 将纹理与形态相近的道路误检为显著建筑）。

<p align="center">
  <img src="assets/Motivation.png" alt="DeSPAR Motivation" width="60%">
</p>

为打破这一瓶颈，我们提出了 **DeSPAR** —— 一种创新的**几何–语义解耦渐进式精炼框架**。该框架针对性地引入**深度几何先验**以突破空间感知局限，并结合**类别语义提示**来彻底消除特征混淆。为了完美融合这两种异构信息，DeSPAR 摒弃了传统的联合训练，采用了渐进式的解耦特征提取策略，最终实现了极具鲁棒性的精准检测。

**性能表现：** DeSPAR 在 3 个公开的 ORSI-SOD 基准测试（ORSSD, EORSSD, ORSI-4199）中超越了 **26 种** SOTA 方法。


## 🧠 核心方法与架构

在同时引入深度几何先验与语义约束时，最直观的做法是采用端到端（End-to-End）联合训练。然而，这种方式存在极其严重的**优化冲突**：由于骨干网的预训练优势，强烈的语义分类信号会极快收敛并主导梯度的更新方向，导致网络“走捷径”而忽略脆弱的几何基础构建。

为了解决这一冲突，DeSPAR 巧妙地将特征学习过程拆分为两个渐进阶段：

<p align="center">
  <img src="assets/Pipline.png" alt="DeSPAR Architecture" width="90%">
</p>

* 💠 **高效的视觉骨干网：** 采用 PVTv2 作为编码器，利用其空间降维注意力机制，在有效提取全局上下文的同时，合理控制了高分辨率遥感图像的计算复杂度。

* 🧱 **Stage 1: 深度引导几何学习 (DGL)：** 专注于构建通用几何基础。通过新颖的轻量级**深度引导精炼器 (DGR)**，利用 RGB 特征引导伪深度去噪，并反向注入纯净的几何线索，极大增强了模型对目标空间结构的表征能力。
* 🎯 **Stage 2: 深度引导语义自适应精炼 (DSR)：** 专注于施加类别特定的语义约束。在继承 DGL 几何地基（权重）的前提下，借助构建的**语义提示库 (Semantic Prompt Bank)**，通过提示引导机制自适应地优化不同类别目标的特征，有效化解了形态差异带来的语义特征混淆。


## 🚀 落地部署与工程特性

本仓库不仅提供了学术复现代码，还专门针对**算法落地与端侧部署**进行了全面重构：

* **⚡ 极致轻量与高帧率：** 模型参数量仅为 **26.4M**，在原生 PyTorch 下推理速度可达 **161 FPS**，极具端侧落地潜力。

* **📦 ONNX 零门槛部署：** 完整支持动态轴的 ONNX 静态计算图导出，通过底层 `Monkey Patch` 解决 `pixel_unshuffle` 算子兼容性问题，并内置误差 `< 1e-4` 的双盲精度对齐测试。
* **🔗 鲁棒的级联推理引擎：** 推理 Demo 脚本内置单目深度估计接口（借力 Depth-Anything-V2）。即便在无深度图输入的盲测场景下，DeSPAR 依然能凭借 DGR 模块强大的去噪与抗干扰能力输出精准预测。


## 🛠️ 准备工作

### 1. 环境依赖
本项目基于 PyTorch 1.8+ 构建，核心依赖极简。请运行以下命令配置环境：
```bash
conda create -n despar python=3.8
conda activate despar
pip install -r requirements.txt
```


### 2. 数据与权重准备

为了保证极其丝滑的开箱体验，我们已将数据集的分类标签（`annotations/*.npy`）内置于本仓库中。您只需要下载图像文件与相关的模型权重：

1. **下载模型权重：** 
   * **DeSPAR 核心权重：** [下载 weights.zip](https://github.com/Lfping919/DeSPAR/archive/refs/tags/v1.0.0.zip)，下载完成后解压至项目根目录。
   * **Depth-Anything 权重 (仅盲测推理需要)：** [下载 depth_anything_v2_vitb.pth](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true)  并将其放置在 `Depth-Anything-V2/checkpoints/` 目录下。
2. **下载数据集：** 请从 [Link](https://pan.baidu.com/s/1HetiMuvt7D3Wdo4IuNu1ZQ?pwd=v7rm) 获取 `ORSI-4199` 的核心图像数据（包含 Image, GT, Depth）。
3. **合并目录：** 确保您的项目根目录结构如下所示：


<details>
<summary>📂 <b>点击展开查看项目目录结构</b></summary>

```text
DeSPAR/
├── assets/                 
│   └── examples/           # Demo 测试图像
├── configs/                # 配置文件 (stage1/stage2)
├── data/
│   └── ORSI-4199/          # ORSI-4199 数据集目录
│       ├── annotations/    # 类别标签文件 (已内置于 GitHub)
│       │   ├── ORSI-4199_train_cls.npy
│       │   └── ORSI-4199_test_cls.npy
│       ├── train_Image/    
│       ├── train_GT/       
│       ├── train_Depth/    
│       ├── test_Image/     
│       ├── test_GT/        
│       └── test_Depth/     
├── datasets/
├── Depth-Anything-V2/
│   └── checkpoints/
│       └── depth_anything_v2_vitb.pth  # 用于盲测推理的深度估计权重
├── models/                 # 核心网络架构
├── outputs/                # 推理结果存放目录 (深度图、掩码图、叠加图)
├── tools/                  # 训练、测试、推理脚本
├── weights/                # 预训练权重存放目录
│   ├── pvt_v2_b2.pth
│   ├── prompt_centers_ORSI-4199.npy
│   ├── stage1_ORSI-4199/
│   │   ├── despar_stage1_best.pth
│   │   └── despar_stage1.onnx     # 导出的 ONNX 部署模型
│   └── stage2_ORSI-4199/
│       ├── despar_stage2_best.pth
│       └── despar_stage2.onnx     # 导出的 ONNX 部署模型
└── requirements.txt
```
</details>

## 🚀 快速开始：推理与部署

我们提供了极其强大且开箱即用的 `demo.py` 推理脚本，内置了**智能路由**与**盲测级联**机制：
* **智能路由：** 当输入指定 `--label` 时，脚本将自动加载 **Stage 2 (DSR)** 模型进行语义自适应精准检测；若未提供标签，则自动回退加载 **Stage 1 (DGL)** 模型，进行类别无关的通用显著性检测。
* **无深度盲测：** 即使不提供 `--depth` 深度图输入，脚本也会无缝调用内置引擎（借力 Depth-Anything-V2）生成伪深度，完美应对真实业务场景。

### 1. 支持的语义类别字典
在使用 Stage 2 进行语义引导推理时，支持以下 11 种遥感显著目标类别：
`stadium` (体育场), `aircraft` (飞机), `road` (道路), `oil_tank` (油罐), `car` (汽车), `urban_landmark` (城市地标), `ship` (船舶), `river` (河流), `rural_building` (乡村建筑), `lake` (湖泊), `bridge` (桥梁)。

### 2. 单图级联推理体验
我们在 `assets/` 目录下为您预置了涵盖所有类别的测试图像。您可以直接复制以下命令体验极其精准的分割效果（脚本会自动保存预测结果与可视化叠图）：

**模式 A：语义引导精准检测 (调用 Stage 2 - 推荐)**
```bash
python tools/demo.py --img 'assets/examples/2013.jpg' --label aircraft
python tools/demo.py --img 'assets/examples/2198.jpg' --label road
python tools/demo.py --img 'assets/examples/cars_MSO_ (41).jpg' --label car
python tools/demo.py --img 'assets/examples/2229.jpg' --label ship
```
*(注：assets/ 目录下还包含了 stadium, oil_tank, river,  rural_building, urban_landmark, lake, bridge 等各类图像，您可自由替换上述命令进行全面体验。)*

**模式 B：类别无关/零样本检测 (调用 Stage 1)** 当您面对未知类别的遥感图像时，直接忽略标签参数，测试模型的通用泛化能力：
```bash
python tools/demo.py --img 'assets/2198.jpg'
```

**推理结果：** 脚本运行完成后，生成的深度图、显著性掩码以及可视化叠加图（原图 + 掩码）将自动分类保存在 `outputs/` 目录下。


### 3. ONNX 导出与精度验证
为了打通端侧部署的最后一公里，我们提供了标准的工业级导出与验证流程，完美支持 Stage 1 和 Stage 2 模型的动态轴导出。

**步骤 1：模型导出**
使用以下命令，将训练好的 PyTorch 权重无缝导出为 `.onnx` 计算图：
```bash
python tools/export_onnx.py --stage 1  # 导出 Stage 1 通用几何检测模型
python tools/export_onnx.py --stage 2  # 导出 Stage 2 语义引导精炼模型
```

**步骤 2：双盲精度对齐测试**
导出后，强烈建议使用我们提供的验证脚本，测试 PyTorch 与 ONNXRuntime 在相同输入下的输出误差，确保算子级对齐：
```bash
python tools/verify_onnx.py --stage 1
python tools/verify_onnx.py --stage 2
```
*在我们的测试环境中，PyTorch 与 ONNX 的最大绝对误差（Max Absolute Difference）均远低于 `1e-4` 级别，实现了完美的无损转换。*


## ⚙️ 训练与评估

本项目采用渐进式解耦训练范式，请严格按照以下三个步骤执行：

**Step 1: 训练 Stage 1 (DGL 几何地基构建)**
```bash
python tools/train.py --config configs/stage1_dgl.yaml
```

**Step 2: 提取语义提示中心 (关键步骤)**
利用 Stage 1 训练好的权重，提取数据集中各层级的语义聚类中心：
```bash
python tools/build_prompt_bank.py
```

**Step 3: 训练 Stage 2 (DSR 语义自适应精炼)**
```bash
python tools/train.py --config configs/stage2_dsr.yaml
```

**模型评估 (Evaluation)**
```bash
python tools/test.py --config configs/stage2_dsr.yaml
```

## 📊 定量评估结果

DeSPAR 在三大公开基准测试中均展现出了卓越的性能。为了验证模型的鲁棒性，我们进行了极其详实的对比实验：在经典的 **ORSSD 和 EORSSD** 数据集上，我们与 **26 种** 现有 SOTA 方法进行了对比；同时，在更具挑战性的大规模数据集 **ORSI-4199** 上，我们与 **17 种** 顶尖开源方法进行了全面评估，均取得了最先进的指标。

<p align="center">
  <img src="assets/result_table0.png" alt="Quantitative Results on ORSSD, EORSSD, and ORSI-4199" width="95%">
</p>

<p align="center">
  <img src="assets/result_table.png" alt="Quantitative Results on ORSSD, EORSSD, and ORSI-4199" width="50%">
</p>



## 🤝 引用

如果本项目对您的研究或工程落地有帮助，欢迎引用我们的论文，并给本仓库点个 ⭐ Star！

```bibtex
@article{zhang2026despar,
  title={DeSPAR: Depth-Guided Semantic-Prompted Adaptive Refinement for ORSI Salient Object Detection},
  author={Zhang, Xiaoli and Liufu, Ping and Hu, Xihang and Li, Xiongfei and Jia, Chuanmin and Ma, Siwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026},
  publisher={IEEE}
}
```


## 🙏 致谢

本项目的成功离不开开源社区的无私贡献。我们在此特别感谢以下优秀开源工作提供的基石与灵感：
* 💠 [PVTv2](https://github.com/whai362/PVT): 为本项目提供了极其高效的视觉骨干网络。
* 🌊 [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2): 为本项目的无深度图盲测推理演示提供了强大的单目深度估计支持。
