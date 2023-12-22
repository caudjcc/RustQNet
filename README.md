# RustQNet，a DL-based quantitative architecture for multimodal image, designed for pixel-level quantitative inversion of wheat stripe rust (WSR) disease index (DI)

## This site is still under construction....

Code for paper: **Accurate and Large-scale Quantitative Inversion of Wheat Stripe Rust Disease Index: A Novel Multimodal UAV Dataset and Deep Learning Benchmark [[Arxiv]](https://scholar.google.com.hk/citations?user=mfrNGLoAAAAJ&hl=zh-CN)**

## Overview
<p align="center">
    <img src="pic/RustQNet.png" width="70%" /> <br />
    <em> 
   Figure 1: Overview of RustQNet. RustQNet architecture is designed for intelligent interpretation of multi-modal remote sensing data. In this architecture, the RGB modality exhibits high spatial resolution characteristics, while the MS and VI modalities possess high spectral resolution characteristics. The abbreviation "MI" refers to the mutual information minimization module, "Conv3×3" represents a 3×3 convolutional layer, and "fc" represents a fully connected layer.
    </em>
</p>

## Usage

1. Requirements
   
   - Python 3.9
   - PyTorch 1.12.1
   - Cuda 11.8
2. Dataset structure
   
<p align="center">
    <img src="pic/Dataset structure_1.png" width="50%" height="50%" /> <br />
    <em> 
    Figure 2: structure of Dataset. 
    </em>
</p>

<p align="center">
    <img src="pic/Dataset structure_2.png" width="30%" height="30%" /> <br />
    <em> 
    Figure 3: structure of txt list. 
    </em>
</p>
<p align="center">

3. Training
   - The training of entire RustQNet utilized one NVIDIA RTX 4090 GPU to accelerate.
     - run  `python tools/train.py` in terminal
   - (PS: The code of this project theoretically supports single-card and multi-card parallelism. However, this research has only been run on a single card. Multi-card parallelism still needs debugging and slight modifications.)

4. Testing
   - Run `python tools/test.py` in the terminal.
## Reference
The code of this project mainly refers to **HRNet-Semantic-Segmentation-HRNet-OCR [[CODE]](https://github.com/HRNet/HRNet-Semantic-Segmentation)** and **RGB-D Saliency Detection via Cascaded Mutual Information Minimization [[CODE]](https://github.com/JingZhang617/cascaded_rgbd_sod)**
   



