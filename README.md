# 4KLSDB: 4K Large Scale Dataset and Benchmark

High-resolution datasets are essential for advancing super-resolution (SR) and text-to-image (T2I) diffusion research. However, existing public collections lack both the native 4K resolution and the scale needed to train today’s state-of-the-art models. **4KLSDB** addresses this gap with a massive, diverse corpus of 4K images plus ready-to-use benchmarks.

---

## 📦 Dataset Overview

- **Total images**: 129,484  
- **Validation set**: 2,000 images  
- **Test set**: 1,984 images  

### Categories
- Nature  
- Urban scenes  
- People  
- Food  
- Artwork  
- CGI  

All images are natively 4K (minimum 3840×2160) and carefully balanced across categories.

---

## 🛠 Data Sources & Curation

1. **Sources**  
   - Photo Concept Bucket  
   - Laion-2B  
   - PD12M  

2. **Filtering & Annotation Pipeline**  
   - **Automated filtering** for resolution, aspect ratio, and duplicate removal  
   - **Aesthetic scoring** with Large Multimodal Models (LMMs)  
   - **Human vetting** to ensure high visual quality and consistency  

---

## 🚀 Benchmarks & Baselines

We demonstrate 4KLSDB’s utility by training representative SR and diffusion models:

- **Super-Resolution**: Trained on true 4K data, showing significant gains in PSNR/SSIM over 2K-trained baselines.  
- **T2I Diffusion**: Fine-tuned on 4KLSDB prompts, yielding higher fidelity in ultra-high-res image synthesis.  

_Comprehensive experiments reveal a clear positive correlation between native 4K training data and improved restoration/synthesis quality._

---
** Finetune repositories 

OSEDiff Super-Resolution
Finetune OSEDiff on custom SR datasets
🔗 https://github.com/cswry/OSEDiff.git

SwinIR x4/x8/x16
Single-scale SwinIR training scripts
🔗 https://github.com/cszn/KAIR.git

Diffusion-4K-α
4K text-to-image diffusion baseline
🔗 [https://github.com/YourUser/diffusion-4k-alpha](https://github.com/zhang0jhon/diffusion-4k.git)

## 📥 Getting Started

```bash
git clone https://github.com/SingleBicycle/4KLSDB_NIPS.git
cd 4KLSDB_NIPS
# Follow instructions in `DATA.md` to download and prepare the 4KLSDB dataset.
