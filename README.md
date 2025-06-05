# Brainfound: A Foundation Model for Multi-modal Brain Medical Imaging

**Brainfound** is a large-scale, multi-modal foundation model for brain medical imaging. It is pre-trained on over **3 million brain CT images** and **7 million brain MRI images**, along with their corresponding radiology reports. By integrating advanced AI-generated content alignment techniques, Brainfound enables powerful visual-text understanding for a wide range of clinical tasks.

## 🌟 Highlights

- **Unified foundation model** trained on large-scale paired brain CT and MRI datasets.
- Supports **multi-modal input/output** (image-to-text, text-to-image, cross-modal reasoning).

## ✅ Code TODO

The detailed usage instructions are provided in each task-specific directory.

### Pretrain Tasks

- [x] stage1: Diffusion-Based Visual Pretraining
- [x] stage2: Contrastive Vision-Language Pretraining 
- [x] Stage3: Multi-modal Fine-tuning with Generated Data

### Downstream Tasks

- [ ] Brain hemorrhage segmentation
- [ ] Brain midline shift segmentation
- [ ] MRI modality translation
- [ ] MRI image quality enhancement
- [ ] Automatic radiology report generation
- [ ] Visual-language dialogue QA
- [ ] Multiple-choice question (MCQ) answering (zero-shot)

---

## 🗂️ Repository Structure

- `pretrain_code/` — Pretraining pipeline for Brainfound  
  - `stage1/` — Data preparation for large-scale CT/MRI and paired reports  
  - `stage2/` — Vision-language model architecture  
  - `stage3/` — Pretraining scripts and loss functions  

- `downstream/` — Downstream task implementations  
  - `hemorrhage_seg/` — Brain hemorrhage segmentation  
  - `midline_shift_seg/` — Midline shift segmentation  
  - `mri_translation/` — MRI modality translation (e.g., T1 ↔ FLAIR)  
  - `mri_enhancement/` — MRI image enhancement  
  - `report_gen/` — Automatic radiology report generation  
  - `dialog_qa/` — Visual-language dialogue QA  
  - `mcq/` — Multiple-choice question answering (zero-shot)  

- `README.md` — Project introduction and documentation  

---
