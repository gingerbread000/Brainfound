# 🧠 Pretraining Pipeline

Brainfound is pretrained in three progressive stages:

## 1. Diffusion-Based Visual Pretraining

### Commands

**Training:**
```bash
accelerate launch --gpu_ids=7 --num_processes=1 --main_process_port 6676 --config_file single.yaml pretrain_ddpm.py
```

**Image Generation (Testing):**
```bash
accelerate launch --gpu_ids=7 --num_processes=1 --main_process_port 6676 --config_file single.yaml generate.py
```

---

## 2. Contrastive Vision-Language Pretraining

### Commands

**Training (`run.sh`):**
```bash
CUDA_VISIBLE_DEVICES=5,6,7,8 python \
  -m torch.distributed.launch \
  --nproc_per_node=4 --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port=9999 \
  pretrain.py \
  --exp_name v5,from_report_pretrain_e79 \
  --epochs 50 \
  --model v5 \
  --sentence_shuffle false \
  --validation true \
  --valid_save_image_freq 100 \
  --tie_text_encoder_decoder false \
  --momentum 0.996 \
  --lr 1e-6 \
  --resume "/resume_path/checkpoint_0.pth" \
  --valid_freq 1 \
  --batch_size 3 \
  --train_print_freq 100
```

**Zero-Shot Evaluation (`run_zeroshot.eval`):**
```bash
CUDA_VISIBLE_DEVICES=5 python \
  -m torch.distributed.launch \
  --nproc_per_node=1 --nnodes=1 \
  --node_rank=0 \
  --master_addr="127.0.0.1" \
  --master_port=8668 \
  zero_shot_cls.py \
  --exp_name TEST_Zero \
  --epochs 1 \
  --model v5 \
  --sentence_shuffle false \
  --validation true \
  --valid_save_image_freq 100 \
  --tie_text_encoder_decoder false \
  --momentum 0.996 \
  --lr 1e-6 \
  --resume "/path_to_weight/checkpoint_50.pth" \
  --valid_freq 1 \
  --batch_size 1 \
  --train_print_freq 100
```

---

## 3. Multi-modal Instruction Fine-tuning

This stage is based on the [InternVL](https://github.com/OpenGVLab/InternVL) framework. We provide two custom patch files that modify InternVL to support Brainfound:

### Modified Files

- `modeling_internvl_chat.py`  
- `UNet2d_condition.py`

Both files should be placed in:

```
InternVL/internvl_chat/internvl/model/internvl_chat/
```

Please replace the original files in InternVL before running the fine-tuning stage.

### 📄 Fine-tuning Data Files

- **`mcq.jsonl`**: This file contains a small amount of fine-tuning data for multiple-choice question (MCQ) style tasks. Each entry in this file consists of a question, a set of possible answers, and the correct answer.
  
- **`qa.jsonl`**: This file contains a small set of question-answer pairs. It is used for training the model to follow question-answer style instructions in natural language, enhancing its ability to respond to specific queries about brain CT/MRI images.

---