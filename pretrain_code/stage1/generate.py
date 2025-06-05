import os
import math
import numpy as np
from PIL import Image

import torch

from dataclasses import dataclass
from accelerate import Accelerator

from diffusers import UNet2DConditionModel
from diffusers import DDPMScheduler
from utils.pipeline_ddpm import DDPMPipeline
from transformers import BertTokenizer, BertModel

# from multiprocessing import set_start_method
# # torch.cuda.set_device(2)

@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 4
    eval_batch_size = 1# how many images to sample during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    eval_freq_step = 2000
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "generate_images"  
    num_workers = 4
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 42

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(config, epoch, pipeline, prompt_ind, prompt_embeding):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(np.random.randint(123456789)),
        condition=[0, prompt_embeding[prompt_ind].view(1,1,-1)],
    ).images

    # Make a grid out of the images
    num_grid = int(math.sqrt(config.eval_batch_size))
    image_grid = make_grid(images, rows=num_grid, cols=num_grid)

    # Save the images
    image_grid.save(f"./generate_images/{epoch:04d}_{prompt_ind}.png")

def train_loop(config, model, noise_scheduler, prompt_embeding):
    
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")
    
    pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler,)
    for i in range(1):
        for j in range(4):
            evaluate(config, i, pipeline, j, prompt_embeding) 

if __name__ == "__main__":
    config = TrainingConfig()
    bert_path = "./chinesebert"
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model = BertModel.from_pretrained(bert_path)
    text_inputs = [
        "CT.",
        "T1 mri.",
        "T2 mri.",
        "pub mri.",
    ]
    prompt_embeding = []
    for text in text_inputs:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k:inputs[k] for k in inputs}
        with torch.no_grad():
            outputs = model(**inputs)
    
        # 获取最后一层的隐藏状态，取 [CLS] token 的向量作为句子嵌入
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        prompt_embeding.append(cls_embedding.view(1,-1))

    model = UNet2DConditionModel(
        sample_size=config.image_size,  # the target image resolution
        in_channels=3,  # the number of input channels, 3 for RGB images
        out_channels=3,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(64, 64, 128, 256, 512),  # the number of output channels for each UNet block
        # block_out_channels=(320, 640, 1280, 1280),
        down_block_types=(
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D", 
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", 
            "CrossAttnUpBlock2D",  
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ),
        norm_num_groups=32,
        # addition_embed_type="text",
        # addition_embed_type_num_heads=64,
        encoder_hid_dim_type="text_proj",
        encoder_hid_dim=768,
    )

    model = model.from_pretrained("./pretrain_w/")

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)  

    accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(config.output_dir, "logs"),
        )

    # Prepare everything
    model = accelerator.prepare(model)
    
    train_loop(config, model, noise_scheduler, prompt_embeding)
    