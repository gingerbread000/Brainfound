CUDA_VISIBLE_DEVICES=5 python \
  -m torch.distributed.launch \
  --nproc_per_node 1 --nnodes=1 \
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
