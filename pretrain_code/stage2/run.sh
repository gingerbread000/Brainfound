CUDA_VISIBLE_DEVICES=5,6,7,8 python \
  -m torch.distributed.launch \
  --nproc_per_node 4 --nnodes=1 \
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
