# !/usr/bin/env python
# -*-coding:utf-8 -*-

import json
import random
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from transformers import AutoTokenizer

from utils import link
from utils import logger
import utils.scheduler as lr_sched
from utils.basic import *
from utils.visualization import *
from utils.optim_utils import add_lr_weight_decay_in_modality
from mdataset import MDataset, get_mod_prompt
from pretrain_model import  PretrainModelv5
from metrics.captions import compute_metrics as compute_caption_metrics
import warnings


warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
_model = {
    'v5': PretrainModelv5,  
}

def get_parse():
    parser = argparse.ArgumentParser()
    # default
    default_group = parser.add_argument_group(title='Default experiment options')
    default_group.add_argument("--local-rank", type=int, default=0)
    default_group.add_argument("--root_dir", type=str, default="/root/directory",)
    default_group.add_argument("--experiment", type=str, default="pretrain")
    default_group.add_argument("--exp_name", type=str, default="debug")
    default_group.add_argument("--output_dir", type=str, default="./output")
    default_group.add_argument("--save_dir", type=str, default="")
    default_group.add_argument("--ckpt_path", type=str, default="")
    default_group.add_argument("--event_path", type=str, default="")
    default_group.add_argument("--result_path", type=str, default="")

    # model
    model_group = parser.add_argument_group(title='Model options')
    model_group.add_argument("--model", type=str, default="v5")
    model_group.add_argument("--resume", type=str, default="")
    model_group.add_argument("--queue_size_slice", type=int, default=65536)
    model_group.add_argument("--queue_size_scan", type=int, default=2048)
    ## vision
    model_group.add_argument("--vision_load_dir", type=str, default="")
    model_group.add_argument("--vision_load_prefix", type=str, default="optimal")
    model_group.add_argument("--vision_backbone", type=str, default="ddpm")
    model_group.add_argument("--vision_pretrained", type=str2bool, default=True)
    model_group.add_argument("--vision_freeze_pretrained", type=str2bool, default=False)
    model_group.add_argument("--vision_freeze_pretrained_layers", type=str2list, default=[],
                             help="list of layers stated as: 1,2,3")
    model_group.add_argument("--vision_fusion", type=str, default='pool')
    ## text
    model_group.add_argument("--text_load_dir", type=str,default="")
    model_group.add_argument("--text_load_prefix", type=str, default="optimal")
    model_group.add_argument("--text_backbone", type=str, default="chinesebert")
    model_group.add_argument("--text_num_hidden_layers", type=int, default=6)
    model_group.add_argument("--text_output_hidden_states", type=str2bool, default=True)
    model_group.add_argument("--text_output_attentions", type=str2bool, default=True)
    model_group.add_argument("--text_freeze_pretrained", type=str2bool, default=False)
    model_group.add_argument("--text_freeze_pretrained_layers", type=str2list, default=[])
    model_group.add_argument("--text_pooling", type=str, default="mean")
    model_group.add_argument("--text_dropout", type=float, default=0.1)
    model_group.add_argument("--tie_text_encoder_decoder", type=str2bool, default=True)

    ## caption
    model_group.add_argument("--caption_load_dir", type=str, default="")
    model_group.add_argument("--caption_load_prefix", type=str, default="optimal")
    model_group.add_argument("--caption_backbone", type=str, default="chinesebert")
    model_group.add_argument("--caption_num_hidden_layers", type=int, default=6)
    model_group.add_argument("--caption_output_hidden_states", type=str2bool, default=True)
    model_group.add_argument("--caption_output_attentions", type=str2bool, default=True)
    model_group.add_argument("--caption_freeze_pretrained", type=str2bool, default=False)
    model_group.add_argument("--caption_freeze_pretrained_layers", type=str2list, default=[])
    model_group.add_argument("--caption_dropout", type=float, default=0.1)
    model_group.add_argument("--caption_beam_size", type=int, default=2)
    ## options
    model_group.add_argument("--global_feature_size", type=int, default=512)
    model_group.add_argument("--momentum", type=float, default=0.999)
    model_group.add_argument("--temperature_multimodal", type=float, default=0.07)
    model_group.add_argument("--temperature_image", type=float, default=0.07)
    model_group.add_argument("--temperature_text", type=float, default=0.07)

    # dataset
    dataset_group = parser.add_argument_group(title='Dataset options')
    dataset_group.add_argument("--dataset_root", type=str, default="/path_to_data")
    dataset_group.add_argument("--buffer_root", type=str, default="/path_to_data/buffer")
    dataset_group.add_argument("--train_pct", type=float, default=1.0, help="precentage of training dataset")
    dataset_group.add_argument("--val_pct", type=float, default=1.0, help="percentage of valid dataset")
    ## vision
    dataset_group.add_argument("--slice_num", type=int, default=32)
    dataset_group.add_argument("--image_size", type=int, default=224)
    ## text
    dataset_group.add_argument("--sentence_shuffle", type=str2bool, default=True)

    # tokenizer
    tokenizer_group = parser.add_argument_group(title='Tokenizer options')
    tokenizer_group.add_argument("--tokenizer", type=str, default='chinesebert')
    tokenizer_group.add_argument("--vocab_size", type=int, default=21128)
    tokenizer_group.add_argument("--encode_max_length", type=int, default=128)
    tokenizer_group.add_argument("--decode_max_length", type=int, default=128)
    tokenizer_group.add_argument("--text_pad_token_id", type=int, default=0)
    tokenizer_group.add_argument("--text_sos_token_id", type=int, default=101)
    tokenizer_group.add_argument("--text_eos_token_id", type=int, default=102)
    tokenizer_group.add_argument("--text_unk_token_id", type=int, default=100)

    # loss
    loss_group = parser.add_argument_group(title='Loss options')
    loss_group.add_argument("--loss", type=str, default="asl", help='bce|asl')
    loss_group.add_argument("--gamma_neg", type=int, default=2)
    loss_group.add_argument("--gamma_pos", type=int, default=1)
    loss_group.add_argument("--clip", type=float, default=0)
    loss_group.add_argument("--disable_torch_grad_focal_loss", default=False, type=bool)
    loss_group.add_argument("--weight_iic_slice", type=float, default=1)
    loss_group.add_argument("--weight_iic_scan", type=float, default=1)
    loss_group.add_argument("--weight_ttc", type=float, default=1)
    loss_group.add_argument("--weight_itc", type=float, default=1)
    loss_group.add_argument("--weight_lm", type=float, default=1)

    # optimizer
    optimizer_group = parser.add_argument_group(title='Optimizer options')
    optimizer_group.add_argument("--optimizer", type=str, default='adamw')
    optimizer_group.add_argument("--lr", type=float, default=0.00001)
    optimizer_group.add_argument("--blr", type=float, default=1e-3, metavar='LR',
                                 help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    optimizer_group.add_argument("--weight_decay", type=float, default=0.05)
    optimizer_group.add_argument("--eps", type=float, default=1e-8)
    optimizer_group.add_argument("--beta0", type=float, default=0.9)
    optimizer_group.add_argument("--beta1", type=float, default=0.999)
    optimizer_group.add_argument("--optimize_in_modality", type=str2bool, default=False)
    optimizer_group.add_argument("--vision_encoder_lr", type=float, default=0.0001)
    optimizer_group.add_argument("--vision_decoder_lr", type=float, default=0.0001)
    optimizer_group.add_argument("--vision_fusion_lr", type=float, default=0.0001)
    optimizer_group.add_argument("--text_encoder_lr", type=float, default=0.000001)
    optimizer_group.add_argument("--text_decoder_lr", type=float, default=0.000001)

    # scheduler
    scheduler_group = parser.add_argument_group(title='Scheduler options')
    scheduler_group.add_argument("--scheduler", type=str, default="cosine")
    scheduler_group.add_argument("--warmup_steps", type=int, default=1000)
    scheduler_group.add_argument("--schedule_in_epoch", type=str2bool, default=False)
    scheduler_group.add_argument("--min_lr", type=float, default=1e-8)

    # training, valid, evaluation
    default_group.add_argument("--epochs", type=int, default=50)
    default_group.add_argument("--clip_grad", type=float, default=3.0)
    default_group.add_argument("--batch_size", type=int, default=2, help="batch size per gpu")
    default_group.add_argument("--num_workers", type=int, default=5)
    default_group.add_argument("--pin_memory", type=str2bool, default=True)
    default_group.add_argument("--non_blocking", type=str2bool, default=True)
    default_group.add_argument("--validation", type=str2bool, default=False)
    default_group.add_argument("--valid_freq", type=int, default=1)
    default_group.add_argument("--valid_index", type=str, default="auc")
    default_group.add_argument("--valid_data", type=str, default="test", choices=["valid", "test"])
    default_group.add_argument("--valid_mask_ratio", type=float, default=0.2)
    default_group.add_argument("--early_stop", type=str2bool, default=False)
    default_group.add_argument("--patience", type=int, default=10)

    # other
    default_group.add_argument("--train_print_freq", type=int, default=500)
    default_group.add_argument("--valid_print_freq", type=int, default=100)
    default_group.add_argument("--valid_save_image_freq", type=int, default=-1)
    default_group.add_argument("--save_base_model", type=str2bool, default=False)
    default_group.add_argument("--distributed", type=str2bool, default=True)
    default_group.add_argument("--local_rank", type=int, default=0, help="node rank for distributed training")
    default_group.add_argument("--dist_url", type=str, default="env://")
    default_group.add_argument("--fp16", type=str2bool, default=True)
    default_group.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()
    return args


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def initialize(args):
    args.exp_name = getTime() + '_' + args.exp_name + f',seed={args.seed}'
    args.output_dir = os.path.join(args.output_dir, 'pretrain')
    if args.save_dir != "":
        args.output_dir = os.path.join(args.output_dir, args.save_dir)
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    ckpt_path = os.path.join(args.output_dir, 'checkpoints')
    event_path = os.path.join(args.output_dir, 'events')
    result_path = os.path.join(args.output_dir, 'results')
    args.ckpt_path = ckpt_path
    args.event_path = event_path
    args.result_path = result_path
    args.weight_ttc = 0

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * args.batch_size * link.get_world_size() / 256

    argsDict = args.__dict__
    if dist.get_rank() == 0:
        makedir(args.output_dir)
        makedir(args.ckpt_path)
        makedir(args.event_path)
        makedir(args.result_path)
        makedir(os.path.join(args.result_path, 'caption'))
        makedir(os.path.join(args.result_path, 'images'))
        with open(os.path.join(args.output_dir, 'train_options.json'), 'w', encoding='utf-8') as f:
            json.dump(argsDict, f)
        show_options = '------------------ training options ------------------' + '\n'
        for eachArg, value in argsDict.items():
            show_options += eachArg + ' : ' + str(value) + '\n'
        show_options += '------------------- end -------------------'
        with open(os.path.join(args.output_dir, 'train_options.txt'), 'w', encoding='utf-8') as f:
            f.write(show_options)
        print(show_options)
        save_code('.', os.path.join(args.output_dir, 'code.zip'))


@torch.no_grad()
def valid_one_epoch(model, data_loader, tokenizer, tokenizer_mod_prompt, epoch, device, args):
    # valid
    model.eval()

    metric_logger = logger.MetricLogger(delimiter="  ")
    
    header = 'Valid Epoch: [{}]/[{}]'.format(epoch, args.epochs)
    caption_records = ''

    for i, data in enumerate(metric_logger.log_every(data_loader, args.valid_print_freq, header)):
        images = data['image']  # len = 1
        texts = data['caption']  # len = 1
        mod_clss = data["mod_cls"]
        encoder_hidden_states = []
        attention_mask = []
        encode_tokens = []
        images[0] = images[0].to(device, non_blocking=True)
        for t in texts:
            temp = tokenizer(t, padding='max_length', max_length=args.encode_max_length, truncation=True,
                             return_tensors='pt')
            for k, v in temp.items():
                temp[k] = v.to(device)
            encode_tokens.append(temp)
        for t in mod_clss:
            temp = get_mod_prompt(tokenizer_mod_prompt, t)
            encoder_hidden_states.append(temp[0].to(device))
            attention_mask.append(temp[1].to(device))
        decode_tokens = tokenizer(texts[0], padding='max_length', max_length=args.decode_max_length, truncation=True,
                                  return_tensors='pt')
        for k, v in decode_tokens.items():
            decode_tokens[k] = v.to(device)
        
        with torch.no_grad():
            output = model(im_q=images[0], text_q=encode_tokens[0], text_c=decode_tokens, mode='valid', prompt_info=(encoder_hidden_states[0], attention_mask[0]))

                
        # record output result
        valid_caption_metrics, valid_caption_record = compute_caption_metrics(
            decode_tokens['input_ids'].detach(), output['pred_caption_ids'].detach(),
            tokenizer=tokenizer,)# img_path=data['image_path'])
        caption_records += valid_caption_record
        metric_logger.update(bleu1=valid_caption_metrics['BLEU_1'])
        metric_logger.update(bleu2=valid_caption_metrics['BLEU_2'])
        metric_logger.update(bleu3=valid_caption_metrics['BLEU_3'])
        metric_logger.update(bleu4=valid_caption_metrics['BLEU_4'])
        metric_logger.update(meteor=valid_caption_metrics['METEOR'])
        metric_logger.update(cider=valid_caption_metrics['CIDER'])
        metric_logger.update(rouge_l=valid_caption_metrics['ROUGE_L'])

        metric_logger.update(loss_itc=output['loss_itc'].item())
        metric_logger.update(loss_lm=output['loss_lm'].item())

        # gather the stats from all processes
    images = images[0].detach().cpu().numpy()
    print(images.shape)
    images = np.transpose(images, (0,1,3,4,2))
    images = (images - images.min()) / (images.max() - images.min())
    print(images.shape)
    for i in range(32):
        plt.imshow(images[0,i])
        plt.savefig(f"{args.result_path}/images/{i}.png")

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    print("")
    with open(os.path.join(args.result_path, 'caption', f'valid_caption_epoch{epoch}_rank{link.get_rank()}.txt'),
                'w',
                encoding='utf-8') as f:
        f.write(caption_records)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    ### env initialize ###
    link.init_distributed_mode(args)
    device = torch.device("cuda", args.local_rank)

    ### experiment initialize ###
    init_seeds(args.seed + link.get_rank())
    initialize(args)

    ### build tokenizer ###
    tokenizer_path = "/path_to/chinesebert"
    print(f"=> Build {args.tokenizer} tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    assert tokenizer.pad_token_id == args.text_pad_token_id, "pad token id not match."
    assert tokenizer.cls_token_id == args.text_sos_token_id, "sos(cls) token id not match."
    assert tokenizer.sep_token_id == args.text_eos_token_id, "eos(sep) token id not match."
    assert tokenizer.unk_token_id == args.text_unk_token_id, "unk token id not match."

    tokenizer_mod_prompt = AutoTokenizer.from_pretrained("path_to/models--bert-base-cased/snapshots/cd5ef92a9fb2f889e972770a36d4ed042daf221e")

    ### build model and optimizer ###
    print(f"=> Build {args.model} Pretrain Model")
    model = _model[args.model](args)

    if args.resume != "":
        checkpoint = torch.load(args.resume, map_location='cpu')
        if "model" in checkpoint: checkpoint = checkpoint["model"]
        new_ckpt = {}
        for k, v in checkpoint.items():
            new_ckpt[k[len("module."):]] = v
            # del checkpoint[k]
        model.load_state_dict(new_ckpt)
        print("+ Resume checkpoint %s" % args.resume)

    ### build dataset ###
    print(f'=> Build Pretrain Dataset')
    train_dataset = MDataset(args, section="train")
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                              num_workers=args.num_workers, drop_last=True, pin_memory=args.pin_memory)

    if args.validation:
        valid_dataset = MDataset(args, section="valid")
        valid_sampler = DistributedSampler(valid_dataset)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler,
                                  num_workers=args.num_workers, drop_last=True, pin_memory=args.pin_memory)

    
    # param_groups = model.parameters()
    param_groups = add_lr_weight_decay_in_modality(model, args,)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=args.eps, betas=(args.beta0, args.beta1))
    for p_groups in optimizer.param_groups:
        p_groups["base_lr"] = p_groups["lr"]

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
    if args.save_base_model and link.is_main_process() == 0:
        torch.save(model.state_dict(), os.path.join(args.ckpt_path, 'checkpoint_base.pth'))

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    print(f"scaler is {scaler}")

    if link.is_main_process():
        writer = SummaryWriter(log_dir=args.event_path)

    print('===============START TRAIN================')
    start_time = time.time()
    epoch = 0
    
    valid_stats = valid_one_epoch(model, valid_loader, tokenizer, tokenizer_mod_prompt, epoch, device, args)
    dist.barrier()
    if link.is_main_process():
        log_stats = {**{f'valid_{k}': f'{v:.4f}' for k, v in valid_stats.items()},
                        'epoch': epoch,
                        }

        with open(os.path.join(args.output_dir, "log_val.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

        for k, v in valid_stats.items():
            try:
                writer.add_scalar(f'pretrain/valid_{k}', v, epoch)
            except:
                pass

    
        caption_dir = os.path.join(args.result_path, 'caption')
        record_files = [p for p in os.listdir(caption_dir) if
                        p.startswith(f'valid_caption_epoch{epoch}_rank') and p.endswith('.txt')]
        p_rank = [int(os.path.splitext(p)[0].split('_rank')[-1]) for p in record_files]
        record_files, _ = [list(p) for p in zip(*sorted(zip(record_files, p_rank), key=lambda x: x[-1]))]
        record_files = [os.path.join(caption_dir, p) for p in record_files]
        assert len(record_files) == link.get_world_size()

        with open(os.path.join(caption_dir, f'valid_caption_epoch{epoch}.txt'), 'w', encoding='utf-8') as f:
            for p in record_files:
                for line in open(p, encoding='utf8'):
                    f.writelines(line)
                f.write('\n')

        for p in record_files:
            os.remove(p)


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_parse()
    main(args)
