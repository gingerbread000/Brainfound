import pdb
import json
import random
import time
import pickle
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from transformers import AutoTokenizer
from transformers import BertTokenizer, BertModel

from utils import link
from utils import logger
from utils.basic import *
from utils.visualization import *
from mdataset import MDataset
from pretrain_model import  PretrainModelv5_Zeroshot
import warnings


warnings.filterwarnings('ignore')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
_model = {
    'v5': PretrainModelv5_Zeroshot,  
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
def valid_zero_shot(model, data_loader, tokenizer, prompt_info, epoch, device, args):
    # valid
    model.eval()
    metric_logger = logger.MetricLogger(delimiter="  ")

    header = 'Valid Epoch: [{}]/[{}]'.format(epoch, args.epochs)
    # image = preprocess(Image.open("docs/CLIP.png")).unsqueeze(0)
    # text = tokenizer(["a diagram", "a dog", "a cat"])
    candidate_texts = [
            "这是正常类别。",
            "这是出血类别。",
            "这是缺血性类别。",
            "这是骨折类别。",
            "这是肿瘤类别。",
        ]
    candidate_enc_tokens = []
    for t in candidate_texts:
        temp = tokenizer(t, padding='max_length', max_length=args.encode_max_length, truncation=True,
                            return_tensors='pt')
        for k, v in temp.items():
            temp[k] = v.to(device)
        candidate_enc_tokens.append(temp)

    y_true = []
    y_prob = []
    y_name = []


    import csv
    save_csv_path = "logits.csv"
    csv_header = ["filename", "正常", "出血","缺血","骨折","肿瘤", "标签"]
    if not os.path.exists(save_csv_path):
        with open(save_csv_path, mode="w",newline="") as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)

    for i, data in enumerate(metric_logger.log_every(data_loader, args.valid_print_freq, header)):
        images = data['image']  # len = 1
        texts = data['caption']  # len = 1
        fns = data['k']
        gt = data['cls'].tolist()
        # pdb.set_trace()
        
        mod_clss = data["mod_cls"]
        encoder_hidden_states = []
        encode_tokens = []
        images[0] = images[0].to(device, non_blocking=True)
        
        for t in texts:
            temp = tokenizer(t, padding='max_length', max_length=args.encode_max_length, truncation=True,
                             return_tensors='pt')
            for k, v in temp.items():
                temp[k] = v.to(device)
            encode_tokens.append(temp)

        
        with torch.no_grad():
            # print(images[0].shape, prompt_info[0][None].to(device).shape)
            # pdb.set_trace()
            image_features, _, _ = model.encode_image(images[0], prompt_info[0][None].to(device).repeat(32,1,1))
            gather_text_feats = [model.encode_text(text_q) for text_q in candidate_enc_tokens]

            sim_i2t = [ image_features @ gather_text_feat.t() / model.temp_multimodal for gather_text_feat in gather_text_feats ]  # similarity matrix (bs, bs)
            sim_i2t = torch.cat(sim_i2t, dim=1)
            text_probs = sim_i2t.softmax(dim=-1)
            # pdb.set_trace()

            with open(save_csv_path, mode="a",newline="") as file:
                writer = csv.writer(file)
                logits_val = text_probs.cpu().numpy().flatten()
                row = fns + logits_val.tolist() + gt
                writer.writerow(row)

            y_true.append(data["cls"].detach().cpu().numpy())   
            y_prob.append(text_probs.detach().cpu().numpy())
            y_name.append(data["k"])

    y_true = np.array(y_true).flatten()
    y_prob = np.vstack(y_prob)
    y_pred = np.argmax(y_prob, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}")

    y_true_binary = label_binarize(y_true, classes=np.arange(5))

    # 计算每个类别的AUC
    class_aucs = []
    for i in range(5):
        class_auc = roc_auc_score(y_true_binary[:, i], y_prob[:, i])
        class_aucs.append(class_auc)
        print(f"Class {candidate_texts[i][2:4]} AUC: {class_auc:.4f}")

    # 计算总体AUC（宏平均 AUC）
    overall_auc = roc_auc_score(y_true_binary, y_prob, average="macro")
    print(f"Overall AUC (Macro-average): {overall_auc:.4f}")
    
    metric_logger.update(acc=accuracy)
    metric_logger.update(auc=overall_auc) 
            
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def main(args):
    ### env initialize ###
    link.init_distributed_mode(args)
    device = torch.device("cuda", args.local_rank)

    ### experiment initialize ###
    init_seeds(args.seed + link.get_rank())
    initialize(args)

    ### build tokenizer ###
    tokenizer_path = "./chinesebert"
    print(f"=> Build {args.tokenizer} tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    assert tokenizer.pad_token_id == args.text_pad_token_id, "pad token id not match."
    assert tokenizer.cls_token_id == args.text_sos_token_id, "sos(cls) token id not match."
    assert tokenizer.sep_token_id == args.text_eos_token_id, "eos(sep) token id not match."
    assert tokenizer.unk_token_id == args.text_unk_token_id, "unk token id not match."
    
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertModel.from_pretrained(tokenizer_path)
    text_inputs = [
        "CT.",
        "T1 mri.",
        "T2 mri.",
    ]
    prompt_embeding = []
    for text in text_inputs:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k:inputs[k] for k in inputs}
        with torch.no_grad():
            outputs = model(**inputs)
    
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze()
        prompt_embeding.append(cls_embedding.view(1,-1))
    prompt_info = prompt_embeding

    ### build model and optimizer ###
    print(f"=> Build {args.model} Pretrain Model")
    model = _model[args.model](args)

    checkpoint = torch.load(args.resume, map_location='cpu')
    if "model" in checkpoint: checkpoint = checkpoint["model"]
    # pdb.set_trace()
    new_ckpt = {}
    for k, v in checkpoint.items():
        new_ckpt[k.replace("module.", "").replace("vision_encoder_q.model.model", "vision_encoder_q.model")] = v
        # del checkpoint[k]
    model.load_state_dict(new_ckpt, strict=True)

    print("+ Resume checkpoint %s" % args.resume)

    test_dataset = MDataset(args, section="test") #MDataset(args, section="test")
    test_sampler = DistributedSampler(test_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,
                                num_workers=args.num_workers, drop_last=False, pin_memory=args.pin_memory)

    model = model.to(device)
    # model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
    
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    print(f"scaler is {scaler}")

    if link.is_main_process():
        writer = SummaryWriter(log_dir=args.event_path)

    print('===============START TEST ZERO================')
    start_time = time.time()
    epoch = 1

    valid_stats = valid_zero_shot(model, test_loader, tokenizer, prompt_info, epoch, device, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Test time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_parse()
    main(args)
