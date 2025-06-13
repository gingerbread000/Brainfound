# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/05/26 17:10
# @Author  : Liangdi.Ma
import os
import torch
from torch import nn
import torch.distributed as dist
from transformers import AutoModel
from typing import List, Optional, Tuple, Union


def _init_weights(module):
    r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.MultiheadAttention):
        module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
        module.out_proj.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

class TransformerEncoderModel(nn.Module):
    def __init__(self, backbone="bert", pretrained=True, freeze=False, freeze_layers=[], root_dir=""):
        super(TransformerEncoderModel, self).__init__()
        load_dir = os.path.join(root_dir, 'models', backbone)
        if os.path.exists(load_dir):
            self.text_encoder = AutoModel.from_pretrained(load_dir)
            if dist.get_rank() == 0:
                print(f'+ build text encoder structure by {load_dir}')
        else:
            if dist.get_rank() == 0:
                print(f'+ no text encoder structure is found: {load_dir}')
            raise FileNotFoundError

        self.hidden_size = self.text_encoder.config.hidden_size

        if not pretrained:
            self.apply(_init_weights)
            if dist.get_rank() == 0:
                print(f'+ initialize text encoder weights randomly.')
        else:
            if dist.get_rank() == 0:
                print(f'+ initialize text encoder weights by {load_dir}')

        if freeze:
            if len(freeze_layers) > 0:
                if 'embeddings' in freeze_layers:
                    for param in list(self.text_encoder.embeddings.parameters()):
                        param.requires_grad = False
                    freeze_layers.remove('embeddings')
                for layer_idx in freeze_layers:
                    for param in list(self.text_encoder.encoder.layer[layer_idx].parameters()):
                        param.requires_grad = False
            else:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False

    def mean_pooling(self, output_embedding, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        Reference: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html
        """

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(output_embedding.size()).float()
        sum_embeddings = torch.sum(output_embedding * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def max_pooling(self, output_embedding, attention_mask):
        """
        MAX Pooling - Take attention mask into account
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(output_embedding.size()).float()
        mask_embeddings = output_embedding * input_mask_expanded
        max_embeddings, _ = torch.max(mask_embeddings, dim=1)
        return max_embeddings

    def forward(self, encoded_text, pool='mean', logits_input=False):
        output = self.text_encoder(**encoded_text)  # last hidden state, pooler_output
        output_embedding = output["last_hidden_state"]  # (bs, max_length, hidden_size)
        # return torch.mean(output_embedding, dim=1)
        with torch.no_grad():
            # sentence_embeddings = self.mean_pooling(output_embedding, encoded_text['attention_mask']).half()
            if pool == 'mean':
                sentence_embeddings = self.mean_pooling(output_embedding, encoded_text['attention_mask'])
            elif pool == 'max':
                sentence_embeddings = self.max_pooling(output_embedding, encoded_text['attention_mask'])
            else:
                raise NotImplementedError
        return sentence_embeddings
        # output_feature = outputs["pooler_output"]  # (bs, hidden_size)
        # projected_feature = self.fc(output_feature)
        # return output_embedding, projected_feature

    @property
    def output_hidden_size(self):
        return self.hidden_size

# if __name__ == '__main__':
#     text_encoder = AutoModel.from_pretrained(r"Z:\models\clinicalbert")
#     print(text_encoder.config)
#     exit()
#     hidden_size = text_encoder.config.hidden_size
#     exit()
#     decoder = TransformerTextualHead(vocab_size=1000,
#             hidden_size=768,
#             num_layers=2,
#             attention_heads=8,
#             feedforward_size=1024,
#             dropout=0.1,
#             mask_future_positions=True,
#             max_caption_length=30,
#             padding_idx=0)
#
#
#     caption_tokens = torch.Tensor([[101, 88,56,87,312,89,789,568,125,468,965,326,982, 0, 0, 0, 0,0,0,0]]).long()  ##(1,20)
#     caption_lengths = 13 * torch.ones(1).long()
#     img_f = torch.rand((1, 9, 768))
#     output, output_f = decoder(img_f, caption_tokens, caption_lengths)
#     print(output.shape)
#     print(output_f.shape)
#
#     # encoder = TransformerEncoderModel('clinicalbert', 768, 1024)
#     # caption_tokens = torch.Tensor(
#     #     [[101, 88, 56, 87, 312, 89, 789, 568, 125, 468, 965, 326, 982, 0, 0, 0, 0, 0, 0, 0]]).long()  ##(1,20)
#     # attn_mask = torch.where(caption_tokens == 0, 0, 1)
#     # output, output_f = encoder(caption_tokens, attn_mask)
#     # print(output.shape)
#     # print(output_f.shap
