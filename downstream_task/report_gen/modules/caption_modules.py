# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2022/07/14 17:24
# @Author  : Liangdi.Ma

import torch
from torch import nn
from .embedding import WordAndPositionalEmbedding


class TransformerTextualHead(nn.Module):
    def __init__(self, config, visual_embedding_size):
        super().__init__()
        self.config = config
        self.visual_embedding_size = visual_embedding_size
        self.vocab_size = self.config.tokenizer.vocab_size
        self.hidden_size = self.config.model.caption.decoder.hidden_size
        self.num_layers = self.config.model.caption.decoder.num_layers
        self.attention_heads = self.config.model.caption.decoder.attention_heads
        self.feedforward_size = self.config.model.caption.decoder.feedforward_size
        self.dropout = self.config.model.caption.decoder.dropout
        self.mask_future_positions = self.config.model.caption.decoder.mask_future_positions
        self.max_caption_length = self.config.tokenizer.max_length
        self.padding_idx = self.config.tokenizer.pad_index

        self.visual_projection = nn.Linear(self.visual_embedding_size, self.hidden_size)
        self.embedding = WordAndPositionalEmbedding(
            self.vocab_size,
            self.hidden_size,
            dropout=self.dropout,
            max_caption_length=self.max_caption_length,
            padding_idx=self.padding_idx,
        )
        # Make decoder layer depending on whether it's a Pre-Norm or Post-Norm.
        LayerClass = (
            nn.TransformerDecoderLayer
        )
        _layer = LayerClass(
            self.hidden_size,
            self.attention_heads,
            dim_feedforward=self.feedforward_size,
            dropout=self.dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerDecoder(_layer, self.num_layers)
        self.apply(self._init_weights)

        # Create an output linear layer and tie the input and output word
        # embeddings to reduce parameters.
        self.output = nn.Linear(self.hidden_size, self.vocab_size)
        self.output.weight = self.embedding.words.weight

    @staticmethod
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

    def forward(self, visual_embeddings, enc_tokens):
        r"""
        :Input:

        visual_embeddings: output embedding of visual backbone,
        a tensor of shape (batch size, patch num, embedding dim(hidden dim))

        enc_tokens: dict of input caption tokens

        :Returns
        -------
        torch.Tensor
            A tensor of shape ``(batch_size, max_caption_length, vocab_size)``
            containing output vocabulary logits for each time-step.
        """
        ### prepare visual features
        projected_visual_features = self.visual_projection(visual_embeddings)  # shape: (bs, patch num, hidden size)

        ### prepare textual features
        token_ids = enc_tokens['input_ids']  # (batch_size, max_caption_length)
        batch_size, max_caption_length = token_ids.size()
        caption_embeddings = self.embedding(token_ids)  # (batch_size, max_caption_length, textual_feature_size)
        padding_mask = enc_tokens['attention_mask'] == 0

        # Create a binary mask: True for padding positions.
        # These positions will be ignored for multi-headed attention.
        # padding_mask = (token_ids == self.padding_idx).float()  # padding mask

        if self.mask_future_positions:
            # An additive mask for masking the future (one direction).
            unidirectional_mask = self._generate_future_mask(
                max_caption_length, caption_embeddings.dtype, caption_embeddings.device
            )
        else:
            unidirectional_mask = None

        # We transpose the first two dimensions of tokens embeddings and visual
        # features, as required by decoder.
        caption_embeddings = caption_embeddings.transpose(0, 1)
        projected_visual_features = projected_visual_features.transpose(0, 1)

        # print('caption embeddings: ', caption_embeddings.shape)
        # print('visual embeddings: ', projected_visual_features.shape)
        # print('unidirectional mask: ', unidirectional_mask.shape)
        # print('padding mask: ', padding_mask.shape)
        # exit()

        textual_features = self.transformer(
            caption_embeddings,
            projected_visual_features,
            tgt_mask=unidirectional_mask,
            tgt_key_padding_mask=padding_mask,
        )

        # Undo the transpose and bring batch to dim 0.
        textual_features = textual_features.transpose(0, 1)  # (batch_size, max_caption_length, hidden_size)
        output_logits = self.output(textual_features)  # (batch_size, max_caption_length, vocab_size)
        return output_logits, textual_features

    def _generate_future_mask(self, msize, mdtype, mdevice):
        r"""
        Generate a mask for "future" positions, useful when using this module
        for language modeling.

        Parameters
        ----------
        size: int
        """
        # Default mask is for forward direction. Flip for backward direction.
        mask = torch.triu(torch.ones(msize, msize, device=mdevice, dtype=mdtype), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask
