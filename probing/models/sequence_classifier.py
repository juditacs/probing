#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn

from probing.models import BaseModel
from probing.models.mlp import MLP


use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class LSTMEncoder(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 lstm_hidden_size=None,
                 lstm_cell=None,
                 lstm_num_layers=1,
                 lstm_dropout=0,
                 embedding_size=None,
                 embedding_dropout=None):
        super().__init__()
        self.output_size = output_size
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.hidden_size = lstm_hidden_size // 2
        self.num_layers = lstm_num_layers
        dropout = 0 if lstm_num_layers == 1 else lstm_dropout
        self.cell = nn.LSTM(
            embedding_size, self.hidden_size,
            num_layers=lstm_num_layers,
            dropout=dropout,
            batch_first=False,
            bidirectional=True,
        )

    def forward(self, input, input_len):
        embedded = self.embedding(input)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, input_len, batch_first=False, enforce_sorted=False)
        outputs, (h, c) = self.cell(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        num_layers = self.num_layers
        num_directions = 2
        batch = input.size(1)
        hidden_size = self.hidden_size
        h = h.view(num_layers, num_directions, batch, hidden_size)
        c = c.view(num_layers, num_directions, batch, hidden_size)
        h = torch.cat((h[:, 0], h[:, 1]), 2)
        c = torch.cat((c[:, 0], c[:, 1]), 2)
        return outputs, (h, c)


class SequenceClassifier(BaseModel):
    def __init__(self, config, dataset):
        super().__init__(config)
        input_size = len(dataset.vocabs.input)
        output_size = len(dataset.vocabs.label)
        self.lstm = LSTMEncoder(
            input_size, output_size,
            lstm_hidden_size=self.config.hidden_size,
            lstm_num_layers=self.config.num_layers,
            lstm_dropout=self.config.dropout,
            embedding_size=self.config.embedding_size,
            embedding_dropout=self.config.dropout,
        )
        if self.config.freeze_lstm_encoder:
            for p in self.lstm.parameters():
                p.requires_grad = False
        hidden = self.config.hidden_size
        self.mlp = MLP(
            input_size=hidden,
            layers=self.config.mlp_layers,
            nonlinearity=self.config.mlp_nonlinearity,
            output_size=output_size,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        input = to_cuda(torch.LongTensor(batch.input))
        input = input.transpose(0, 1)  # time_major
        input_len = batch.input_len
        outputs, hidden = self.lstm(input, input_len)
        labels = self.mlp(hidden[0][-1])
        return labels

    def compute_loss(self, target, output):
        target = to_cuda(torch.LongTensor(target.label)).view(-1)
        loss = self.criterion(output, target)
        return loss


class MidSequenceClassifier(SequenceClassifier):
    def forward(self, batch):
        input = to_cuda(torch.LongTensor(batch.input))
        input = input.transpose(0, 1)  # time_major
        input_len = batch.input_len
        outputs, hidden = self.lstm(input, input_len)
        batch_size = input.size(1)
        helper = to_cuda(torch.arange(batch_size))
        idx = to_cuda(torch.LongTensor(batch.target_idx))
        out = outputs[idx, helper]
        labels = self.mlp(out)
        return labels
