#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, layers, nonlinearity='Sigmoid'):
        super().__init__()
        mlp_layers = []
        if not isinstance(nonlinearity, list):
            nonlinearity = [nonlinearity] * (len(layers) + 1)
        layers = [input_size] + layers
        mlp_layers = []
        # input and hidden layers
        for i in range(0, len(layers)-1):
            n = layers[i]
            m = layers[i+1]
            mlp_layers.append(nn.Linear(n, m))
            mlp_layers.append(getattr(nn, nonlinearity[i])())
        # output layer without nonlinearity
        mlp_layers.append(nn.Linear(layers[-1], output_size))
        self.layers = nn.Sequential(*mlp_layers)

    def forward(self, input):
        return self.layers(input)
