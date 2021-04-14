#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2020 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import logging
import os
import gc
import torch

from probing.config import Config


def generate_configs(config_fn):
    this_dir = os.path.dirname(__file__)
    train_file = f"{this_dir}/../data/pos_tagging/english/train"
    dev_file = f"{this_dir}/../data/pos_tagging/english/dev"
    for subword in ['first', 'last']:
        config = Config.from_yaml(config_fn)
        config.train_size = 100
        config.dev_size = 20
        config.subword_pooling = subword
        config.train_file = train_file
        config.dev_file = dev_file
        yield config
        gc.collect()
        torch.cuda.empty_cache()