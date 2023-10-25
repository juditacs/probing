#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import os
import logging
from sys import stdin, stdout
import pickle

import torch

from probing.config import InferenceConfig
import probing.data as data_module
from probing.experiment import Experiment
from probing import models


use_cuda = torch.cuda.is_available()


class Inference(Experiment):
    def __init__(self, experiment_dir, stream_or_file,
                 max_samples=None,
                 save_attention_weights=None,
                 param_str=None,
                 model_file=None):
        self.config = InferenceConfig.from_yaml(
            os.path.join(experiment_dir, 'config.yaml'))
        data_class = getattr(data_module, self.config.dataset_class)
        self.test_data = data_class(self.config, stream_or_file, max_samples=max_samples,
                                    is_unlabeled=True)
        self.test_data.is_unlabeled = True
        self.set_random_seeds()
        self.init_model(model_file)

    def init_model(self, model_file=None):
        model_class = getattr(models, self.config.model)
        self.model = model_class(self.config, self.test_data)
        if use_cuda:
            self.model = self.model.cuda()
        self.model.train(False)
        if model_file is None:
            model_file = self.find_last_model()
        self.model._load(model_file)

    def find_last_model(self):
        model_pre = os.path.join(self.config.experiment_dir, 'model')
        if os.path.exists(model_pre):
            return model_pre
        saves = filter(lambda f: f.startswith(
            'model.epoch_'), os.listdir(self.config.experiment_dir))
        last_epoch = max(saves, key=lambda f: int(f.split("_")[-1]))
        return os.path.join(self.config.experiment_dir, last_epoch)

    def run(self):
        model_output = self.model.run_inference(self.test_data)
        words = self.test_data.decode(model_output)
        return words

    def run_and_print(self, stream=stdout):
        model_output = self.model.run_inference(self.test_data)
        self.test_data.decode_and_print(model_output, stream)
        if hasattr(self.model, 'all_weights'):
            with open(self.config.experiment_dir + '/mlp_weights', 'wb') as f:
                pickle.dump(self.model.all_weights, f)


def parse_args():
    p = ArgumentParser()
    p.add_argument("-e", "--experiment-dir", type=str,
                   help="Experiment directory")
    p.add_argument("--model-file", type=str, default=None,
                   help="Model pickle. If not specified, the latest "
                   "model is used.")
    p.add_argument("-t", "--test-file", type=str, default=None,
                   help="Test file location. If unspecified, the input is read from STDIN.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.test_file:
        inf = Inference(args.experiment_dir, args.test_file,
                        model_file=args.model_file)
    else:
        inf = Inference(args.experiment_dir, stdin,
                        model_file=args.model_file)
    inf.run_and_print()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
