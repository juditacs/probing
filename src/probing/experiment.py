#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import os
import shutil
import logging
import platform

import numpy as np

import torch

from probing.config import Config
from probing import data as data_module
from probing.result import Result
from probing import models
from probing.utils import check_and_get_commit_hash


use_cuda = torch.cuda.is_available()


class Experiment:
    def __init__(self, config, train_data=None, dev_data=None,
                 override_params=None, debug=False):
        git_hash = check_and_get_commit_hash(debug)
        if isinstance(config, Config):
            self.config = config
        else:
            self.config = Config.from_yaml(config, override_params)
        self.set_random_seeds()
        self.config.commit_hash = git_hash
        self.data_class = getattr(data_module, self.config.dataset_class)
        self.__load_data(train_data, dev_data)
        logging.info("Data loaded")
        logging.info(f"Train data size: {len(self.train_data)}")
        logging.info(f"Dev data size: {len(self.dev_data)}")
        data_fields = list(self.train_data.raw[0].keys())
        logging.info(f"Data fields: {', '.join(data_fields)}")
        self.init_model()
        try:
            self.model.check_params()
        except Exception as e:
            self.remove_experiment_dir()
            raise e

    def remove_experiment_dir(self):
        if os.path.exists(self.config.experiment_dir):
            shutil.rmtree(self.config.experiment_dir)

    def set_random_seeds(self):
        if not hasattr(self.config, 'torch_random_seed'):
            self.config.torch_random_seed = np.random.randint(0, 2**31)
        if not hasattr(self.config, 'numpy_random_seed'):
            self.config.numpy_random_seed = np.random.randint(0, 2**31)
        np.random.seed(self.config.numpy_random_seed)
        torch.manual_seed(self.config.torch_random_seed)
        if use_cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def __load_data(self, train_data, dev_data):
        if train_data is None and dev_data is None:
            train_fn = self.config.train_file
            dev_fn = self.config.dev_file
            self.__load_train_dev_data(train_fn, dev_fn)
        elif isinstance(train_data, str) and isinstance(dev_data, str):
            self.config.train_file = os.path.abspath(train_data)
            self.config.dev_file = os.path.abspath(dev_data)
            train_fn = train_data
            dev_fn = dev_data
            self.__load_train_dev_data(train_fn, dev_fn)
        else:
            assert isinstance(train_data, self.data_class)
            assert isinstance(dev_data, self.data_class)
            self.train_data = train_data
            self.dev_data = dev_data

    def __load_train_dev_data(self, train_fn, dev_fn):
        # FIXME max_samples=None should be the default
        if hasattr(self.config, 'train_size'):
            self.train_data = self.data_class(self.config, train_fn,
                                              max_samples=self.config.train_size)
        else:
            self.train_data = self.data_class(self.config, train_fn)
        self.train_data.save_vocabs()
        # FIXME share_vocabs should be handled in a nicer way
        if hasattr(self.config, 'dev_size'):
            self.dev_data = self.data_class(
                self.config, dev_fn, max_samples=self.config.dev_size,
                share_vocabs_with=self.train_data)
        else:
            self.dev_data = self.data_class(
                self.config, dev_fn,
                share_vocabs_with=self.train_data)

    def init_model(self):
        model_class = getattr(models, self.config.model)
        self.model = model_class(self.config, self.train_data)
        num_params = sum(p.nelement() for p in self.model.parameters()
            if p.requires_grad)
        logging.info(f"Number of parameters: {num_params}")
        if use_cuda:
            self.model = self.model.cuda()
        else:
            logging.warning("CUDA not available")

    def __enter__(self):
        logging.info(f"Starting experiment: {self.config.experiment_dir}")
        self.result = Result()
        self.result.node = platform.node()
        self.result.parameters = sum(
            p.nelement() for p in self.model.parameters() if p.requires_grad)
        if use_cuda:
            self.result.gpu = torch.cuda.get_device_name(
                torch.cuda.current_device())
        else:
            self.result.gpu = None
        self.result.train_size = len(self.train_data)
        self.result.dev_size = len(self.dev_data)
        self.config.save()
        self.result.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.result.end()
        logging.info(f"Experiment dir: {self.config.experiment_dir}")
        if len(self.result.dev_loss) > 0:
            min_ = int(self.result.running_time // 60)
            sec = int(self.result.running_time - min_ * 60)
            logging.info(
                f"Experiment finished in {min_}m{sec}s, "
                f"max dev acc: {max(self.result.dev_acc)}")
        else:
            logging.info("Experiment failed, no successful training epochs.")
        if exc_type is None:
            self.result.exception = None
        else:
            self.result.exception = {
                'type': exc_type.__name__,
                'value': str(exc_value),
            }
        self.result.epochs_run = len(self.result.train_loss)
        self.config.save()
        self.result.save(self.config.experiment_dir)

    def run(self):
        self.model.run_train(
            self.train_data, self.result, dev_data=self.dev_data)
