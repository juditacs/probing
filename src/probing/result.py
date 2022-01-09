#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.
import os
import yaml
from datetime import datetime


class Result:
    __slots__ = ('train_loss', 'dev_loss', 'train_acc', 'dev_acc',
                 'running_time', 'start_time', 'train_size', 'dev_size',
                 'parameters', 'epochs_run', 'node', 'gpu', 'exception')

    def __init__(self):
        self.train_loss = []
        self.dev_loss = []
        self.train_acc = []
        self.dev_acc = []
        self.train_size = self.dev_size = None

    def start(self):
        self.start_time = datetime.now()

    def end(self):
        self.running_time = (datetime.now() - self.start_time).total_seconds()

    def save(self, expdir):
        d = {k: getattr(self, k, None) for k in self.__slots__}
        with open(os.path.join(expdir, 'result.yaml'), 'w') as f:
            yaml.dump(d, f, default_flow_style=False)

    def merge(self, other):
        for i in range(len(other.train_loss)):
            self.train_loss.append(other.train_loss[i])
            self.dev_loss.append(other.dev_loss[i])
            self.train_acc.append(other.train_acc[i])
            self.dev_acc.append(other.dev_acc[i])
