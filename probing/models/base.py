#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
import logging

import torch
import torch.nn as nn
from torch import optim


use_cuda = torch.cuda.is_available()


def to_cuda(var):
    if use_cuda:
        return var.cuda()
    return var


class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def check_params(self):
        pass

    def run_train(self, train_data, result, dev_data):

        self.init_optimizers()

        saved = False
        if self.config.lr_decay:
            lrd = self.config.lr_decay
            patience = self.config.lr_decay_patience
            for opt in self.optimizers:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    opt, mode='min', factor=lrd, patience=patience,
                    threshold=10e-6,
                    verbose=True)
        else:
            scheduler = None

        for epoch in range(self.config.epochs):
            self.train(True)
            train_loss, train_acc = self.run_epoch(train_data, do_train=True,
                                                   result=result)
            result.train_loss.append(train_loss)
            result.train_acc.append(train_acc)
            self.train(False)
            dev_loss, dev_acc = self.run_epoch(dev_data, do_train=False)
            result.dev_loss.append(dev_loss)
            result.dev_acc.append(dev_acc)
            s = self.save_if_best(result, epoch)
            saved = saved or s
            logging.info("Epoch {}, Train loss: {}, Train acc: {}, "
                         "Dev loss: {}, Dev acc: {}".format(
                             epoch+1,
                             round(train_loss, 4),
                             round(train_acc * 100, 2),
                             round(dev_loss, 4),
                             round(dev_acc * 100, 2),
                         ))
            if self.should_early_stop(epoch, result):
                logging.info("Early stopping.")
                break
            if epoch == 0:
                self.config.save()
            result.save(self.config.experiment_dir)
            if scheduler:
                scheduler.step(dev_loss)
        if saved is False:
            self._save(epoch)

    def should_early_stop(self, epoch, result):
        """Returns True if early stopping condition is reached"""
        if epoch < self.config.min_epochs - 1:
            return False
        window = self.config.early_stopping_window
        if len(result.dev_loss) < 2 * window:
            return False
        if self.config.early_stopping_strict:
            ea_loss = sum(result.dev_loss[-2*window:-window]) <= \
                    sum(result.dev_loss[-window:])
            ea_acc = sum(result.dev_acc[-2*window:-window]) >= \
                    sum(result.dev_acc[-window:])
        else:
            ea_loss = sum(result.dev_loss[-2*window:-window]) < \
                    sum(result.dev_loss[-window:])
            ea_acc = sum(result.dev_acc[-2*window:-window]) > \
                    sum(result.dev_acc[-window:])
        if self.config.early_stopping_monitor == 'dev_acc':
            return ea_acc
        if self.config.early_stopping_monitor == 'dev_loss':
            return ea_loss
        if self.config.early_stopping_monitor == 'both':
            return ea_acc and ea_loss
        if self.config.early_stopping_monitor == 'either':
            return ea_acc or ea_loss
        return False

    def run_epoch(self, data, do_train, pass_dataset_to_forward=False,
                  result=None):
        epoch_loss = 0
        all_correct = all_guess = 0
        for step, batch in enumerate(data.batched_iter(self.config.batch_size)):
            if pass_dataset_to_forward:
                output = self.forward(batch, data)
            else:
                output = self.forward(batch)
            for opt in self.optimizers:
                opt.zero_grad()
            loss = self.compute_loss(batch, output)
            if do_train:
                loss.backward()
                if getattr(self.config, 'clip', None):
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(), self.config.clip)
                for opt in self.optimizers:
                    opt.step()
            target = torch.LongTensor(batch.tgt)
            prediction = output.max(dim=-1)[1].cpu()
            correct = torch.eq(prediction, target)
            if hasattr(batch, 'tgt_len'):
                doc_lens = to_cuda(torch.LongTensor(batch.tgt_len))
                tgt_size = target.size()
                m = torch.arange(tgt_size[1]).unsqueeze(0).expand(tgt_size)
                mask = doc_lens.unsqueeze(1).expand(tgt_size) <= \
                    to_cuda(m.long())
                correct[mask] = 1
                correct = correct.min(-1)[0]
                numel = correct.size(0)
            else:
                numel = target.numel()
            all_correct += correct.sum().item()
            all_guess += numel
            epoch_loss += loss.item()
        return epoch_loss / (step + 1), all_correct / max(all_guess, 1)

    def save_if_best(self, result, epoch):
        if epoch < self.config.save_min_epoch:
            return False
        if self.config.save_metric == 'dev_loss':
            # FIXME why was this added?
            # loss = dev_loss if dev_loss is not None else train_loss
            loss = result.dev_loss[-1]
            if not hasattr(self, 'min_loss') or self.min_loss > loss:
                self.min_loss = loss
                self._save(epoch)
                return True
            return False
        elif self.config.save_metric == 'dev_acc':
            acc = result.dev_acc[-1]
            if not hasattr(self, 'max_acc') or self.max_acc < acc:
                self.max_acc = acc
                self._save(epoch)
                return True
            return False

    def _load(self, model_file):
        logging.info("Loading model from {}".format(model_file))
        try:
            self.load_state_dict(torch.load(model_file))
        except RuntimeError:
            logging.warning("Strict loading failed. I'll try strict=False now")
            self.load_state_dict(torch.load(model_file), strict=False)

    def _save(self, epoch):
        if self.config.overwrite_model is True:
            save_path = os.path.join(self.config.experiment_dir, "model")
        else:
            save_path = os.path.join(
                self.config.experiment_dir,
                "model.epoch_{}".format("{0:04d}".format(epoch)))
        logging.info("Saving model to {}".format(save_path))
        torch.save(self.state_dict(), save_path)

    def run_inference(self, data, pass_dataset_to_forward=False):
        self.train(False)
        all_output = []
        for bi, batch in enumerate(data.batched_iter(self.config.batch_size)):
            if pass_dataset_to_forward:
                output = self.forward(batch, data)
            else:
                output = self.forward(batch)
            output = output.data.cpu().numpy()
            if output.ndim == 3:
                output = output.argmax(axis=2)
            all_output.extend(list(output))
        return all_output

    def init_optimizers(self):
        opt_type = getattr(optim, self.config.optimizer)
        kwargs = self.config.optimizer_kwargs
        self.optimizers = [opt_type(
            (p for p in self.parameters() if p.requires_grad),
            **kwargs)
        ]

    def compute_loss(self):
        raise NotImplementedError("Subclass should implement compute_loss")
