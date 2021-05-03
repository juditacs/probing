#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import os
import logging
from sys import stdin, stdout
import yaml
import gc

import torch

from probing.inference import Inference


class NotAnExperimentDir(ValueError):
    pass


def find_last_model(experiment_dir):
    model_pre = os.path.join(experiment_dir, 'model')
    if os.path.exists(model_pre):
        return model_pre
    saves = filter(lambda f: f.startswith(
        'model.epoch_'), os.listdir(experiment_dir))
    last_epoch = max(saves, key=lambda f: int(f.split("_")[-1]))
    return os.path.join(experiment_dir, last_epoch)


def find_in_out_file_name(experiment_dir, prefix='test'):
    cfg = os.path.join(experiment_dir, 'config.yaml')
    if not os.path.exists(cfg):
        raise NotAnExperimentDir(f"{cfg} does not exist")
    with open(cfg) as f:
        train_fn = yaml.load(f, Loader=yaml.FullLoader)['train_file']
    inf = train_fn.replace('/train', f'/{prefix}')
    outf = os.path.join(experiment_dir, f'{prefix}.out')
    accf = os.path.join(experiment_dir, f'{prefix}.word_accuracy')
    return inf, outf, accf


def skip_dir(experiment_dir, test_out):
    if not os.path.exists(test_out):
        return False
    model_fn = find_last_model(experiment_dir)
    return os.path.getmtime(model_fn) < os.path.getmtime(test_out)


def compute_accuracy(reference, prediction):
    acc = 0
    samples = 0
    with open(reference) as r, open(prediction) as p:
        for rline in r:
            try:
                pline = next(p)
            except StopIteration:
                logging.error(f"Prediction file {prediction} shorter "
                              f"than reference {reference}")
                return acc / samples
            if not rline.strip() and not pline.strip():
                continue
            rlabel = rline.rstrip("\n").split("\t")[-1]
            plabel = pline.rstrip("\n").split("\t")[-1]
            acc += (rlabel == plabel)
            samples += 1
    return acc / samples


def parse_args():
    p = ArgumentParser()
    p.add_argument("experiment_dirs", nargs="+", type=str,
                   help="Experiment directory")
    p.add_argument("--run-on-dev", action="store_true")
    p.add_argument("--run-on-test", action="store_true")
    p.add_argument("--max-samples", default=None, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    for experiment_dir in args.experiment_dirs:
        if not os.path.isdir(experiment_dir):
            logging.info(f"{experiment_dir} not directory, skipping")
            continue
        if args.run_on_test:
            try:
                test_in, test_out, test_acc = find_in_out_file_name(experiment_dir, 'test')
                if not skip_dir(experiment_dir, test_out):
                    logging.info(f"Running inference on {experiment_dir}")
                    inf = Inference(experiment_dir, test_in, max_samples=args.max_samples)
                    with open(test_out, 'w') as f:
                        inf.run_and_print(f)
                    acc = compute_accuracy(test_in, test_out)
                    logging.info(f"{experiment_dir} test acc: {acc}")
                    with open(test_acc, 'w') as f:
                        f.write(f"{acc}\n")
                    gc.collect()
                    torch.cuda.empty_cache()
            except NotAnExperimentDir:
                logging.info(f"{experiment_dir}: no config.yaml, skipping")
        if args.run_on_dev:
            try:
                dev_in, dev_out, dev_acc = find_in_out_file_name(experiment_dir, 'dev')
                if not skip_dir(experiment_dir, dev_out):
                    inf = Inference(experiment_dir, dev_in, max_samples=args.max_samples)
                    with open(dev_out, 'w') as f:
                        inf.run_and_print(f)
                    acc = compute_accuracy(dev_in, dev_out)
                    logging.info(f"{experiment_dir} dev acc: {acc}")
                    with open(dev_acc, 'w') as f:
                        f.write(f"{acc}\n")
                    gc.collect()
                    torch.cuda.empty_cache()
            except NotAnExperimentDir:
                logging.info(f"{experiment_dir}: no config.yaml, skipping")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
