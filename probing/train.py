#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import logging

from probing.experiment import Experiment


def parse_args():
    p = ArgumentParser()
    p.add_argument("-c", "--config", type=str,
                   help="YAML config file location")
    p.add_argument("--load-model", type=str, default=None,
                   help="Continue training this model. The model"
                   " must have the same parameters.")
    p.add_argument("--train-file", type=str, default=None)
    p.add_argument("--dev-file", type=str, default=None)
    p.add_argument("--params", type=str, default=None)
    p.add_argument("--debug", action="store_true",
                   help="Do not raise exception when the working "
                   "directory is not clean.")
    return p.parse_args()


def parse_param_str(params):
    param_d = {}
    for p in params.split(','):
        key, val = p.split('=')
        try:
            param_d[key] = int(val)
        except ValueError:
            try:
                param_d[key] = float(val)
            except ValueError:
                param_d[key] = val
    return param_d


def main():
    args = parse_args()
    if args.params:
        override_params = parse_param_str(args.params)
    else:
        override_params = None
    with Experiment(args.config, train_data=args.train_file,
                    dev_data=args.dev_file,
                    override_params=override_params,
                    debug=args.debug) as e:
        e.run()

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
