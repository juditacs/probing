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
    p.add_argument("--train-file", type=str, default=None)
    p.add_argument("--dev-file", type=str, default=None)
    p.add_argument("--debug", action="store_true",
                   help="Do not raise exception when the working "
                   "directory is not clean.")
    return p.parse_args()


def main():
    args = parse_args()
    with Experiment(args.config, train_data=args.train_file,
                    dev_data=args.dev_file,
                    debug=args.debug) as e:
        e.run()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
