#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import logging
import os

from probing.config import Config
from probing.experiment import Experiment


def parse_args():
    p = ArgumentParser()
    p.add_argument("-c", "--config", type=str,
                   help="YAML config file location")
    p.add_argument("-e", "--experiment-dir", type=str,
                   help="Save identical experiment to this directory.")
    p.add_argument("-n", "--no-subdir", action="store_true",
                   help="Do not generate empty subdir.")
    return p.parse_args()


def get_config_name(config_or_dir):
    if os.path.splitext(config_or_dir)[-1] == ".yaml":
        return config_or_dir
    return os.path.join(config_or_dir, "config.yaml")


def main():
    args = parse_args()
    generate_empty_subdir = not args.no_subdir
    override_params = {
        'generate_empty_subdir': generate_empty_subdir,
        'experiment_dir': args.experiment_dir,
    }
    cfg_fn = get_config_name(args.config)
    cfg = Config.from_yaml(cfg_fn, override_params=override_params)
    with Experiment(config=cfg) as e:
        e.run()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
