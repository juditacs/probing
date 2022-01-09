#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2021 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

from argparse import ArgumentParser
import pandas as pd

from probing.utils import load_experiment_dirs


def parse_args():
    p = ArgumentParser()
    p.add_argument('input_dirs', nargs='+', type=str)
    p.add_argument('-c', '--concat', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    all_exps = []
    for indir in args.input_dirs:
        print("Loading directory {}".format(indir))
        exps = load_experiment_dirs(indir)
        all_exps.append(exps)
        print("Loaded {} experiments".format(len(exps)))
    if args.concat:
        all_exps = pd.concat(all_exps, sort=True).reset_index(drop=True)
        all_exps.to_csv(args.concat, sep="\t", index=False)

if __name__ == '__main__':
    main()
