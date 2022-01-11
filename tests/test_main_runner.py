#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import json
import pandas as pd

from probing.experiment import Experiment


BASE_CONFIG = {
    "epochs": 2,
    "dataset_class": "SLSTMDataset",
    "model": "MidSequenceClassifier",
    "hidden_size": 8,
    "num_layers": 1,
    "embedding_size": 10,
    "mlp_layers": [10],
    "mlp_nonlinearity": "ReLU",
}

SAMPLES = pd.DataFrame([
    ("b", "b", 0, "Acc"),
    ("a b", "b", 1, "Nom"),
    # ("a b c", "a", 0, "Acc"),
    # ("a b c e d f", "c", 2, "Nom"),
], columns=["sentence", "target", "target_idx", "label"])


def create_train_file(tmp_path, name, size):
    fn = tmp_path / name
    all_samples = pd.concat([SAMPLES] * int(size / len(SAMPLES) + 1))
    all_samples.iloc[:size].to_csv(fn, sep="\t", index=False, header=False)
    # FIXME remove str conversion when all path handling is done with pathlib
    return str(fn)


def create_tmp_yaml(basedir):
    cfg = BASE_CONFIG.copy()
    cfg["experiment_dir"] = str(basedir)
    with open(basedir / "config.yaml", "w", encoding="utf-8") as f:
        json.dump(cfg, f)


def test_experiment_setup(tmp_path):
    create_tmp_yaml(tmp_path)
    config_fn = tmp_path / "config.yaml"
    train_fn = create_train_file(tmp_path, name="train.tsv", size=10)
    dev_fn = create_train_file(tmp_path, name="dev.tsv", size=5)
    Experiment(config_fn, train_data=train_fn, dev_data=dev_fn)
    assert (tmp_path / "0000").exists()
    assert (tmp_path / "0000" / "vocab_input").exists()
    assert (tmp_path / "0000" / "vocab_label").exists()
    # TODO check vocab size, why is space part of the vocab


def check_create_parent_dir(tmp_path):
    cfg = BASE_CONFIG.copy()
    exp_dir = tmp_path / "subdir1" / "subdir2"
    cfg["experiment_dir"] = str(exp_dir)
    with open(tmp_path / "config.yaml", "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    config_fn = tmp_path / "config.yaml"
    train_fn = create_train_file(tmp_path, name="train.tsv", size=10)
    dev_fn = create_train_file(tmp_path, name="dev.tsv", size=5)
    Experiment(config_fn, train_data=train_fn, dev_data=dev_fn)
    assert (exp_dir / "0000").exists()
    assert (exp_dir / "0000" / "vocab_input").exists()
    assert (exp_dir / "0000" / "vocab_label").exists()
