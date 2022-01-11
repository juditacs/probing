#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import json

from probing.config import Config


MINIMAL_CONFIG = {}


def create_tmp_yaml(basedir):
    cfg = MINIMAL_CONFIG.copy()
    cfg["experiment_dir"] = str(basedir)
    with open(basedir / "config.yaml", "w", encoding="utf-8") as f:
        json.dump(cfg, f)


def test_create_minimal_config(tmp_path):
    create_tmp_yaml(tmp_path)
    Config.from_yaml(tmp_path / "config.yaml")


def test_create_minimal_config_stream(tmp_path):
    create_tmp_yaml(tmp_path)
    with open(tmp_path / "config.yaml") as f:
        Config.from_yaml(f)
