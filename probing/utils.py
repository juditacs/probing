#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import subprocess
import os
import logging


class UncleanWorkingDirectoryException(Exception):
    pass


def run_command(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    stdout, stderr = p.communicate()
    stdout = stdout.decode('utf8')
    stderr = stderr.decode('utf8')
    return stdout, stderr


def check_and_get_commit_hash(debug):
    src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stdout, stderr = run_command(
        "cd {}; git status --porcelain".format(src_path))

    unstaged = []
    staged = []
    for line in stdout.split("\n"):
        if not line.strip():
            continue
        x = line[0]
        y = line[1]
        if x == '?' and y == '?':
            continue
        filename = line[2:].strip().split(" ")[0]
        if x == ' ' and (y == 'M' or y == 'D'):
            unstaged.append(filename)
        elif x in 'MADRC':
            staged.append(filename)
        else:
            raise ValueError("Unable to parse status message")
    if len(unstaged) > 0 or len(staged) > 0:
        CRED = '\033[91m'
        CGREEN = '\033[32m'
        CEND = '\033[0m'
        error_msg = []
        if len(unstaged) > 0:
            error_msg.append("Unstaged files:{}".format(CRED))
            error_msg.extend(unstaged)
            error_msg[-1] += CEND
        if len(staged) > 0:
            error_msg.append("Staged but not committed:{}".format(CGREEN))
            error_msg.extend(staged)
            error_msg[-1] += CEND
        error_msg = "\n".join(error_msg)
        if debug:
            logging.warning(error_msg)
        else:
            raise UncleanWorkingDirectoryException(error_msg)

    commit_hash, _ = run_command(
        "cd {}; git log --pretty=format:'%H' -n 1".format(
            src_path))
    return commit_hash


def find_last_model(experiment_dir):
    model_pre = os.path.join(experiment_dir, 'model')
    if os.path.exists(model_pre):
        return model_pre
    saves = filter(lambda f: f.startswith(
        'model.epoch_'), os.listdir(experiment_dir))
    last_epoch = max(saves, key=lambda f: int(f.split("_")[-1]))
    return os.path.join(experiment_dir, last_epoch)
