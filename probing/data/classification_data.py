#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os

from probing.data.base_data import BaseDataset, Vocab, DataFields


class ClassificationFields(DataFields):
    _fields = ('src', 'src_len', 'tgt')
    _alias = {
        'input': 'src',
        'input_len': 'src_len',
        'label': 'tgt',
    }
    _needs_vocab = ('src', 'tgt')


class ClassificationDataset(BaseDataset):

    unlabeled_data_class = 'UnlabeledClassificationDataset'
    data_recordclass = ClassificationFields
    constants = ['UNK', 'SOS', 'EOS', 'PAD']

    def extract_sample_from_line(self, line):
        src, tgt = line.split("\t")[:2]
        src = src.split(" ")
        return ClassificationFields(src, len(src)+2, tgt)

    def load_or_create_vocabs(self):
        vocabs = ClassificationFields(None, None, None)
        existing = getattr(self.config, 'vocab_src',
                           os.path.join(self.config.experiment_dir, 'vocab_src'))
        if os.path.exists(existing):
            vocabs.src = Vocab(file=existing, frozen=True)
        elif getattr(self.config, 'pretrained_embedding', False):
            vocabs.src = Vocab(file=None, constants=['UNK', 'SOS', 'EOS', 'PAD'])
            vocabs.src.load_word2vec_format(self.config.pretrained_embedding)
        else:
            vocabs.src = Vocab(constants=self.constants)
        existing = getattr(self.config, 'vocab_tgt',
                           os.path.join(self.config.experiment_dir, 'vocab_tgt'))
        if os.path.exists(existing):
            vocabs.tgt = Vocab(file=existing, frozen=True)
        else:
            vocabs.tgt = Vocab()
        self.vocabs = vocabs

    def decode(self, model_output):
        for i, sample in enumerate(self.raw):
            output = model_output[i].argmax().item()
            sample.tgt = self.vocabs.tgt.inv_lookup(output)

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\n".format(" ".join(sample.src), sample.tgt))


class UnlabeledClassificationDataset(ClassificationDataset):

    def extract_sample_from_line(self, line):
        src = line.split("\t")[0]
        src = src.split(" ")
        return ClassificationFields(src, len(src), None)


class NoSpaceClassificationDataset(ClassificationDataset):

    unlabeled_data_class = 'UnlabeledNoSpaceClassificationDataset'

    def extract_sample_from_line(self, line):
        src, tgt = line.split("\t")[:2]
        src = list(src)
        return ClassificationFields(src, len(src)+2, tgt)


class UnlabeledNoSpaceClassificationDataset(UnlabeledClassificationDataset):

    def extract_sample_from_line(self, line):
        src = line.split("\t")[0]
        return ClassificationFields(list(src), len(src)+2, None)

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\n".format("".join(sample.src), sample.tgt))
