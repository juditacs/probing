#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
import gzip
import logging
from sys import stdout
import numpy as np
from collections import OrderedDict, defaultdict

from probing.utils import find_ndim


class Vocab:
    # FIXME remove constants parameter
    def __init__(self, from_file=None, frozen=False, use_constants=False,
                 use_padding=False,
                 pad_token='[PAD]', unk_token='[UNK]', bos_token='[BOS]',
                 eos_token='[EOS]', constants=None,
                ):
        # FIXME add unk_symbol, bos_symbol etc.
        self.vocab = {}
        self.pad_token = None
        self.unk_token = None
        self.bos_token = None
        self.eos_token = None
        if from_file:
            with open(from_file) as f:
                for line in f:
                    fields = line.rstrip("\n").split("\t")
                    # If it's a constant
                    if len(fields) == 3:
                        symbol, id_, const = fields
                        id_ = int(id_)
                        # FIXME exception handling instead of assert
                        assert const in ('pad_token', 'unk_token', 'bos_token', 'eos_token')
                        setattr(self, const, symbol)
                        self.vocab[symbol] = id_
                    elif len(fields) == 2:
                        symbol, id_ = fields
                        id_ = int(id_)
                        self.vocab[symbol] = id_
            self.frozen = True
        else:
            self.vocab = {}
            if use_constants:
                self.pad_token = pad_token
                self.unk_token = unk_token
                self.bos_token = bos_token
                self.eos_token = eos_token
                self.vocab[self.pad_token] = 0
                self.vocab[self.unk_token] = 1
                self.vocab[self.bos_token] = 2
                self.vocab[self.eos_token] = 3
            if use_padding:
                self.pad_token = pad_token
                self.vocab[self.pad_token] = 0
                self.unk_token = unk_token
                self.vocab[self.unk_token] = 1
            self.frozen = False
        self.__inv_vocab = None

    def __getitem__(self, key):
        if self.frozen:
            if self.unk_token:
                return self.vocab.get(key, self.vocab[self.unk_token])
            return self.vocab[key]
        return self.vocab.setdefault(key, len(self.vocab))

    def __len__(self):
        return len(self.vocab)

    def __str__(self):
        return str(self.vocab)

    def __iter__(self):
        return iter(self.vocab)

    def items(self):
        return self.vocab.items()

    def keys(self):
        return self.vocab.keys()

    def inv_lookup(self, key):
        if self.__inv_vocab is None:
            self.__inv_vocab = {i: s for s, i in self.vocab.items()}
        return self.__inv_vocab.get(key, self.unk_token)

    def save(self, fn):
        with open(fn, 'w') as f:
            offset = 0
            for constant in ('pad_token', 'unk_token',
                             'bos_token', 'eos_token'):
                if getattr(self, constant, None) is not None:
                    field = getattr(self, constant)
                    id_ = self.vocab[field]
                    f.write(f"{field}\t{id_}\t{constant}\n")
                    offset += 1
            for symbol, id_ in self.vocab.items():
                if self.pad_token and id_ < offset:
                    continue
                f.write(f"{symbol}\t{id_}\n")

    def encode(self, data):
        ndim = find_ndim(data)
        if ndim == 0:
            return self[data]
        if ndim == 1:
            if self.bos_token is not None:
                data = [self.bos_token] + data + [self.eos_token]
            return [self[d] for d in data]
        # if ndim == 2:
        #     indexed = []
        #     bos = self[self.bos_token]
        #     eos = self[self.eos_token]
        #     for row in data:
        #         indexed.append(
        #             [bos] + [self[s] for s in row] + [eos]
        #         )
        #     return indexed
        raise ValueError(f"Data dimension ({ndim}) too high. Input: {data}")

    def pad(self, data):
        ndim = find_ndim(data)
        # If it's 2D and it needs padding.
        if ndim == 2 and self.pad_token:
            maxlen = max(len(d) for d in data)
            padded = []
            pad = self[self.pad_token]
            for row in data:
                padded.append(
                    list(row) + [pad] * (maxlen - len(row))
                )
            return padded
        else:
            return data


class DataFields:
    _fields = ('src', 'tgt')
    _alias = {}
    needs_vocab = ()
    needs_constants = ()
    needs_padding = ()

    def __init__(self, *args, **kwargs):
        for field in self._fields:
            setattr(self, field, None)
        for i, arg in enumerate(args):
            setattr(self, self._fields[i], arg)
        for kw, arg in kwargs.items():
            setattr(self, kw, arg)

    def __setattr__(self, attr, value):
        if attr in self._alias:
            attr = self._alias[attr]
        if attr not in self._fields:
            raise AttributeError(
                f"{self.__class__.__name__} has no attribute {attr}")
        return super().__setattr__(attr, value)

    def __getitem__(self, field):
        return getattr(self, field)

    def __iter__(self):
        for field in self._fields:
            yield getattr(self, field)

    def __getattr__(self, attr):
        if attr in self._alias:
            return getattr(self, self._alias[attr])
        raise AttributeError(
            f"{self.__class__.__name__} has no attribute {attr}")

    def __len__(self):
        return len(self._fields)

    def __repr__(self):
        out = []
        none_fields = []
        for field in self._fields:
            val = getattr(self, field)
            if val is None:
                none_fields.append(field)
            else:
                out.append(f"{field}={val!r}")
        if none_fields:
            return f"{self.__class__.__name__}({', '.join(out)}, " \
                    "None fields: {', '.join(none_fields)}"
        return f"{self.__class__.__name__}({', '.join(out)})"

    @classmethod
    def initialize_all(cls, initializer):
        d = cls()
        for field in d._fields:
            setattr(d, field, initializer())
        return d

    def _asdict(self):
        return OrderedDict((k, getattr(self, k, None)) for k in self._fields)

    def keys(self):
        for field in self._fields:
            value = getattr(self, field, None)
            if value is not None:
                yield field

    def values(self):
        for field in self._fields:
            value = getattr(self, field, None)
            if value:
                yield value

    def items(self):
        for field in self._fields:
            value = getattr(self, field, None)
            if value is not None:
                yield field, value


class BaseDataset:

    def __init__(self, config, stream_or_file, max_samples=None,
                 share_vocabs_with=None, is_unlabeled=False):
        self.config = config
        self.is_unlabeled = is_unlabeled
        self.max_samples = max_samples
        if share_vocabs_with is None:
            self.load_or_create_vocabs()
        else:
            self.vocabs = share_vocabs_with.vocabs
            for vocab in self.vocabs:
                if vocab:
                    vocab.frozen = True
        self.load_stream_or_file(stream_or_file)
        self.to_idx()
        self.sort_data_by_length()

    def load_or_create_vocabs(self):
        vocab_pre = os.path.join(self.config.experiment_dir, 'vocab_')
        vocabs = {}
        for field in self.datafield_class.needs_vocab:
            vocab_fn = f"{vocab_pre}{field}"
            if os.path.exists(vocab_fn):
                vocabs[field] = Vocab(from_file=vocab_fn)
            else:
                needs_constants = field in self.datafield_class.needs_constants
                needs_padding = field in self.datafield_class.needs_padding
                # FIXME I don't like the mix in naming convention use vs needs
                vocabs[field] = Vocab(use_constants=needs_constants,
                                      use_padding=needs_padding)
        self.vocabs = self.datafield_class(**vocabs)

    def load_stream_or_file(self, stream_or_file):
        if isinstance(stream_or_file, str):
            if os.path.splitext(stream_or_file)[-1] == '.gz':
                with gzip.open(stream_or_file, 'rt') as stream:
                    self.load_stream(stream)
            else:
                with open(stream_or_file) as stream:
                    self.load_stream(stream)
        else:
            self.load_stream(stream_or_file)

    def load_stream(self, stream):
        self.raw = []
        for line in stream:
            sample = self.extract_sample_from_line(line.rstrip('\n'))
            if not self.ignore_sample(sample):
                self.raw.append(sample)
            if self.max_samples is not None and len(self.raw) >= self.max_samples:
                logging.info("Reached max samples ({self.max_samples}) before "
                             "finishing the file.")
                break

    def extract_sample_from_line(self, line):
        raise NotImplementedError("Subclass of BaseData must define "
                                  "extract_sample_from_line")

    def ignore_sample(self, sample):
        return False

    def to_idx(self):
        mtx = defaultdict(list)
        for sample in self.raw:
            for field, value in sample.items():
                if field in self.datafield_class.needs_vocab:
                    mtx[field].append(self.vocabs[field].encode(value))
                else:
                    mtx[field].append(value)
        self.mtx = self.datafield_class(**mtx)

    def batched_iter(self, batch_size):
        starts = list(range(0, len(self), batch_size))
        if self.is_unlabeled is False and self.config.shuffle_batches:
            np.random.shuffle(starts)
        for start in starts:
            self._start = start
            end = start + batch_size
            batch = {}
            for field, mtx in self.mtx.items():
                if field in self.datafield_class.needs_padding:
                    batch[field] = self.vocabs[field].pad(mtx[start:end])
                else:
                    batch[field] = mtx[start:end]
            yield self.datafield_class(**batch)

    def sort_data_by_length(self, sort_field=None):
        if self.is_unlabeled:
            return
        if self.config.sort_data_by_length is False:
            return
        if hasattr(self.mtx, 'input_len'):
            order = np.argsort(-np.array(self.mtx.input_len))
        else:
            order = np.argsort([-len(m) for m in self.mtx.input])
        self.order = order
        ordered = []
        for m in self.mtx:
            if m is None or len(m) == 0 or m[0] is None:
                ordered.append(m)
            else:
                ordered.append([m[idx] for idx in order])
        self.mtx = self.datafield_class(*ordered)

    def decode_and_print(self, model_output, stream=stdout):
        self.decode(model_output)
        self.print_raw(stream)

    def decode(self, model_output):
        raise NotImplementedError("Subclass of BaseData must define "
                                  "decode")

    def print_raw(self, stream):
        for sample in self.raw:
            self.print_sample(sample, stream)

    def print_sample(self, sample, stream):
        raise NotImplementedError("Subclass of BaseData must define "
                                  "print_sample")

    def save_vocabs(self):
        vocab_list = list(self.vocabs._asdict().keys())
        for vocab_name in vocab_list:
            vocab = getattr(self.vocabs, vocab_name)
            if vocab is None:
                continue
            path = os.path.join(
                self.config.experiment_dir, f'vocab_{vocab_name}')
            vocab.save(path)

    def __len__(self):
        return len(self.raw)
