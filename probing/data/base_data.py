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
from collections import OrderedDict


class Vocab:
    def __init__(self, file=None, frozen=False, constants=None):
        self.vocab = {}
        self.constants = {}
        if file is not None:
            with open(file) as f:
                for line in f:
                    fd = line.rstrip("\n").split("\t")
                    if len(fd) == 3:
                        symbol, id_, const = fd
                        self.constants[const] = int(symbol)
                        self.vocab[self.constants[const]] = int(id_)
                        setattr(self, const, int(id_))
                    else:
                        symbol, id_ = fd
                        self.vocab[symbol] = int(id_)
        else:
            if constants is not None:
                for const in constants:
                    setattr(self, const, len(self.constants))
                    self.constants[const] = len(self.constants)
                    self.vocab[self.constants[const]] = len(self.vocab)
        self.frozen = frozen
        self.__inv_vocab = None

    def __getitem__(self, key):
        if self.frozen:
            if 'UNK' in self.constants:
                return self.vocab.get(key, self.UNK)
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
            for const, idx in self.constants.items():
                self.__inv_vocab[idx] = const
        return self.__inv_vocab.get(key, 'UNK')

    def save(self, fn):
        with open(fn, 'w') as f:
            inv_const = {i: v for v, i in self.constants.items()}
            for symbol, id_ in sorted(self.vocab.items(), key=lambda x: x[1]):
                if symbol in inv_const:
                    f.write('{}\t{}\t{}\n'.format(
                        symbol, id_, inv_const[symbol]))
                else:
                    f.write('{}\t{}\n'.format(symbol, id_))

    def load_word2vec_format(self, fn):
        with open(fn) as f:
            first = next(f).rstrip('\n').split(" ")
            if len(first) != 2:
                word = first[0]
                self.vocab[word] = len(self.vocab)
            for line in f:
                fd = line.rstrip('\n').split(" ")
                word = fd[0]
                self.vocab[word] = len(self.vocab)
        self.frozen = True

    def post_load_embedding(self, fn):
        # constants such as UNK are not accounted for
        assert not self.constants
        if fn.endswith('.gz'):
            stream = gzip.open(fn, 'rt')
        else:
            stream = open(fn, 'rt')
        embedding = []
        emb_vocab = {}
        first = next(stream)
        fd = first.split(" ")
        if len(fd) > 2:
            word = fd[0]
            if word in self.vocab:
                embedding.append(list(map(float(fd[1:]))))
                emb_vocab[word] = len(emb_vocab)
        for line in stream:
            word = fd[0]
            if word in self.vocab:
                embedding.append(list(map(float(fd[1:]))))
                emb_vocab[word] = len(emb_vocab)
        stream.close()
        self.vocab = emb_vocab
        self.embedding = embedding
        self.frozen = True
        return embedding


class DataFields:
    _fields = ('src', 'tgt')
    _alias = {}

    def __init__(self, *args, **kwargs):
        for field in self._fields:
            setattr(self, field, None)
        for i, arg in enumerate(args):
            setattr(self, self._fields[i], arg)
        for kw, arg in kwargs.items():
            setattr(self, kw, arg)

    def __setattr__(self, attr, value):
        if attr not in self._fields:
            raise AttributeError("{} has no attribute {}".format(
                self.__class__.__name__, attr))
        return super().__setattr__(attr, value)

    def __getitem__(self, idx):
        return getattr(self, self._fields[idx])

    def __setitem__(self, idx, value):
        return setattr(self, self._fields[idx], value)

    def __iter__(self):
        for field in self._fields:
            yield getattr(self, field)

    def get_existing_fields_and_values(self):
        for field in self._fields:
            val = getattr(self, field)
            if val is not None:
                yield field, val

    def __getattr__(self, attr):
        if attr in self._alias:
            return getattr(self, self._alias[attr])
        raise AttributeError("{} does not have a {} field".format(
            self.__class__.__name__, attr))

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
                out.append("{}={}".format(field, repr(val)))
        if none_fields:
            return "{}({}, None fields: {})".format(
                self.__class__.__name__, ", ".join(out),
                ", ".join(none_fields))
        return "{}({})".format(self.__class__.__name__, ", ".join(out))


    @classmethod
    def initialize_all(cls, initializer):
        d = cls()
        for field in d._fields:
            setattr(d, field, initializer())
        return d

    def _asdict(self):
        return OrderedDict((k, getattr(self, k, None)) for k in self._fields)


class BaseDataset:

    def __init__(self, config, stream_or_file, max_samples=None, share_vocabs_with=None):
        self.config = config
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
        need_vocab = getattr(self.data_recordclass, '_needs_vocab', None)
        if need_vocab is None:
            need_vocab = list(self.data_recordclass()._asdict().keys())
        need_constants = getattr(self.data_recordclass, '_needs_constants', None)
        if need_constants is None:
            need_constants = list(self.data_recordclass()._asdict().keys())
        self.vocabs = self.data_recordclass()
        for field in need_vocab:
            vocab_fn = getattr(self.config, 'vocab_{}'.format(field),
                               vocab_pre+field)
            if os.path.exists(vocab_fn):
                setattr(self.vocabs, field, Vocab(file=vocab_fn, frozen=True))
            else:
                if field in need_constants:
                    setattr(self.vocabs, field, Vocab(constants=self.constants))
                else:
                    setattr(self.vocabs, field, Vocab(constants=[]))

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
                logging.info("Read max_samples ({}) before finishing the file.".format(
                    self.max_samples))
                break

    def extract_sample_from_line(self, line):
        raise NotImplementedError("Subclass of BaseData must define "
                                  "extract_sample_from_line")

    def ignore_sample(self, sample):
        return False

    def to_idx(self):
        mtx = [[] for _ in range(len(self.raw[0]))]
        for sample in self.raw:
            for i, part in enumerate(sample):
                if part is None:  # unlabeled data
                    mtx[i] = None
                elif isinstance(part, int):
                    mtx[i].append(part)
                elif isinstance(part, str):
                    mtx[i].append(self.vocabs[i][part])
                else:
                    vocab = self.vocabs[i]
                    idx = []
                    if 'SOS' in vocab.constants:
                        idx.append(vocab.SOS)
                    idx.extend([vocab[s] for s in part])
                    if 'EOS' in vocab.constants:
                        idx.append(vocab.EOS)
                    mtx[i].append(idx)
        self.mtx = self.create_recordclass(*mtx)

    def sort_data_by_length(self, sort_field=None):
        if self.is_unlabeled:
            return
        if self.config.sort_data_by_length is False:
            return
        if sort_field is None:
            sort_field = 'src_len'
        if hasattr(self.mtx, sort_field):
            order = np.argsort(-np.array(getattr(self.mtx, sort_field)))
        else:
            order = np.argsort([-len(m) for m in self.mtx.src])
        self.order = order
        ordered = []
        for m in self.mtx:
            if m is None or len(m) == 0 or m[0] is None:
                ordered.append(m)
            else:
                ordered.append([m[idx] for idx in order])
        self.mtx = self.create_recordclass(*ordered)

    @property
    def is_unlabeled(self):
        return self.raw[0].tgt is None or "Unlabeled" in self.__class__.__name__

    def create_recordclass(self, *data):
        return self.__class__.data_recordclass(*data)

    def decode_and_print(self, model_output, stream=stdout):
        self.decode(model_output)
        self.print_raw(stream)

    def decode(self, model_output):
        if hasattr(self, 'order'):
            new_order = np.argsort(self.order)
            model_output = np.array(model_output)[new_order]
        for i, sample in enumerate(self.raw):
            output = list(model_output[i])
            decoded = [self.vocabs.tgt.inv_lookup(s)
                       for s in output]
            if decoded[0] == 'SOS':
                decoded = decoded[1:]
            if 'EOS' in decoded:
                decoded = decoded[:decoded.index('EOS')]
            self.raw[i].tgt = decoded

    def print_raw(self, stream):
        for sample in self.raw:
            self.print_sample(sample, stream)

    def print_sample(self, sample, stream):
        stream.write("{}\n".format("\t".join(" ".join(s) for s in sample)))

    def save_vocabs(self):
        # FIXME recordclass removal
        if hasattr(self.vocabs, '_fields'):
            vocab_list = list(self.vocabs._fields)
        else:
            vocab_list = list(self.vocabs._asdict().keys())
        for vocab_name in vocab_list:
            vocab = getattr(self.vocabs, vocab_name)
            if vocab is None:
                continue
            path = os.path.join(
                self.config.experiment_dir, 'vocab_{}'.format(vocab_name))
            vocab.save(path)

    def batched_iter(self, batch_size):
        starts = list(range(0, len(self), batch_size))
        if self.is_unlabeled is False and self.config.shuffle_batches:
            np.random.shuffle(starts)
        for start in starts:
            self._start = start
            end = start + batch_size
            batch = []
            for i, mtx in enumerate(self.mtx):
                if mtx is None or len(mtx) == 0:
                    batch.append(None)
                elif isinstance(mtx[0], (int, np.integer)):
                    batch.append(mtx[start:end])
                else:
                    PAD = self.vocabs[i].PAD
                    this_batch = mtx[start:end]
                    maxlen = max(len(d) for d in this_batch)
                    padded = [
                        sample + [PAD] * (maxlen-len(sample))
                        for sample in this_batch
                    ]
                    batch.append(padded)
            yield self.create_recordclass(*batch)

    def __len__(self):
        return len(self.raw)
