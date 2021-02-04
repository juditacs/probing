#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
import gzip
import numpy as np
import logging

from transformers import AutoTokenizer

from probing.data.base_data import BaseDataset, Vocab, DataFields


class WordOnlyFields(DataFields):
    _fields = ('sentence', 'target_word', 'target_word_len', 'target_idx',
               'label')
    _alias = {
        'input': 'target_word',
        'input_len': 'target_word_len',
        'src_len': 'target_word_len',
        'tgt': 'label',
    }
    _needs_vocab = ('target_word', 'label')


class EmbeddingOnlyFields(DataFields):
    _fields = ('sentence', 'target_word', 'target_word_idx', 'label')
    _alias = {
        'tgt': 'label',
        'src': 'target_word',
    }
    _needs_vocab = ('label', )


class TokenInSequenceProberFields(DataFields):
    _fields = (
        'raw_sentence', 'raw_target', 'raw_idx',
        'tokens', 'num_tokens', 'target_idx', 'label', 'token_starts',
    )
    _alias = {
        'tgt': 'label',
        # 'src_len': 'num_tokens',
        'input_len': 'num_tokens'}
    # token_starts needs a vocabulary because we manually set PAD=1000
    _needs_vocab = ('tokens', 'label', 'token_starts')
    _needs_constants = ('tokens', )


class MidSequenceProberFields(DataFields):
    _fields = (
        'raw_sentence', 'raw_target', 'raw_idx',
        'input', 'input_len', 'target_idx', 'label', 'target_ids',
    )
    _alias = {'tgt': 'label', 'src_len': 'input_len'}
    _needs_vocab = ('input', 'label', 'target_ids')
    _needs_constants = ('input', )


class SequenceClassificationWithSubwordsDataFields(DataFields):
    _fields = (
        'raw_sentence', 'labels',
        'sentence_len', 'subwords', 'sentence_subword_len', 'token_starts',
    )
    _alias = {'input': 'subwords',
              'input_len': 'sentence_subword_len',
              'tgt': 'labels'}
    _needs_vocab = ('labels', )


class Embedding:
    def __init__(self, embedding_file, filter=None):
        self.filter_ = filter
        if embedding_file.endswith('.gz'):
            with gzip.open(embedding_file, 'rt') as f:
                self.load_stream(f)
        else:
            with open(embedding_file, 'rt') as f:
                self.load_stream(f)

    def load_stream(self, stream):
        self.mtx = []
        self.vocab = {}
        for line in stream:
            fd = line.strip().split(" ")
            if len(fd) == 2:
                continue
            word = fd[0]
            if self.filter_ and word not in self.filter_:
                continue
            self.vocab[word] = len(self.mtx)
            self.mtx.append(list(map(float, fd[1:])))
        self.mtx = np.array(self.mtx)

    def __len__(self):
        return self.mtx.shape[0]

    def __getitem__(self, key):
        if key not in self.vocab:
            return self.mtx[0]
        return self.mtx[self.vocab[key]]

    @property
    def embedding_dim(self):
        return self.mtx.shape[1]


class EmbeddingProberDataset(BaseDataset):
    unlabeled_data_class = 'UnlabeledEmbeddingProberDataset'
    constants = []
    data_recordclass = EmbeddingOnlyFields

    def load_or_create_vocabs(self):
        vocab_pre = os.path.join(self.config.experiment_dir, 'vocab_')
        needs_vocab = getattr(self.data_recordclass, '_needs_vocab',
                              self.data_recordclass._fields)
        self.vocabs = self.data_recordclass()
        for field in needs_vocab:
            vocab_fn = getattr(self.config, 'vocab_{}'.format(field),
                               vocab_pre+field)
            if field == 'label':
                constants = []
            else:
                constants = ['SOS', 'EOS', 'PAD', 'UNK']
            if os.path.exists(vocab_fn):
                setattr(self.vocabs, field, Vocab(file=vocab_fn, frozen=True))
            else:
                setattr(self.vocabs, field, Vocab(constants=constants))

    def to_idx(self):
        vocab = set(r.target_word for r in self.raw)
        if self.config.embedding == 'discover':
            language = self.config.train_file.split("/")[-2]
            emb_fn = os.path.join(os.environ['HOME'], 'resources',
                                  'fasttext', language, 'common.vec')
            self.config.embedding = emb_fn
        else:
            emb_fn = self.config.embedding
        self.embedding = Embedding(emb_fn, filter=vocab)
        self.embedding_size = self.embedding.embedding_dim
        word_vecs = []
        labels = []
        for r in self.raw:
            word_vecs.append(self.embedding[r.target_word])
            labels.append(self.vocabs.label[r.label])
        self.mtx = EmbeddingOnlyFields(
            target_word=word_vecs,
            label=labels
        )

    def extract_sample_from_line(self, line):
        fd = line.rstrip("\n").split("\t")
        sent, target, idx = fd[:3]
        if len(fd) > 3:
            label = fd[3]
        else:
            label = None
        return EmbeddingOnlyFields(
            sentence=sent,
            target_word=target,
            target_word_idx=int(idx),
            label=label
        )

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\t{}\t{}\n".format(
            sample.sentence, sample.target_word,
            sample.target_word_idx, sample.label
        ))

    def decode(self, model_output):
        for i, sample in enumerate(self.raw):
            output = model_output[i].argmax().item()
            sample.label = self.vocabs.label.inv_lookup(output)

    def batched_iter(self, batch_size):
        starts = list(range(0, len(self), batch_size))
        if self.is_unlabeled is False and self.config.shuffle_batches:
            np.random.shuffle(starts)
        for start in starts:
            end = start + batch_size
            yield EmbeddingOnlyFields(
                target_word=self.mtx.target_word[start:end],
                label=self.mtx.label[start:end]
            )


class UnlabeledEmbeddingProberDataset(EmbeddingProberDataset):
    pass



class WordOnlySentenceProberDataset(BaseDataset):

    data_recordclass = WordOnlyFields
    unlabeled_data_class = 'UnlabeledWordOnlySentenceProberDataset'
    constants = []

    def load_or_create_vocabs(self):
        vocab_pre = os.path.join(self.config.experiment_dir, 'vocab_')
        needs_vocab = getattr(self.data_recordclass, '_needs_vocab',
                              self.data_recordclass._fields)
        self.vocabs = self.data_recordclass()
        for field in needs_vocab:
            vocab_fn = getattr(self.config, 'vocab_{}'.format(field),
                               vocab_pre+field)
            if field == 'label':
                constants = []
            else:
                constants = ['SOS', 'EOS', 'PAD', 'UNK']
            if os.path.exists(vocab_fn):
                setattr(self.vocabs, field, Vocab(file=vocab_fn, frozen=True))
            else:
                setattr(self.vocabs, field, Vocab(constants=constants))

    def extract_sample_from_line(self, line):
        fd = line.rstrip("\n").split("\t")
        if len(line) > 3:
            sent, target, idx, label = fd[:4]
        else:
            sent, target, idx = fd[:3]
            label = None
        idx = int(idx)
        return WordOnlyFields(
            sentence=sent,
            target_word=target,
            target_idx=idx,
            target_word_len=len(target),
            label=label,
        )

    def to_idx(self):
        words = []
        lens = []
        labels = []
        if self.config.use_global_padding:
            maxlen = self.get_max_seqlen()
            longer = sum(s.target_word_len > maxlen for s in self.raw)
            if longer > 0:
                logging.warning('{} elements longer than maxlen'.format(longer))
        for sample in self.raw:
            idx = list(self.vocabs.target_word[c] for c in sample.target_word)
            if self.config.use_global_padding:
                idx = idx[:maxlen-2]
                idx = [self.vocabs.target_word.SOS] + \
                    idx + [self.vocabs.target_word.EOS]
                idx = idx + [self.vocabs.target_word.PAD] * (maxlen - len(idx))
                lens.append(maxlen)
            else:
                idx = [self.vocabs.target_word.SOS] + \
                    idx + [self.vocabs.target_word.EOS]
                lens.append(len(idx))
            words.append(idx)
            labels.append(self.vocabs.label[sample.label])
        self.mtx = WordOnlyFields(
            target_word=words, target_word_len=lens, label=labels
        )

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\t{}\t{}\n".format(
            sample.sentence, sample.target_word,
            sample.target_idx, sample.label
        ))

    def decode(self, model_output):
        for i, sample in enumerate(self.raw):
            output = model_output[i].argmax().item()
            sample.label = self.vocabs.label.inv_lookup(output)

    def __len__(self):
        return len(self.raw)

    def get_max_seqlen(self):
        if hasattr(self.config, 'max_seqlen'):
            return self.config.max_seqlen
        return max(s.target_word_len for s in self.raw) + 2


class UnlabeledWordOnlySentenceProberDataset(WordOnlySentenceProberDataset):
    def is_unlabeled(self):
        return True


# TODO replace MidSentenceProberDataset with TokenInSequenceProberFields
class MidSentenceProberDataset(BaseDataset):
    unlabeled_data_class = 'UnlabeledMidSentenceProberDataset'
    data_recordclass = MidSequenceProberFields
    constants = ['SOS', 'EOS', 'UNK', 'PAD']

    def extract_sample_from_line(self, line):
        raw_sent, raw_target, raw_idx, label = line.rstrip("\n").split("\t")
        raw_idx = int(raw_idx)
        input = list(raw_sent)
        words = raw_sent.split(' ')
        if self.config.probe_first_char:
            target_idx = sum(len(w) for w in words[:raw_idx]) + raw_idx
        else:
            target_idx = sum(len(w) for w in words[:raw_idx]) + raw_idx + len(raw_target) - 1
        return self.data_recordclass(
            raw_sentence=raw_sent,
            raw_target=raw_target,
            raw_idx=raw_idx,
            input=input,
            input_len=len(input),
            target_idx=target_idx,
            label=label,
        )

    def to_idx(self):
        mtx = self.data_recordclass(input=[], input_len=[],
                                    target_idx=[], label=[])
        SOS = self.vocabs.input['SOS']
        EOS = self.vocabs.input['EOS']
        for sample in self.raw:
            mtx.label.append(self.vocabs.label[sample.label])
            mtx.input_len.append(sample.input_len)
            mtx.target_idx.append(sample.target_idx)
            mtx.input.append(
                [SOS] + [self.vocabs.input[s] for s in sample.input] + [EOS]
            )
        self.mtx = mtx

    def decode(self, model_output):
        for i, sample in enumerate(self.raw):
            output = np.argmax(model_output[i])
            self.raw[i].label = self.vocabs.label.inv_lookup(output)

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\t{}\t{}\n".format(
            sample.raw_sentence, sample.raw_target, sample.raw_idx, sample.label
        ))


class UnlabeledMidSentenceProberDataset(MidSentenceProberDataset):

    @property
    def is_unlabeled(self):
        return True


class SequenceClassificationWithSubwords(BaseDataset):
    unlabeled_data_class = 'UnlabeledSequenceClassificationWithSubwords'
    data_recordclass = SequenceClassificationWithSubwordsDataFields
    constants = ['UNK']

    def __init__(self, config, stream_or_file, max_samples=None,
                 share_vocabs_with=None, **kwargs):
        self.config = config
        self.max_samples = max_samples
        if share_vocabs_with is None:
            self.load_or_create_vocabs()
        else:
            self.vocabs = share_vocabs_with.vocabs
            for vocab in self.vocabs:
                if vocab:
                    vocab.frozen = True
        global_key = f'{self.config.model_name}_tokenizer'
        if global_key in globals():
            self.tokenizer = globals()[global_key]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            globals()[global_key] = self.tokenizer
        self.load_or_create_vocabs()
        self.load_stream_or_file(stream_or_file)
        self.to_idx()
        self.PAD = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]

    def load_stream(self, stream):
        self.raw = []
        sent = []
        for line in stream:
            if not line.strip():
                if sent:
                    sample = self.create_sentence_from_lines(sent)
                    if not self.ignore_sample(sample):
                        self.raw.append(sample)
                    if self.max_samples and len(self.raw) >= self.max_samples:
                        break
                sent = []
            else:
                sent.append(line.rstrip("\n"))
        if sent:
            if self.max_samples is None or len(self.raw) < self.max_samples:
                sample = self.create_sentence_from_lines(sent)
                if not self.ignore_sample(sample):
                    self.raw.append(sample)

    def create_sentence_from_lines(self, lines):
        sent = []
        labels = []
        token_starts = [0]
        subwords = [self.tokenizer.cls_token]
        for line in lines:
            fd = line.rstrip("\n").split("\t")
            sent.append(fd[0])
            if len(fd) > 1:
                labels.append(fd[1])
            token_starts.append(len(subwords))
            pieces = self.tokenizer.tokenize(fd[0])
            subwords.extend(pieces)
        token_starts.append(len(subwords))
        subwords.append(self.tokenizer.sep_token)
        if len(labels) == 0:
            labels = None
        return self.data_recordclass(
            raw_sentence=sent, labels=labels,
            sentence_len=len(sent),
            subwords=subwords,
            sentence_subword_len=len(subwords),
            token_starts=token_starts,
        )

    def ignore_sample(self, sample):
        return sample.sentence_subword_len > 500

    def to_idx(self):
        mtx = self.data_recordclass.initialize_all(list)
        for sample in self.raw:
            mtx.sentence_len.append(sample.sentence_len)
            mtx.sentence_subword_len.append(sample.sentence_subword_len)
            mtx.token_starts.append(sample.token_starts)
            mtx.subwords.append(self.tokenizer.convert_tokens_to_ids(sample.subwords))
            if sample.labels is None:
                mtx.labels.append(None)
            else:
                mtx.labels.append([self.vocabs.labels[l] for l in sample.labels])
        self.mtx = mtx
        if not self.is_unlabeled:
            if self.config.sort_data_by_length:
                self.sort_data_by_length(sort_field='sentence_subword_len')

    def batched_iter(self, batch_size):
        starts = list(range(0, len(self), batch_size))
        if self.is_unlabeled is False and self.config.shuffle_batches:
            np.random.shuffle(starts)
        for start in starts:
            self._start = start
            end = start + batch_size
            batch = self.data_recordclass()
            maxlen = max(self.mtx.sentence_subword_len[start:end])
            subwords = [
                s + [self.PAD] * (maxlen-len(s))
                for s in self.mtx.subwords[start:end]]
            batch.subwords = subwords
            if self.mtx.labels[0] is not None:
                batch.labels = np.concatenate(self.mtx.labels[start:end])
            else:
                batch.labels = None
            batch.sentence_len = self.mtx.sentence_len[start:end]
            padded_token_starts = []
            # Include [CLS] and [SEP].
            token_maxcount = max(batch.sentence_len) + 2
            for si in range(start, min(len(self.mtx.token_starts), end)):
                starts = self.mtx.token_starts[si]
                pad_count = token_maxcount - len(starts)
                starts.extend([0 for _ in range(pad_count)])
                padded_token_starts.append(starts)
            batch.token_starts = np.array(padded_token_starts)
            batch.sentence_subword_len = self.mtx.sentence_subword_len[start:end]
            yield batch

    def decode(self, model_output):
        offset = 0
        for si, sample in enumerate(self.raw):
            labels = []
            for ti in range(sample.sentence_len):
                label_idx = model_output[offset + ti].argmax()
                labels.append(self.vocabs.labels.inv_lookup(label_idx))
            sample.labels = labels
            offset += sample.sentence_len

    def print_sample(self, sample, stream):
        stream.write("\n".join(
            "{}\t{}".format(sample.raw_sentence[i], sample.labels[i])
            for i in range(sample.sentence_len)
        ))
        stream.write("\n")

    def print_raw(self, stream):
        for si, sample in enumerate(self.raw):
            self.print_sample(sample, stream)
            if si < len(self.raw) - 1:
                stream.write("\n")


class UnlabeledSequenceClassificationWithSubwords(SequenceClassificationWithSubwords):
    @property
    def is_unlabeled(self):
        return True


class SentenceProberDataset(BaseDataset):
    unlabeled_data_class = 'UnlabeledSentenceProberDataset'
    data_recordclass = TokenInSequenceProberFields
    constants = []

    def __init__(self, config, stream_or_file, max_samples=None, **kwargs):
        self.config = config
        self.max_samples = max_samples
        global_key = f'{self.config.model_name}_tokenizer'
        if global_key in globals():
            self.tokenizer = globals()[global_key]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name)
            globals()[global_key] = self.tokenizer
        self.MASK = self.tokenizer.mask_token
        self.mask_positions = set(self.config.mask_positions)
        self.load_or_create_vocabs()
        self.load_stream_or_file(stream_or_file)
        self.to_idx()
        self.tgt_field_idx = -1
        self.max_seqlen = max(s.input_len for s in self.raw)

    def load_or_create_vocabs(self):
        super().load_or_create_vocabs()
        self.vocabs.tokens.PAD = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.pad_token])[0]
        self.vocabs.token_starts.PAD = 1000

    def batched_iter(self, batch_size):
        for batch in super().batched_iter(batch_size):
            batch.token_starts = np.array(batch.token_starts)
            yield batch

    def extract_sample_from_line(self, line):
        raw_sent, raw_target, raw_idx, label = line.rstrip("\n").split("\t")
        raw_idx = int(raw_idx)
        # Build a list-of-lists from the tokenized words.
        # This allows shuffling it later.
        tokenized = [[self.tokenizer.cls_token]]
        for ti, token in enumerate(raw_sent.split(" ")):
            if ti - raw_idx in self.mask_positions:
                pieces = [self.MASK]
            else:
                pieces = self.tokenizer.tokenize(token)
            tokenized.append(pieces)
        # Add [SEP] token start.
        tokenized.append([self.tokenizer.sep_token])
        # Perform BOW.
        if self.config.bow:
            all_idx = np.arange(1, len(tokenized) - 1)
            np.random.shuffle(all_idx)
            all_idx = np.concatenate(([0], all_idx, [len(tokenized)-1]))
            tokenized = [tokenized[i] for i in all_idx]
            target_map = np.argsort(all_idx)
            # Add 1 to include [CLS].
            target_idx = target_map[raw_idx + 1]
        else:
            # Add 1 to include [CLS].
            target_idx = raw_idx + 1
        merged = []
        token_starts = []
        for pieces in tokenized:
            token_starts.append(len(merged))
            merged.extend(pieces)
        return self.data_recordclass(
            raw_sentence=raw_sent,
            raw_target=raw_target,
            raw_idx=raw_idx,
            tokens=merged,
            num_tokens=len(merged),
            target_idx=target_idx,
            token_starts=token_starts,
            label=label,
        )

    def ignore_sample(self, sample):
        if self.config.exclude_short_sentences is False or self.is_unlabeled:
            return False
        sent_len = len(sample.raw_sentence.split(" "))
        for pi in self.mask_positions:
            if sample.raw_idx + pi < 0:
                return True
            if sample.raw_idx + pi >= sent_len:
                return True
        return False

    def to_idx(self):
        mtx = self.data_recordclass.initialize_all(list)
        for sample in self.raw:
            # int fields
            mtx.num_tokens.append(sample.num_tokens)
            mtx.target_idx.append(sample.target_idx)
            mtx.raw_idx.append(sample.raw_idx)
            # int list
            mtx.token_starts.append(sample.token_starts)
            # sentence
            encoded_tokens = self.tokenizer.convert_tokens_to_ids(
                sample.tokens)
            mtx.tokens.append(encoded_tokens)
            # label
            if sample.label is None:
                mtx.label.append(None)
            else:
                mtx.label.append(self.vocabs.label[sample.label])
        self.mtx = mtx
        if not self.is_unlabeled:
            if self.config.sort_data_by_length:
                self.sort_data_by_length(sort_field='input_len')

    @property
    def is_unlabeled(self):
        return False

    def decode(self, model_output):
        for i, sample in enumerate(self.raw):
            output = model_output[i].argmax().item()
            sample.label = self.vocabs.label.inv_lookup(output)

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\t{}\t{}\n".format(
            sample.raw_sentence, sample.raw_target, sample.raw_idx, sample.label
        ))


class UnlabeledSentenceProberDataset(SentenceProberDataset):
    @property
    def is_unlabeled(self):
        return True
