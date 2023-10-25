#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 Judit Acs <judit@sch.bme.hu>
#
# Distributed under terms of the MIT license.

import os
import gzip
import logging
import numpy as np
import unidecode

from transformers import AutoTokenizer

from probing.data.base_data import BaseDataset, DataFields


class WLSTMFields(DataFields):
    _fields = (
        'probe_target', 'label', 'probe_target_len', 'target_idx',
        'raw_idx', 'raw_target', 'raw_sentence',)
    _alias = {
        'input': 'probe_target',
        'input_len': 'probe_target_len',
    }
    needs_vocab = ('probe_target', 'label')
    needs_padding = ('probe_target', )


class Word2vecProberFields(DataFields):
    _fields = (
        'sentence', 'probe_target', 'probe_target_idx', 'label')
    _alias = {
        'input': 'probe_target',
    }
    needs_vocab = ('label',)


class TokenInSequenceProberFields(DataFields):
    _fields = (
        'raw_sentence', 'raw_target', 'raw_idx', 'label',
        'subword_tokens', 'input_len', 'probe_target', 'token_starts',
        'probe_target_idx',
    )
    _alias = {
        'input': 'subword_tokens'
    }
    needs_vocab = ('subword_tokens', 'label')
    needs_padding = ('subword_tokens', )
    needs_constants = ('subword_tokens', )


class SLSTMFields(DataFields):
    _fields = (
        'raw_sentence', 'raw_target', 'raw_idx',
        'input', 'input_len', 'target_idx', 'label',
    )
    needs_vocab = ('input', 'label', )
    needs_constants = ('input', )
    needs_padding = ('input', )


class SequenceClassificationWithSubwordsDataFields(DataFields):
    _fields = (
        'raw_sentence', 'labels',
        'sentence_len', 'tokens', 'sentence_subword_len', 'token_starts',
    )
    _alias = {
        'input': 'tokens',
        'input_len': 'sentence_subword_len',
        'label': 'labels',
    }

    needs_vocab = ('tokens', 'labels')
    needs_padding = ('tokens', )
    needs_constants = ('tokens', )


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


class Word2vecProberDataset(BaseDataset):

    datafield_class = Word2vecProberFields

    def to_idx(self):
        vocab = set(r.probe_target for r in self.raw)
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
            word_vecs.append(self.embedding[r.probe_target])
            if r.label:
                labels.append(self.vocabs.label[r.label])
            else:
                labels.append(None)
        self.mtx = self.datafield_class(
            probe_target=word_vecs,
            label=labels
        )

    def extract_sample_from_line(self, line):
        fd = line.rstrip("\n").split("\t")
        sent, target, idx = fd[:3]
        if len(fd) > 3:
            label = fd[3]
        else:
            label = None
        return self.datafield_class(
            sentence=sent,
            probe_target=target,
            probe_target_idx=int(idx),
            label=label
        )

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\t{}\t{}\n".format(
            sample.sentence, sample.probe_target,
            sample.probe_target_idx, sample.label
        ))

    def decode(self, model_output):
        for i, sample in enumerate(self.raw):
            output = model_output[i].argmax().item()
            sample.label = self.vocabs.label.inv_lookup(output)


class WLSTMDataset(BaseDataset):

    datafield_class = WLSTMFields

    def __init__(self, config, stream_or_file, **kwargs):
        if config.external_tokenizer:
            lower = 'uncased' in config.external_tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.external_tokenizer, do_lower_case=lower)
        else:
            self.tokenizer = None
        super().__init__(config, stream_or_file, **kwargs)

    def extract_sample_from_line(self, line):
        fd = line.rstrip("\n").split("\t")
        if len(fd) > 3:
            sent, target, idx, label = fd[:4]
        else:
            sent, target, idx = fd[:3]
            label = None
        idx = int(idx)
        if self.tokenizer:
            tokens = self.tokenizer.tokenize(target)
        else:
            tokens = list(target)
        if self.config.probe_first:
            target_idx = 0
        else:
            target_idx = len(tokens) - 1
        return self.datafield_class(
            raw_sentence=sent,
            probe_target=tokens,
            target_idx=target_idx,
            raw_idx=idx,
            raw_target=target,
            input_len=len(tokens),
            label=label,
        )

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\t{}\t{}\n".format(
            sample.raw_sentence, sample.raw_target, sample.raw_idx, sample.label
        ))

    def decode(self, model_output):
        for i, sample in enumerate(self.raw):
            output = model_output[i].argmax().item()
            sample.label = self.vocabs.label.inv_lookup(output)


class SLSTMDataset(BaseDataset):

    datafield_class = SLSTMFields

    def __init__(self, config, stream_or_file, **kwargs):
        if config.external_tokenizer:
            lower = 'uncased' in config.external_tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.external_tokenizer, do_lower_case=lower)
            self.mask_token = self.tokenizer.mask_token
        else:
            self.tokenizer = None
            self.mask_token = "\u258c"
        super().__init__(config, stream_or_file, **kwargs)

    def extract_sample_from_line(self, line):
        fd = line.rstrip("\n").split("\t")
        raw_sent, raw_target, raw_idx = fd[:3]
        if len(fd) > 3:
            label = fd[3]
        else:
            label = None
        if len(fd) > 4:
            mask_positions = set(int(i) for i in fd[4].split(","))
        else:
            mask_positions = set(self.config.mask_positions)
        raw_idx = int(raw_idx)
        tokenized_words = []
        for word in raw_sent.split(" "):
            if self.tokenizer:
                subwords = self.tokenizer.tokenize(word)
                tokenized_words.append(subwords)
            else:
                tokenized_words.append(list(word))

        invalid_masks = set()
        for position in mask_positions:
            real_position = raw_idx + position
            if real_position >= 0 and real_position < len(tokenized_words):
                orig_len = len(tokenized_words[real_position])
                tokenized_words[real_position] = [self.mask_token] * orig_len
            else:
                invalid_masks.add(position)

        if invalid_masks:
            invalid_masks = list(map(str, invalid_masks))
            logging.debug(f"Invalid mask positions ({','.join(invalid_masks)}) in "
                            f"sentence [{raw_sent}].")
        # Perform BOW.
        if self.config.bow:
            all_idx = np.arange(len(tokenized_words))
            np.random.shuffle(all_idx)
            tokenized_words = [tokenized_words[i] for i in all_idx]
            target_map = np.argsort(all_idx)
            target_idx = target_map[raw_idx]
        else:
            target_idx = raw_idx
        # Add spaces if not using a subword tokenizer
        if not self.tokenizer:
            for toks in tokenized_words[:-1]:
                toks.append(" ")

        # Compute target idx
        if self.config.probe_first:
            target_idx = sum(len(t) for t in tokenized_words[:raw_idx])
        else:
            # The last word does not have an extra space
            if raw_idx == len(tokenized_words) - 1:
                target_idx = sum(len(t) for t in tokenized_words[:target_idx+1]) - 1
            else:
                target_idx = sum(len(t) for t in tokenized_words[:target_idx+1]) - 2

        # Merge tokenized words
        input = []
        for toks in tokenized_words:
            input.extend(toks)

        return self.datafield_class(
            raw_sentence=raw_sent,
            raw_target=raw_target,
            raw_idx=raw_idx,
            input=input,
            input_len=len(input),
            target_idx=target_idx,
            label=label
        )

    def to_idx(self):
        super().to_idx()
        self.mtx.target_idx = np.array(self.mtx.target_idx) + 1
        self.mtx.input_len = np.array(self.mtx.input_len) + 2

    def decode(self, model_output):
        for i, sample in enumerate(self.raw):
            output = np.argmax(model_output[i])
            self.raw[i].label = self.vocabs.label.inv_lookup(output)

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\t{}\t{}\n".format(
            sample.raw_sentence, sample.raw_target, sample.raw_idx, sample.label
        ))


class SequenceClassificationWithSubwords(BaseDataset):

    datafield_class = SequenceClassificationWithSubwordsDataFields

    def __init__(self, config, stream_or_file, max_samples=None,
                 share_vocabs_with=None, is_unlabeled=False):
        global_key = f'{config.model_name}_tokenizer'
        if global_key in globals():
            self.tokenizer = globals()[global_key]
        else:
            lower = 'uncased' in config.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name, do_lower_case=lower)
            globals()[global_key] = self.tokenizer
        super().__init__(config, stream_or_file, max_samples, share_vocabs_with, is_unlabeled)

    def load_or_create_vocabs(self):
        super().load_or_create_vocabs()
        self.vocabs.tokens.vocab = self.tokenizer.get_vocab()
        self.vocabs.tokens.pad_token = self.tokenizer.pad_token
        self.vocabs.tokens.bos_token = self.tokenizer.cls_token
        self.vocabs.tokens.eos_token = self.tokenizer.sep_token
        self.vocabs.tokens.unk_token = self.tokenizer.unk_token
        self.vocabs.tokens.frozen = True

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
        token_starts = []
        subwords = []
        for line in lines:
            fd = line.rstrip("\n").split("\t")
            sent.append(fd[0])
            if len(fd) > 1:
                labels.append(fd[1])
            token_starts.append(len(subwords))
            token = fd[0]
            if self.config.remove_diacritics:
                token = unidecode.unidecode(token)
            pieces = self.tokenizer.tokenize(token)
            subwords.extend(pieces)
        token_starts.append(len(subwords))
        if len(labels) == 0:
            labels = None
        return self.datafield_class(
            raw_sentence=sent, labels=labels,
            sentence_len=len(sent),
            tokens=subwords,
            sentence_subword_len=len(subwords),
            token_starts=token_starts,
        )

    def ignore_sample(self, sample):
        return sample.sentence_subword_len > 500

    def to_idx(self):
        super().to_idx()
        prefixed_token_starts = []
        for ti, tokstarts in enumerate(self.mtx.token_starts):
            tokstarts = [t+1 for t in tokstarts]
            token_starts = [0] + tokstarts + [len(self.mtx.tokens[ti]) + 1]
            prefixed_token_starts.append(token_starts)
        self.mtx.token_starts = prefixed_token_starts

    def batched_iter(self, batch_size):
        for batch in super().batched_iter(batch_size):
            padded_token_starts = []
            maxlen = max(len(t) for t in batch.token_starts)
            pad = 1000
            for sample in batch.token_starts:
                padded = sample + [pad] * (maxlen - len(sample))
                padded_token_starts.append(padded)
            batch.token_starts = np.array(padded_token_starts)
            if batch.labels:
                batch.labels = np.concatenate(batch.labels)
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


class SentenceProberDataset(BaseDataset):

    datafield_class = TokenInSequenceProberFields

    def __init__(self, config, stream_or_file, max_samples=None,
                 share_vocabs_with=None, is_unlabeled=False):
        global_key = f'{config.model_name}_tokenizer'
        if global_key in globals():
            self.tokenizer = globals()[global_key]
        else:
            lower = 'uncased' in config.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.model_name, do_lower_case=lower)
            globals()[global_key] = self.tokenizer
        self.MASK = self.tokenizer.mask_token
        self.mask_positions = set(config.mask_positions)
        if config.use_character_tokenization:
            if not 'bert-' in config.model_name and \
               not 'roberta-' in config.model_name:
                raise ValueError("Character tokenization is only "
                                "supported for BERT and RoBERTa models.")
            logging.info("Using character tokenization.")
        super().__init__(config, stream_or_file, max_samples, share_vocabs_with, is_unlabeled)

    def load_or_create_vocabs(self):
        super().load_or_create_vocabs()
        self.vocabs.subword_tokens.vocab = self.tokenizer.get_vocab()
        self.vocabs.subword_tokens.pad_token = self.tokenizer.pad_token
        self.vocabs.subword_tokens.bos_token = self.tokenizer.cls_token
        self.vocabs.subword_tokens.eos_token = self.tokenizer.sep_token
        self.vocabs.subword_tokens.unk_token = self.tokenizer.unk_token
        self.vocabs.subword_tokens.frozen = True

    def to_idx(self):
        super().to_idx()
        prefixed_token_starts = []
        for ti, tokstarts in enumerate(self.mtx.token_starts):
            tokstarts = [t+1 for t in tokstarts]
            token_starts = [0] + tokstarts + [len(self.mtx.subword_tokens[ti]) - 1]
            prefixed_token_starts.append(token_starts)
        self.mtx.token_starts = prefixed_token_starts
        self.mtx.probe_target_idx = np.array(self.mtx.probe_target_idx) + 1
        self.mtx.input_len = np.array(self.mtx.input_len) + 2

    def batched_iter(self, batch_size):
        for batch in super().batched_iter(batch_size):
            padded_token_starts = []
            maxlen = max(len(t) for t in batch.token_starts)
            pad = 1000
            for sample in batch.token_starts:
                padded = sample + [pad] * (maxlen - len(sample))
                padded_token_starts.append(padded)
            batch.token_starts = np.array(padded_token_starts)
            yield batch

    def extract_sample_from_line(self, line):
        fd = line.rstrip("\n").split("\t")
        raw_sent, raw_target, raw_idx = fd[:3]
        if len(fd) > 3:
            label = fd[3]
        else:
            label = None
        if len(fd) > 4:
            mask_positions = set(int(i) for i in fd[4].split(","))
        else:
            mask_positions = self.mask_positions
        raw_idx = int(raw_idx)
        # Only include the target from the sentence.
        if self.config.target_only:
            if self.config.remove_diacritics:
                target = unidecode.unidecode(raw_target)
            else:
                target = raw_target
            tokenized = [self.tokenizer.tokenize(target)]
            target_idx = 0
        # Build a list-of-lists from the tokenized words.
        # This allows shuffling it later.
        else:
            tokenized = []
            masked = set()
            for ti, token in enumerate(raw_sent.split(" ")):
                if ti - raw_idx in mask_positions:
                    pieces = [self.MASK]
                    masked.add(ti - raw_idx)
                else:
                    if self.config.remove_diacritics:
                        token = unidecode.unidecode(token)
                    if self.config.use_character_tokenization == 'full':
                        pieces = self.character_tokenize_token(token)
                    elif self.config.use_character_tokenization == 'target_only':
                        if ti == raw_idx:
                            pieces = self.character_tokenize_token(token)
                        else:
                            pieces = self.tokenizer.tokenize(token)
                    else:
                        pieces = self.tokenizer.tokenize(token)
                tokenized.append(pieces)
            # Add [SEP] token start.
            # Perform BOW.
            if self.config.bow:
                all_idx = np.arange(len(tokenized))
                np.random.shuffle(all_idx)
                tokenized = [tokenized[i] for i in all_idx]
                target_map = np.argsort(all_idx)
                target_idx = target_map[raw_idx]
            else:
                target_idx = raw_idx

        if masked < mask_positions:
            missing = map(str, mask_positions - masked)
            logging.debug(f"Invalid mask positions ({','.join(missing)}) in "
                            f"sentence [{raw_sent}].")
        merged = []
        token_starts = []
        for pieces in tokenized:
            token_starts.append(len(merged))
            merged.extend(pieces)
        return self.datafield_class(
            raw_sentence=raw_sent,
            raw_target=raw_target,
            raw_idx=raw_idx,
            probe_target_idx=target_idx,
            subword_tokens=merged,
            input_len=len(merged),
            token_starts=token_starts,
            label=label,
        )

    def character_tokenize_token(self, token):
        if 'roberta-' in self.config.model_name:
            start_char = "▁"
            subwords = []
            subwords.extend(self.tokenizer.tokenize(f"{start_char}{token[0]}"))
            subwords.extend(token[1:])
        elif 'bert-' in self.config.model_name:
            subwords = [token[0]]
            subwords.extend(f'##{c}' for c in token[1:])
        return subwords

    def ignore_sample(self, sample):
        return False
        if self.config.exclude_short_sentences is False or self.is_unlabeled:
            return False
        sent_len = len(sample.raw_sentence.split(" "))
        for pi in self.mask_positions:
            if sample.raw_idx + pi < 0:
                return True
            if sample.raw_idx + pi >= sent_len:
                return True
        return False

    def decode(self, model_output):
        for i, sample in enumerate(self.raw):
            output = model_output[i].argmax().item()
            sample.label = self.vocabs.label.inv_lookup(output)

    def print_sample(self, sample, stream):
        stream.write("{}\t{}\t{}\t{}\n".format(
            sample.raw_sentence, sample.raw_target, sample.raw_idx, sample.label
        ))
