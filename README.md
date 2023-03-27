# Probing

Probing is a popular evaluation method for blackbox language models.
In the simplest case, the representation of a token or a sentence is fed to a small classifier that tries to predict some linguistic label.
This setup is exposed to a small amount of training data but only the classifier parameters are trained, the blackbox model's parameters are kept fixed.

This library was developed for probing contextualized language models such as BERT.

# Our papers that use this library

This framework was used in the following projects:

## Subword Pooling Makes a Difference

[EACL2021 paper](https://arxiv.org/abs/2102.10864)

[Github repository](https://github.com/juditacs/subword-choice)

## Evaluating Contextualized Language Models for Hungarian

[Paper](https://arxiv.org/abs/2102.10848)

[Github repository](https://github.com/juditacs/hubert_eval)


## Evaluating Transferability of BERT Models on Uralic Languages

[Paper](https://arxiv.org/abs/2109.06327)

[Github repository](https://github.com/juditacs/uralic_eval)

# Probing types and tasks

It supports two types of evaluation.

## Morphology probing

Probe a single word in its sentence context, such as deriving the tense of the English word cut in these examples, where context clearly plays an important role:

    I cut my hair yesterday.
    Make sure you cut the edges.

### Data format

Morphology probing uses TSV files as input.
One line represents one sample.
Each line has 4 tab-separated columns:

1. the full sentence
2. the target word
3. the index of the target word when using space tokenization
4. the label.

The above two examples would look like this:

```
I cut my hair yesterday.        cut        1        Past        
Make sure you cut the edges.    cut        3        Pres
```

In **tagging tasks** each line represents a word and sentence boundaries are denoted by empty lines.
Each line consists of a word and its tag or label separated by a tab.

An example for English POS tagging:

```
If      SCONJ
you     PRON
need    VERB
any     DET
more    ADJ
info    NOUN
,       PUNCT
please  INTJ
advise  VERB
.       PUNCT

Hi      INTJ
all     DET
.       PUNCT
```

See `examples/data` for more.

## Tagging

Tagging assigns a label to each word in the sentence.
Common examples are part of speech tagging or named entity recognition.

# Probing setup

We support most BERT-like language models available in [Huggingface's repository](https://huggingface.co/models).
We add a small multilayer perceptron on top of the representation and train its weights.
The language models are not finetuned and we cache the output of the language models when possible.
This allows running a very large number of experiments on a single GPU.
For tagging tasks, the parameters of the MLP are shared across all tokens.

# Installation

Requirements:
- Python >= 3.6
- PyTorch >= 1.7 (we recommend installing PyTorch before installing this package)

Install command:

    cd <PATH_TO_PROBING_REPO>
    pip install .

# Usage

## Configuration

Experiment configuration is managed through YAML config files.
We provided some examples in the `examples/config` directory along with example toy datasets in the `examples/data` directory.
The train and dev files can be provided as command line arguments as well.

## Training a single experiment

Morphology probing:

    python probing/train.py \
        --config examples/config/transformer_probing.yaml \
        --train-file examples/data/morphology_probing/english_verb_tense/train.tsv \
        --dev-file examples/data/morphology_probing/english_verb_tense/dev.tsv

POS tagging:

    python probing/train.py \
        --config examples/config/pos_tagging.yaml \
        --train-file examples/data/pos_tagging/english/train \
        --dev-file examples/data/pos_tagging/english/dev


## Training multiple experiments

`train_many_configs.py` takes a Python source file as its parameter which must contain a function named `generate_configs` returns or yields `Config` objects.
This makes it possible to run an arbitrary number of experiments with varying configuration.
This toy example trains two models for English POS tagging, one that uses the first subword token of each token and one that used the last subword token.
We explore these options in detail in [this paper](https://arxiv.org/abs/2102.10864).

    python probing/train_many_configs.py \
        --config examples/config/pos_tagging.yaml \
        --param-generator examples/config/generate_pos_configs.py


## Inference

Inference on one experiment:

    python probing/inference.py \
        --experiment-dir PATH_TO_AN_EXPDIR \
        --test-file PATH_TO_A_TEST_FILE \
        > output

If no test file is provided, it reads from the standard input.

Inference on multiple experiments:

    python probing/batch_inference.py \
        EXPERIMENT_DIRS \
        --run-on-dev \
        --run-on-test

`EXPERIMENT_DIRS` is an arbitrary number of positional arguments accepted by `batch_inference.py`.
It may by a glob such as `~/workdir/exps/*`.
If `--run-on-dev` or `--run-on-test` is not provided, the dev or the test set will not be evaluated.
`batch_inference.py` only evaluated directories where there is no `test.out` or it is older than the last model checkpoint.
