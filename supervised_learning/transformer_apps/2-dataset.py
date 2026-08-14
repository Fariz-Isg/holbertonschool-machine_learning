#!/usr/bin/env python3
"""Dataset module"""

import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Loads and tokenizes the Portuguese-English translation dataset."""

    def __init__(self):
        """Class constructor."""
        self.data_train = load_pt2en(split="train")
        self.data_valid = load_pt2en(split="validation")

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Creates subword tokenizers for Portuguese and English.
        """
        def pt_iterator():
            batch = []
            for pt, _ in data:
                batch.append(pt.numpy().decode('utf-8'))
                if len(batch) >= 1000:
                    yield batch
                    batch = []
            if batch:
                yield batch

        def en_iterator():
            batch = []
            for _, en in data:
                batch.append(en.numpy().decode('utf-8'))
                if len(batch) >= 1000:
                    yield batch
                    batch = []
            if batch:
                yield batch

        # Create tokenizers for Portuguese and English
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True,
            clean_up_tokenization_spaces=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True,
            clean_up_tokenization_spaces=True)

        # Train the tokenizers
        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iterator(), vocab_size=2 ** 13)
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iterator(), vocab_size=2 ** 13)

        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en

        return self.tokenizer_pt, self.tokenizer_en

    def encode(self, pt, en):
        """Encode a translation pair into token lists with SOS/EOS.

        Args:
            pt: tf.Tensor containing the Portuguese sentence.
            en: tf.Tensor containing the English sentence.

        Returns:
            Tuple of (pt_tokens, en_tokens) where each is a list of
            ints. The start token is vocab_size and end token is
            vocab_size + 1.
        """
        pt_vocab_size = self.tokenizer_pt.vocab_size
        en_vocab_size = self.tokenizer_en.vocab_size

        pt_tokens = self.tokenizer_pt.encode(
            pt.numpy().decode('utf-8'), add_special_tokens=False
        )
        en_tokens = self.tokenizer_en.encode(
            en.numpy().decode('utf-8'), add_special_tokens=False
        )

        pt_tokens = [pt_vocab_size] + pt_tokens + [pt_vocab_size + 1]
        en_tokens = [en_vocab_size] + en_tokens + [en_vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """TensorFlow wrapper for the encode method.

        Args:
            pt: tf.Tensor containing the Portuguese sentence.
            en: tf.Tensor containing the English sentence.

        Returns:
            Tuple of (pt_tensor, en_tensor), both tf.int64 tensors
            with shapes set.
        """
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens
