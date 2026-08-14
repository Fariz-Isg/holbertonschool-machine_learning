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