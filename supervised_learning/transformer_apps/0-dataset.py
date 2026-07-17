#!/usr/bin/env python3
"""Dataset class for machine translation tokenizer setup."""
import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Load and prepare the Portuguese-to-English translation dataset."""

    def __init__(self):
        """Initialize the dataset with train/validation splits and
        tokenizers.

        Creates instance attributes:
            data_train: Training tf.data.Dataset of (pt, en) string pairs.
            data_valid: Validation tf.data.Dataset of (pt, en) string pairs.
            tokenizer_pt: Portuguese sub-word tokenizer.
            tokenizer_en: English sub-word tokenizer.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Create sub-word tokenizers trained on the dataset.

        Args:
            data: tf.data.Dataset of (pt, en) tf.string pairs.

        Returns:
            Tuple of (tokenizer_pt, tokenizer_en), both
            BertTokenizerFast instances with vocab size 2**13.
        """
        def pt_iterator():
            """Yield Portuguese sentences from the dataset."""
            for pt, en in data:
                yield pt.numpy().decode('utf-8')

        def en_iterator():
            """Yield English sentences from the dataset."""
            for pt, en in data:
                yield en.numpy().decode('utf-8')

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_iterator(), vocab_size=2 ** 13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_iterator(), vocab_size=2 ** 13
        )

        return tokenizer_pt, tokenizer_en
