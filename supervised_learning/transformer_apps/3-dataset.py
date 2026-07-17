#!/usr/bin/env python3
"""Dataset class with full data pipeline for machine translation."""
import tensorflow as tf
import transformers
from setup import load_pt2en


class Dataset:
    """Load and prepare the Portuguese-to-English translation dataset."""

    def __init__(self, batch_size, max_len):
        """Initialize the dataset with batching and filtering.

        Args:
            batch_size (int): Batch size for training/validation.
            max_len (int): Maximum number of tokens per sentence.

        Creates instance attributes:
            data_train: Training tf.data.Dataset, filtered, cached,
                shuffled, padded-batched, and prefetched.
            data_valid: Validation tf.data.Dataset, filtered and
                padded-batched.
            tokenizer_pt: Portuguese sub-word tokenizer.
            tokenizer_en: English sub-word tokenizer.
        """
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def filter_max_len(pt, en):
            """Filter out examples exceeding max_len tokens."""
            return tf.logical_and(
                tf.size(pt) <= max_len,
                tf.size(en) <= max_len
            )

        self.data_train = self.data_train.filter(filter_max_len)
        self.data_train = self.data_train.cache()
        self.data_train = self.data_train.shuffle(20000)
        self.data_train = self.data_train.padded_batch(batch_size)
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE
        )

        self.data_valid = self.data_valid.filter(filter_max_len)
        self.data_valid = self.data_valid.padded_batch(batch_size)

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
            pt.numpy().decode('utf-8')
        )
        en_tokens = self.tokenizer_en.encode(
            en.numpy().decode('utf-8')
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
