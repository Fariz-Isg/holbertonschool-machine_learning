#!/usr/bin/env python3
"""Transformer training loop for Portuguese-to-English translation."""
import tensorflow as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Warm-up + inverse-sqrt-decay learning rate schedule."""

    def __init__(self, dm, warmup_steps=4000):
        """Initialize the schedule.

        Args:
            dm (int): Model dimensionality.
            warmup_steps (int): Number of warm-up steps.
        """
        super().__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """Compute learning rate for the given step.

        Args:
            step: Current training step (scalar tensor).

        Returns:
            Learning rate scalar.
        """
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Create and train a transformer for PT-to-EN translation.

    Args:
        N (int): Number of encoder/decoder blocks.
        dm (int): Model dimensionality.
        h (int): Number of attention heads.
        hidden (int): Feed-forward hidden units.
        max_len (int): Maximum tokens per sequence.
        batch_size (int): Batch size.
        epochs (int): Number of training epochs.

    Returns:
        The trained Transformer model.
    """
    data = Dataset(batch_size, max_len)

    input_vocab = data.tokenizer_pt.vocab_size + 2
    target_vocab = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(
        N, dm, h, hidden,
        input_vocab, target_vocab,
        max_len, max_len,
        drop_rate=0.1
    )

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9
    )

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none'
    )

    def loss_function(real, pred):
        """Compute masked sparse categorical crossentropy.

        Args:
            real: Ground-truth token IDs.
            pred: Model logits.

        Returns:
            Scalar mean loss (ignoring padding).
        """
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def accuracy_function(real, pred):
        """Compute masked accuracy.

        Args:
            real: Ground-truth token IDs.
            pred: Model logits.

        Returns:
            Scalar accuracy (ignoring padding).
        """
        accuracies = tf.equal(real, tf.argmax(pred, axis=2))
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        accuracies = tf.math.logical_and(mask, accuracies)
        accuracies = tf.cast(accuracies, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for batch_idx, (inp, tar) in enumerate(data.data_train):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_mask, combined_mask, dec_mask = create_masks(inp, tar_inp)

            with tf.GradientTape() as tape:
                predictions = transformer(
                    inp, tar_inp, True,
                    enc_mask, combined_mask, dec_mask
                )
                loss = loss_function(tar_real, predictions)

            gradients = tape.gradient(
                loss, transformer.trainable_variables
            )
            optimizer.apply_gradients(
                zip(gradients, transformer.trainable_variables)
            )

            train_loss(loss)
            train_accuracy(accuracy_function(tar_real, predictions))

            if batch_idx % 50 == 0:
                print(
                    'Epoch {}, Batch {}: Loss {}, Accuracy {}'.format(
                        epoch + 1, batch_idx,
                        train_loss.result(),
                        train_accuracy.result()
                    )
                )

        print('Epoch {}: Loss {}, Accuracy {}'.format(
            epoch + 1,
            train_loss.result(),
            train_accuracy.result()
        ))

    return transformer
