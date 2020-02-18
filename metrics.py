"""Quality metrics for the model."""

import tensorflow as tf


def char_accuracy(predictions, targets, rej_char, streaming=False):
    """Computes character level accuracy.
    Both predictions and targets should have the same shape
    [batch_size x seq_length].
    Args:
        predictions: predicted characters ids.
        targets: ground truth character ids.
        rej_char: the character id used to mark an empty element (end of sequence).
        streaming: if True, uses the streaming mean from the slim.metric module.
    Returns:
        a update_ops for execution and value tensor whose value on evaluation
        returns the total character accuracy.
    """
    with tf.variable_scope('CharAccuracy'):
        predictions.get_shape().assert_is_compatible_with(targets.get_shape())

        targets = tf.to_int32(targets)
        const_rej_char = tf.constant(rej_char, shape=targets.get_shape())
        weights = tf.to_float(tf.not_equal(targets, const_rej_char))
        correct_chars = tf.to_float(tf.equal(predictions, targets))
        accuracy_per_example = tf.div(
            tf.reduce_sum(tf.multiply(correct_chars, weights), 1),
            tf.reduce_sum(weights, 1))
        if streaming:
            return tf.contrib.metrics.streaming_mean(accuracy_per_example)
        else:
            return tf.reduce_mean(accuracy_per_example)

def sequence_accuracy(predictions, targets, rej_char, streaming=False):
    """Computes sequence level accuracy.
    Both input tensors should have the same shape: [batch_size x seq_length].
    Args:
        predictions: predicted character classes.
        targets: ground truth character classes.
        rej_char: the character id used to mark empty element (end of sequence).
        streaming: if True, uses the streaming mean from the slim.metric module.
    Returns:
        a update_ops for execution and value tensor whose value on evaluation
        returns the total sequence accuracy.
    """

    with tf.variable_scope('SequenceAccuracy'):
        predictions.get_shape().assert_is_compatible_with(targets.get_shape())

        targets = tf.to_int32(targets)
        const_rej_char = tf.constant(
            rej_char, shape=targets.get_shape(), dtype=tf.int32)
        include_mask = tf.not_equal(targets, const_rej_char)
        include_predictions = tf.to_int32(
            tf.where(include_mask, predictions,
                    tf.zeros_like(predictions) + rej_char))
        correct_chars = tf.to_float(tf.equal(include_predictions, targets))
        correct_chars_counts = tf.cast(
            tf.reduce_sum(correct_chars, reduction_indices=[1]), dtype=tf.int32)
        target_length = targets.get_shape().dims[1].value
        target_chars_counts = tf.constant(
            target_length, shape=correct_chars_counts.get_shape())
        accuracy_per_example = tf.to_float(
            tf.equal(correct_chars_counts, target_chars_counts))
        if streaming:
            return tf.contrib.metrics.streaming_mean(accuracy_per_example)
        else:
            return tf.reduce_mean(accuracy_per_example)
