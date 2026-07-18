"""TensorFlow 1-style graph mode on supported TensorFlow 2 releases."""

import os


# TensorFlow 2.16+ ships with Keras 3 by default. Spinning Up's TF1 examples
# use the legacy ``tf.layers`` API, which is provided by the ``tf-keras``
# package when this switch is set before importing TensorFlow.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow.compat.v1 as tf  # noqa: E402


tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()
