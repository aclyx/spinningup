============
Installation
============

Spinning Up supports Python 3.10 through 3.12. It uses Gymnasium for
environments, current PyTorch releases, and TensorFlow 2 in TF1-compatible
graph mode.

Installing Spinning Up with uv
==============================

From the repository root, create the virtual environment and install the
project with `uv`_:

.. parsed-literal::

    uv sync

Run Python commands and project tools through the managed environment:

.. parsed-literal::

    uv run python -m spinup.run help
    uv run pytest

.. _`uv`: https://docs.astral.sh/uv/

Optional environment families
=============================

The default installation includes Gymnasium's Classic Control environments.
Install other environment families only when you need them:

.. parsed-literal::

    uv sync --extra box2d
    uv sync --extra mujoco
    uv sync --extra atari

Multiple ``--extra`` flags can be supplied to install more than one family.

Installing OpenMPI (Optional)
=============================

OpenMPI is needed only for multi-process runs such as ``--cpu 4``.

Ubuntu
------

.. parsed-literal::

    sudo apt-get update
    sudo apt-get install libopenmpi-dev openmpi-bin

macOS
-----

.. parsed-literal::

    brew install open-mpi

Check Your Install
==================

Run the test suite:

.. parsed-literal::

    uv run pytest

For a short end-to-end training smoke test, run PPO on CartPole:

.. parsed-literal::

    uv run python -m spinup.run ppo_pytorch --env CartPole-v1 \
        --hid "[32,32]" --epochs 1 --steps_per_epoch 100 \
        --exp_name installtest

Experiment output is written under ``data/`` by default. Plot a completed run
with:

.. parsed-literal::

    uv run python -m spinup.run plot data/installtest/installtest_s0

MuJoCo (Optional)
=================

After installing the MuJoCo extra, verify it with a current Gymnasium MuJoCo
environment:

.. parsed-literal::

    uv sync --extra mujoco
    uv run python -m spinup.run ppo_pytorch --env Walker2d-v5 \
        --hid "[32,32]" --exp_name mujocotest
