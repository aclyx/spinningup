# Repository Agent Guidance

## Python runtime and dependency management

- This repository supports Python 3.10 through 3.12.
- Treat `pyproject.toml` and `uv.lock` as the canonical package metadata. Do not
  restore the removed `setup.py` dependency pins.
- Prefer `uv` for Python environments and dependency installation in this
  repository. Create and synchronize the local environment with `uv sync`, and
  use `uv run` to execute Python commands and project tools.
- After changing dependencies, update and verify the lockfile with `uv lock`
  and `uv lock --check`.
- Avoid invoking `pip` directly unless the user explicitly requests it or `uv`
  cannot support the required operation.
- Optional environment families are installed with extras: `uv sync --extra
  box2d`, `uv sync --extra atari`, or `uv sync --extra mujoco`. MuJoCo uses the
  maintained `mujoco` Python package through Gymnasium; it does not require
  `mujoco-py` or a license key.

## Modern compatibility boundaries

- Use Gymnasium and current environment IDs in new code, such as
  `CartPole-v1` and `Walker2d-v5`.
- Spinning Up's teaching algorithms intentionally retain their legacy internal
  environment API. Pass newly created Gymnasium environments through
  `spinup.utils.gym_compat.adapt_env` instead of duplicating reset, step, or
  seed conversion logic.
- Preserve the distinction between termination and truncation. A true terminal
  state must not bootstrap its value estimate; a time-limit truncation should.
- TF1 examples and algorithms run on modern TensorFlow through
  `spinup.utils.tf_compat`. Import `tf` from that module rather than importing
  TensorFlow directly in TF1 code.
- Saved PyTorch policies are trusted local artifacts and are loaded explicitly
  with `weights_only=False`; do not rely on changing framework defaults.

## Validation

- Run the default regression suite with `uv run pytest -q`.
- Validate package metadata with `uv lock --check` and `uv build`.
- For a short classic-control integration check, run:
  `uv run python -m spinup.run ppo_pytorch --env CartPole-v1 --hid "[32,32]"
  --epochs 1 --steps_per_epoch 100 --exp_name installtest`.
- After `uv sync --extra mujoco`, use `Walker2d-v5` for MuJoCo smoke tests.
- Legacy TF1 tests currently emit deprecation warnings for `tf.layers.dense`;
  warnings alone are not test failures.
