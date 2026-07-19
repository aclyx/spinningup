"""Small compatibility boundary for Gymnasium's modern environment API."""

from __future__ import annotations

import warnings
from typing import Any

import gymnasium


class LegacyEnvAdapter:
    """Expose Gymnasium environments through the four-value API used here.

    The algorithms in Spinning Up predate Gymnasium's ``terminated`` and
    ``truncated`` split. Keeping the conversion in one adapter avoids hiding
    compatibility details throughout the educational algorithm code.
    """

    def __init__(self, env: Any, seed: int | None = None):
        self.env = env
        self._env_id = getattr(getattr(env, "spec", None), "id", None)
        self._next_seed = seed

        if seed is not None:
            for space_name in ("action_space", "observation_space"):
                space = getattr(env, space_name, None)
                if space is not None and hasattr(space, "seed"):
                    space.seed(seed)

    def __getattr__(self, name: str) -> Any:
        try:
            env = object.__getattribute__(self, "env")
        except AttributeError as error:
            raise AttributeError(name) from error
        return getattr(env, name)

    def reset(self, **kwargs: Any) -> Any:
        if self._next_seed is not None and "seed" not in kwargs:
            kwargs["seed"] = self._next_seed
            self._next_seed = None

        result = self.env.reset(**kwargs)
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], dict)
        ):
            return result[0]
        return result

    def step(self, action: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        result = self.env.step(action)
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            info = dict(info)
            info.setdefault("terminated", bool(terminated))
            info.setdefault("truncated", bool(truncated))
            info.setdefault("TimeLimit.truncated", bool(truncated))
            done = bool(terminated or truncated)
            return observation, reward, done, info
        return result


def adapt_env(env: Any, seed: int | None = None) -> LegacyEnvAdapter:
    """Return an idempotently wrapped environment."""

    if isinstance(env, LegacyEnvAdapter):
        if seed is not None:
            env._next_seed = seed
        return env
    return LegacyEnvAdapter(env, seed=seed)


def _infer_env_id(env: LegacyEnvAdapter) -> str | None:
    env_id = env.__dict__.get("_env_id")
    if env_id is not None:
        return env_id

    spec = getattr(env.env, "spec", None)
    if spec is not None and spec.id is not None:
        return spec.id

    base_env = env.unwrapped
    target = f"{type(base_env).__module__}:{type(base_env).__qualname__}"
    candidates = []
    for candidate in gymnasium.envs.registry.values():
        entry_point = candidate.entry_point
        if not (
            entry_point == target
            or (callable(entry_point) and entry_point is type(base_env))
        ):
            continue
        if any(
            hasattr(base_env, name) and getattr(base_env, name) != value
            for name, value in candidate.kwargs.items()
        ):
            continue
        candidates.append(candidate)

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            candidate.version or -1,
            len(candidate.kwargs),
        ),
    ).id


def with_render_mode(env: Any, render_mode: str = "human") -> LegacyEnvAdapter:
    """Recreate a registered environment with a Gymnasium render mode."""

    env = adapt_env(env)
    if getattr(env, "render_mode", None) == render_mode:
        return env

    env_id = _infer_env_id(env)
    if env_id is None:
        warnings.warn(
            "Could not identify the saved environment, so rendering may be "
            "unavailable.",
            stacklevel=2,
        )
        return env

    make_kwargs = {"render_mode": render_mode}
    max_episode_steps = getattr(env.env, "_max_episode_steps", None)
    if max_episode_steps is not None:
        make_kwargs["max_episode_steps"] = max_episode_steps

    rendered_env = gymnasium.make(env_id, **make_kwargs)
    env.close()
    return adapt_env(rendered_env)


def make_env(env_id: str, **kwargs: Any) -> LegacyEnvAdapter:
    """Create a Gymnasium environment with Spinning Up's expected API."""

    return adapt_env(gymnasium.make(env_id, **kwargs))
