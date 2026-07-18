"""Small compatibility boundary for Gymnasium's modern environment API."""

from __future__ import annotations

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
        self._next_seed = seed

        if seed is not None:
            for space_name in ("action_space", "observation_space"):
                space = getattr(env, space_name, None)
                if space is not None and hasattr(space, "seed"):
                    space.seed(seed)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

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


def make_env(env_id: str, **kwargs: Any) -> LegacyEnvAdapter:
    """Create a Gymnasium environment with Spinning Up's expected API."""

    return adapt_env(gymnasium.make(env_id, **kwargs))
