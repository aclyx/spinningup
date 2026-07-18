import gymnasium as gym
import numpy as np

from spinup import ppo_pytorch
from spinup.utils.gym_compat import adapt_env
from spinup.utils.test_policy import load_pytorch_policy


def test_gymnasium_adapter_exposes_legacy_api():
    env = adapt_env(gym.make("CartPole-v1", max_episode_steps=1), seed=7)

    observation = env.reset()
    transition = env.step(env.action_space.sample())

    assert observation.shape == (4,)
    assert len(transition) == 4
    assert transition[2] is True
    assert transition[3]["TimeLimit.truncated"] is True
    env.close()


def test_pytorch_ppo_trains_saves_and_loads(tmp_path):
    ppo_pytorch(
        lambda: gym.make("CartPole-v1"),
        steps_per_epoch=64,
        epochs=1,
        train_pi_iters=1,
        train_v_iters=1,
        ac_kwargs={"hidden_sizes": (16,)},
        logger_kwargs={"output_dir": str(tmp_path)},
    )

    get_action = load_pytorch_policy(str(tmp_path), "")
    env = adapt_env(gym.make("CartPole-v1"), seed=11)
    action = get_action(env.reset())

    assert np.asarray(action).shape == ()
    env.close()
