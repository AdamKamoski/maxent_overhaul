import os
import sys
from typing import Any

ENV_ALIASES = {
    "Ant-v2": "Ant-v4",
    "Humanoid-v2": "Humanoid-v4",
    "HalfCheetah-v2": "HalfCheetah-v4",
    "Walker2d-v2": "Walker2d-v4",
    "Swimmer-v2": "Swimmer-v4",
    "Pendulum-v0": "Pendulum-v1",
}


def add_project_paths(file_path: str) -> None:
    root = os.path.abspath(os.path.dirname(file_path))
    parent = os.path.abspath(os.path.join(root, ".."))
    for candidate in (root, parent):
        if candidate not in sys.path:
            sys.path.append(candidate)


def resolve_env_id(env_id: str) -> str:
    return ENV_ALIASES.get(env_id, env_id)


class LegacyEnvAdapter:
    """Wraps Gym/Gymnasium envs to expose old Gym<=0.21 reset/step/seed API."""

    def __init__(self, env: Any):
        self._env = env

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    @property
    def env(self) -> Any:
        return getattr(self._env, "env", self._env)

    @property
    def unwrapped(self) -> Any:
        return self._env.unwrapped

    @property
    def action_space(self) -> Any:
        return self._env.action_space

    @property
    def observation_space(self) -> Any:
        return self._env.observation_space

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        out = self._env.reset(*args, **kwargs)
        if isinstance(out, tuple):
            return out[0]
        return out

    def step(self, action: Any):
        out = self._env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return obs, reward, (terminated or truncated), info
        return out

    def seed(self, seed: int) -> None:
        try:
            self._env.reset(seed=seed)
        except TypeError:
            pass
        try:
            self._env.action_space.seed(seed)
        except Exception:
            pass
        try:
            self._env.seed(seed)
        except Exception:
            pass

    def close(self) -> None:
        self._env.close()


def make_env(env_id: str) -> LegacyEnvAdapter:
    resolved = resolve_env_id(env_id)
    try:
        import gym

        env = gym.make(resolved)
        return LegacyEnvAdapter(env)
    except Exception:
        import gymnasium as gym

        env = gym.make(resolved)
        return LegacyEnvAdapter(env)


def monitor_env(env: Any, video_dir: str) -> LegacyEnvAdapter:
    os.makedirs(video_dir, exist_ok=True)
    base = env._env if isinstance(env, LegacyEnvAdapter) else env

    try:
        import gym

        if hasattr(gym.wrappers, "RecordVideo"):
            wrapped = gym.wrappers.RecordVideo(
                base,
                video_folder=video_dir,
                episode_trigger=lambda _: True,
                disable_logger=True,
            )
            return LegacyEnvAdapter(wrapped)
    except Exception:
        pass

    return env if isinstance(env, LegacyEnvAdapter) else LegacyEnvAdapter(base)


def raw_obs_from_state(env: Any) -> Any:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    if hasattr(base, "_get_obs"):
        return base._get_obs()
    return env.reset()
