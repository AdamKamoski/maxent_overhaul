import argparse

from maxent_compat import make_env, resolve_env_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    env_ids = ["Pendulum-v1", "Ant-v4", "Humanoid-v4"]

    for env_id in env_ids:
        resolved = resolve_env_id(env_id)
        env = make_env(resolved)
        obs = env.reset()
        print(f"{env_id} -> {resolved}: reset obs shape={getattr(obs, 'shape', None)}")

        for _ in range(args.steps):
            action = env.action_space.sample()
            obs, rew, done, _ = env.step(action)
            print(f"  step reward={rew:.4f} done={done}")
            if done:
                obs = env.reset()

        env.close()


if __name__ == "__main__":
    main()
