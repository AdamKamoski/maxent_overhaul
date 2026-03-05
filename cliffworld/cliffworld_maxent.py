import argparse
import os
from collections import deque

import numpy as np
import scipy.stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class CliffWorld:
    def __init__(self, height=4, width=12):
        self.height = height
        self.width = width
        self.n_states = height * width
        self.start = (height - 1, 0)
        self.goal = (height - 1, width - 1)
        self.cliff = {(height - 1, c) for c in range(1, width - 1)}

    def to_idx(self, rc):
        return rc[0] * self.width + rc[1]

    def from_idx(self, idx):
        return divmod(idx, self.width)

    def transition(self, state, action):
        r, c = self.from_idx(state)
        if action == 0:
            nr, nc = max(0, r - 1), c
        elif action == 1:
            nr, nc = min(self.height - 1, r + 1), c
        elif action == 2:
            nr, nc = r, max(0, c - 1)
        else:
            nr, nc = r, min(self.width - 1, c + 1)

        nxt = (nr, nc)
        if nxt in self.cliff:
            return self.to_idx(self.start)
        if nxt == self.goal:
            return self.to_idx(self.start)
        return self.to_idx(nxt)


def softmax(x, temp=1.0):
    z = (x - np.max(x)) / max(temp, 1e-8)
    e = np.exp(z)
    return e / np.sum(e)


def reachable_states(env):
    start_idx = env.to_idx(env.start)
    q = deque([start_idx])
    seen = {start_idx}
    while q:
        s = q.popleft()
        for a in range(4):
            ns = env.transition(s, a)
            if ns not in seen:
                seen.add(ns)
                q.append(ns)
    return sorted(seen)


def solve_oracle_policy(env, reward_state, gamma=0.99, vi_iters=200, temperature=0.15):
    n_s, n_a = env.n_states, 4
    v = np.zeros(n_s, dtype=np.float64)
    next_s = np.zeros((n_s, n_a), dtype=np.int32)
    for s in range(n_s):
        for a in range(n_a):
            next_s[s, a] = env.transition(s, a)

    for _ in range(vi_iters):
        q = reward_state[next_s] + gamma * v[next_s]
        new_v = np.max(q, axis=1)
        if np.max(np.abs(new_v - v)) < 1e-8:
            v = new_v
            break
        v = new_v

    q = reward_state[next_s] + gamma * v[next_s]
    pi = np.zeros((n_s, n_a), dtype=np.float64)
    for s in range(n_s):
        pi[s] = softmax(q[s], temp=temperature)
    return pi


def sample_occupancy(env, policy, steps=2000, n_rollouts=8, seed=0):
    rng = np.random.RandomState(seed)
    occ = np.zeros(env.n_states, dtype=np.float64)
    start_idx = env.to_idx(env.start)

    for _ in range(n_rollouts):
        s = start_idx
        for _ in range(steps):
            occ[s] += 1.0
            a = rng.choice(4, p=policy[s])
            s = env.transition(s, a)

    total = occ.sum()
    if total > 0:
        occ /= total
    return occ


def mixture_policy(policies, weights):
    n_s, n_a = policies[0].shape
    pi = np.zeros((n_s, n_a), dtype=np.float64)
    for w, p in zip(weights, policies):
        pi += w * p
    pi /= np.clip(pi.sum(axis=1, keepdims=True), 1e-12, None)
    return pi


def occupancy_gradient(occ, eps=1e-6, mode="inverse"):
    if mode == "inverse":
        return 1.0 / (occ + eps)
    if mode == "log":
        return -np.log(occ + eps)
    raise ValueError("gradient mode must be one of: inverse, log")


def hazard_mask(env):
    m = np.zeros((env.height, env.width), dtype=np.float64)
    for rc in env.cliff:
        m[rc] = 1.0
    m[env.start] = 0.5
    m[env.goal] = 0.75
    return m


def save_entropy_plot(out_dir, iterations, ent_hazan, ent_uniform, ent_best_theoretical):
    plt.figure(figsize=(8.2, 4.8))
    plt.plot(iterations, ent_hazan, label="Hazan (empirical)", linewidth=1.8)
    plt.plot(iterations, ent_uniform, label="Uniform policy (empirical)", linewidth=1.6)
    plt.plot(iterations, ent_best_theoretical, label="Best theoretical policy", linestyle="--", linewidth=1.6)
    plt.xlabel("Iteration")
    plt.ylabel("Empirical occupancy entropy (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "entropy_vs_iteration.png"), dpi=150)
    plt.close()


def save_heatmap(out_dir, env, occ, name):
    mat = occ.reshape(env.height, env.width)
    hz = hazard_mask(env)

    plt.figure(figsize=(8, 3))
    plt.imshow(np.log(mat + 1e-9), cmap="viridis", interpolation="nearest")
    ys, xs = np.where(hz > 0)
    plt.scatter(xs, ys, c="red", s=25, marker="s", label="cliff/start/goal")
    plt.xticks(range(env.width))
    plt.yticks(range(env.height))
    plt.title(name)
    plt.colorbar(label="log occupancy")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150)
    plt.close()


def run(args):
    env = CliffWorld(height=args.height, width=args.width)
    out_dir = os.path.join("figs", "cliffworld", args.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    reachable = reachable_states(env)
    theoretical_max_entropy_raw = float(np.log(len(reachable)))
    norm_factor = args.target_max_entropy / theoretical_max_entropy_raw

    uniform_action_pi = np.ones((env.n_states, 4), dtype=np.float64) / 4.0
    uniform_action_occ = sample_occupancy(
        env,
        uniform_action_pi,
        steps=args.steps,
        n_rollouts=args.n_rollouts,
        seed=args.seed + 7,
    )
    uniform_action_entropy_raw = float(scipy.stats.entropy(uniform_action_occ))

    occ_mix = uniform_action_occ.copy()
    policies = [uniform_action_pi.copy()]
    weights = [1.0]

    ent_hazan = []
    ent_uniform = []
    ent_best_theoretical = []
    iterations = []

    first_occ = occ_mix.copy()

    for t in range(args.iterations):
        grad_r = occupancy_gradient(occ_mix, eps=args.eps, mode=args.grad_mode)
        oracle_pi = solve_oracle_policy(
            env,
            reward_state=grad_r,
            gamma=args.gamma,
            vi_iters=args.vi_iters,
            temperature=args.temperature,
        )
        occ_oracle = sample_occupancy(
            env,
            oracle_pi,
            steps=args.steps,
            n_rollouts=args.n_rollouts,
            seed=args.seed + 31 + t,
        )

        eta = 2.0 / (t + 2.0) if args.step_schedule == "hazan" else args.eta
        occ_mix = (1.0 - eta) * occ_mix + eta * occ_oracle

        policies.append(oracle_pi)
        weights = [(1.0 - eta) * w for w in weights] + [eta]
        weights = np.array(weights, dtype=np.float64)
        weights /= weights.sum()
        mix_pi = mixture_policy(policies, weights)

        occ_eval = sample_occupancy(
            env,
            mix_pi,
            steps=args.steps,
            n_rollouts=args.n_rollouts,
            seed=args.seed + 100 + t,
        )

        it = t + 1
        hazan_entropy_raw = float(scipy.stats.entropy(occ_eval))

        iterations.append(it)
        ent_hazan.append(hazan_entropy_raw * norm_factor)
        ent_uniform.append(uniform_action_entropy_raw * norm_factor)
        ent_best_theoretical.append(args.target_max_entropy)

        print(
            f"iter={it:03d} eta={eta:.4f} "
            f"hazan={ent_hazan[-1]:.4f} "
            f"uniform={ent_uniform[-1]:.4f} "
            f"theoretical={ent_best_theoretical[-1]:.4f}"
        )

    save_entropy_plot(out_dir, iterations, ent_hazan, ent_uniform, ent_best_theoretical)
    save_heatmap(out_dir, env, first_occ, "occupancy_iteration0")
    save_heatmap(out_dir, env, occ_mix, "occupancy_final")

    np.save(os.path.join(out_dir, "occupancy_final.npy"), occ_mix)
    np.save(os.path.join(out_dir, "occupancy_uniform_policy.npy"), uniform_action_occ)

    out = np.c_[iterations, ent_hazan, ent_uniform, ent_best_theoretical]
    np.savetxt(
        os.path.join(out_dir, "entropy_vs_iteration.csv"),
        out,
        delimiter=",",
        header="iteration,hazan_empirical_entropy_normalized,uniform_policy_empirical_entropy_normalized,best_theoretical_entropy_normalized",
        comments="",
    )

    with open(os.path.join(out_dir, "normalization_info.txt"), "w", encoding="utf-8") as f:
        f.write(f"reachable_states={len(reachable)}\\n")
        f.write(f"theoretical_max_entropy_raw={theoretical_max_entropy_raw}\\n")
        f.write(f"target_max_entropy={args.target_max_entropy}\\n")
        f.write(f"normalization_factor={norm_factor}\\n")
        f.write(f"uniform_policy_entropy_raw={uniform_action_entropy_raw}\\n")
        f.write(f"uniform_policy_entropy_normalized={uniform_action_entropy_raw * norm_factor}\\n")

    print(f"reachable_states={len(reachable)}")
    print(f"theoretical_max_raw={theoretical_max_entropy_raw:.6f}")
    print(f"normalization_factor={norm_factor:.6f}")
    print(f"saved={os.path.abspath(out_dir)}")


def parse_args():
    p = argparse.ArgumentParser(description="Run MaxEnt Frank-Wolfe in tabular CliffWorld.")
    p.add_argument("--exp_name", type=str, default="quick_cliff")
    p.add_argument("--height", type=int, default=4)
    p.add_argument("--width", type=int, default=12)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--n_rollouts", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--vi_iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eta", type=float, default=0.2)
    p.add_argument("--step_schedule", type=str, default="hazan", choices=["hazan", "constant"])
    p.add_argument("--grad_mode", type=str, default="inverse", choices=["inverse", "log"])
    p.add_argument("--target_max_entropy", type=float, default=5.0)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
