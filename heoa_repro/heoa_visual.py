import argparse
from dataclasses import dataclass
from math import gamma, sin, pi
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
except ImportError:  # pragma: no cover
    plt = None
    cm = None


@dataclass
class HEOAConfig:
    n_pop: int = 50
    max_iter: int = 300
    a: float = 0.6
    alpha: float = 4.0
    gamma_val: float = 1.5
    seed: int = 0
    snapshot_every: int = 20


def logistic_chaos_init(n: int, dim: int, lb: np.ndarray, ub: np.ndarray, alpha: float, rng: np.random.Generator):
    x = rng.random(dim)
    pop = np.empty((n, dim), dtype=float)
    for i in range(n):
        x = alpha * x * (1.0 - x)
        pop[i] = x
    return lb + (ub - lb) * pop


def levy_flight(dim: int, gamma_val: float, rng: np.random.Generator):
    sigma = (gamma(1 + gamma_val) * sin(pi * gamma_val / 2) /
             (gamma((1 + gamma_val) / 2) * gamma_val * 2 ** ((gamma_val - 1) / 2))) ** (1 / gamma_val)
    u = rng.normal(0, sigma, size=dim)
    v = rng.normal(0, 1, size=dim)
    return u / (np.abs(v) ** (1 / gamma_val))


def split_roles(rank_idx: np.ndarray, n: int):
    n_leader = int(0.4 * n)
    n_explorer = int(0.4 * n)
    n_follower = int(0.1 * n)
    leaders = rank_idx[:n_leader]
    explorers = rank_idx[n_leader:n_leader + n_explorer]
    followers = rank_idx[n_leader + n_explorer:n_leader + n_explorer + n_follower]
    losers = rank_idx[n_leader + n_explorer + n_follower:]
    return leaders, explorers, followers, losers


def sphere(x: np.ndarray) -> float:
    return float(np.sum(x ** 2))


def rastrigin(x: np.ndarray) -> float:
    return float(np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10))


def ackley(x: np.ndarray) -> float:
    n = x.size
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / n))
                 - np.exp(np.sum(np.cos(2 * np.pi * x)) / n) + 20 + np.e)


def griewank(x: np.ndarray) -> float:
    i = np.arange(1, x.size + 1)
    return float(np.sum(x ** 2) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1)


def rosenbrock(x: np.ndarray) -> float:
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))


BENCHMARKS: Dict[str, Tuple[Callable[[np.ndarray], float], Tuple[float, float]]] = {
    "sphere": (sphere, (-100.0, 100.0)),
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "ackley": (ackley, (-32.0, 32.0)),
    "griewank": (griewank, (-600.0, 600.0)),
    "rosenbrock": (rosenbrock, (-30.0, 30.0)),
}


def heoa_optimize(
    func: Callable[[np.ndarray], float],
    dim: int,
    lb: np.ndarray,
    ub: np.ndarray,
    cfg: HEOAConfig,
):
    rng = np.random.default_rng(cfg.seed)
    x = logistic_chaos_init(cfg.n_pop, dim, lb, ub, cfg.alpha, rng)

    f = np.array([func(xi) for xi in x])
    best_idx = int(np.argmin(f))
    x_best = x[best_idx].copy()
    f_best = float(f[best_idx])

    history = {
        "best_f": [],
        "mean_f": [],
        "diversity": [],
        "roles": [],
        "best_x": [],
        "stage": [],
        "snapshots": [],
    }

    for t in range(1, cfg.max_iter + 1):
        stage = "exploration" if t <= cfg.max_iter / 4 else "development"
        x_mean = np.mean(x, axis=0)

        if stage == "development":
            omega = 0.2 * np.cos((pi / 2.0) * (1.0 - t / cfg.max_iter))
            order = np.argsort(f)
            leaders, explorers, followers, losers = split_roles(order, cfg.n_pop)
            roles = (len(leaders), len(explorers), len(followers), len(losers))

            for i in leaders:
                r = rng.random()
                if r < cfg.a:
                    x_new = omega * x[i] * np.exp(-t / (rng.random() * cfg.max_iter))
                else:
                    r_n = rng.normal(0.0, 1.0, size=dim)
                    x_new = omega * x[i] + r_n * np.ones(dim)
                x[i] = np.clip(x_new, lb, ub)

            x_worst = x[order[-1]]
            for rank, i in enumerate(explorers, start=1):
                r_n = rng.normal(0.0, 1.0, size=dim)
                x_new = r_n * np.exp((x_worst ** 2 - x[i] ** 2) / (rank ** 2))
                x[i] = np.clip(x_new, lb, ub)

            for i in followers:
                r_d = rng.uniform(1.0, dim)
                x_new = x[i] + omega * r_d * (x_best - x[i])
                x[i] = np.clip(x_new, lb, ub)

            for i in losers:
                r_n = rng.normal(0.0, 1.0, size=dim)
                x_new = x_best + (x_best - x[i]) * r_n
                x[i] = np.clip(x_new, lb, ub)
        else:
            roles = (0, 0, 0, 0)
            for i in range(cfg.n_pop):
                beta = 0.2 * (1.0 - t / cfg.max_iter) ** t * (x[i] - x_mean)
                delta = rng.uniform(100.0, 2000.0)
                fjump = np.abs((ub - lb)) / delta
                fjump = np.where(fjump == 0.0, 1e-6, fjump)
                jump = np.floor(rng.random(dim) / fjump) * fjump
                x_new = (beta * (1.0 - t / cfg.max_iter) * (x[i] - x_best)
                         * levy_flight(dim, cfg.gamma_val, rng)
                         + x_best * (1.0 - t / cfg.max_iter)
                         + (x_mean - x_best) * jump)
                x[i] = np.clip(x_new, lb, ub)

        f = np.array([func(xi) for xi in x])
        best_idx = int(np.argmin(f))
        if f[best_idx] < f_best:
            f_best = float(f[best_idx])
            x_best = x[best_idx].copy()

        diversity = float(np.mean(np.linalg.norm(x - x.mean(axis=0), axis=1)))
        history["best_f"].append(float(f_best))
        history["mean_f"].append(float(np.mean(f)))
        history["diversity"].append(diversity)
        history["roles"].append(roles)
        history["best_x"].append(x_best.copy())
        history["stage"].append(stage)

        if cfg.snapshot_every > 0 and (t % cfg.snapshot_every == 0 or t == 1 or t == cfg.max_iter):
            history["snapshots"].append((t, x.copy(), f.copy(), stage))

    return x_best, f_best, history


def plot_results_2d(
    func: Callable[[np.ndarray], float],
    lb: np.ndarray,
    ub: np.ndarray,
    history: Dict,
    out_dir: Path,
    title: str,
):
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization.")

    out_dir.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(lb[0], ub[0], 200)
    ys = np.linspace(lb[1], ub[1], 200)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.vectorize(lambda a, b: func(np.array([a, b])))(xx, yy)

    # Surface
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, zz, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.85)
    ax.set_title(f"{title} - 目标函数曲面")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")
    fig.tight_layout()
    fig.savefig(out_dir / "surface.png", dpi=160)
    plt.close(fig)

    # Contour + snapshots
    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contourf(xx, yy, zz, levels=40, cmap=cm.viridis)
    fig.colorbar(cs, ax=ax, shrink=0.8)
    for t, x, _, stage in history["snapshots"]:
        ax.scatter(x[:, 0], x[:, 1], s=12, alpha=0.6, label=f"t={t} {stage}")
    ax.scatter(history["best_x"][-1][0], history["best_x"][-1][1], c="red", s=60, marker="*", label="best")
    ax.set_title(f"{title} - 种群搜索轨迹")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory.png", dpi=160)
    plt.close(fig)

    # Convergence
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["best_f"], label="best")
    ax.plot(history["mean_f"], label="mean", alpha=0.7)
    ax.set_yscale("log")
    ax.set_title(f"{title} - 收敛曲线")
    ax.set_xlabel("迭代")
    ax.set_ylabel("适应度")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=160)
    plt.close(fig)

    # Diversity + roles
    fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax[0].plot(history["diversity"], color="#1f77b4")
    ax[0].set_title("多样性变化")
    ax[0].set_ylabel("平均距离")
    roles = np.array(history["roles"])
    if roles.size > 0:
        ax[1].plot(roles[:, 0], label="leaders")
        ax[1].plot(roles[:, 1], label="explorers")
        ax[1].plot(roles[:, 2], label="followers")
        ax[1].plot(roles[:, 3], label="losers")
        ax[1].set_ylabel("人数")
        ax[1].set_xlabel("迭代")
        ax[1].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "diversity_roles.png", dpi=160)
    plt.close(fig)


def plot_results_nd(history: Dict, out_dir: Path, title: str):
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization.")

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["best_f"], label="best")
    ax.plot(history["mean_f"], label="mean", alpha=0.7)
    ax.set_yscale("log")
    ax.set_title(f"{title} - 收敛曲线")
    ax.set_xlabel("迭代")
    ax.set_ylabel("适应度")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "convergence.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["diversity"], color="#1f77b4")
    ax.set_title(f"{title} - 多样性变化")
    ax.set_xlabel("迭代")
    ax.set_ylabel("平均距离")
    fig.tight_layout()
    fig.savefig(out_dir / "diversity.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="HEOA 复现与可视化演示")
    parser.add_argument("--func", choices=BENCHMARKS.keys(), default="rastrigin")
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--n_pop", type=int, default=50)
    parser.add_argument("--max_iter", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--snapshot_every", type=int, default=20)
    parser.add_argument("--out", type=str, default="heoa_outputs")
    args = parser.parse_args()

    func, bound = BENCHMARKS[args.func]
    lb = np.full(args.dim, bound[0])
    ub = np.full(args.dim, bound[1])

    cfg = HEOAConfig(
        n_pop=args.n_pop,
        max_iter=args.max_iter,
        seed=args.seed,
        snapshot_every=args.snapshot_every,
    )

    x_best, f_best, history = heoa_optimize(func, args.dim, lb, ub, cfg)

    out_dir = Path(args.out)
    title = f"HEOA - {args.func} (dim={args.dim})"
    if args.dim == 2:
        plot_results_2d(func, lb, ub, history, out_dir, title)
    else:
        plot_results_nd(history, out_dir, title)

    summary = out_dir / "summary.txt"
    summary.write_text(
        "\n".join([
            f"func: {args.func}",
            f"dim: {args.dim}",
            f"best_f: {f_best:.8e}",
            f"best_x: {np.array2string(x_best, precision=4)}",
            f"n_pop: {args.n_pop}",
            f"max_iter: {args.max_iter}",
            f"seed: {args.seed}",
        ]),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()


def configure_matplotlib_chinese():
    if plt is None:
        return
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False