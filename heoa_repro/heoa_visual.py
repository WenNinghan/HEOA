import argparse
from dataclasses import dataclass
from math import gamma, sin, pi
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.patches import Patch
except ImportError:  # pragma: no cover
    plt = None
    cm = None
    Patch = None

JOURNAL_PALETTE = {
    "navy": "#1B4F72",
    "emerald": "#0B6E4F",
    "crimson": "#B03A2E",
    "ochre": "#C55A11",
    "plum": "#5B2C6F",
    "slate": "#475569",
    "gold": "#F59E0B",
}


def _configure_journal_style():
    if plt is None:
        return
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "axes.edgecolor": "#111111",
        "axes.linewidth": 1.30,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.5,
        "ytick.major.size": 4.5,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "xtick.minor.width": 0.9,
        "ytick.minor.width": 0.9,
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.35,
        "grid.color": "#94A3B8",
        "legend.fancybox": False,
        "legend.edgecolor": "#111111",
        "legend.facecolor": "#FFFFFF",
        "legend.framealpha": 0.96,
        "savefig.dpi": 300,
        "figure.dpi": 300,
    })


def _style_legend(leg):
    if leg is None:
        return
    frame = leg.get_frame()
    if frame is not None:
        frame.set_linewidth(1.1)
        frame.set_edgecolor("#111111")
        frame.set_facecolor("#FFFFFF")
        frame.set_alpha(0.96)
    for txt in leg.get_texts():
        txt.set_fontfamily("Times New Roman")
        txt.set_fontweight("bold")
    if leg.get_title() is not None:
        leg.get_title().set_fontfamily("Times New Roman")
        leg.get_title().set_fontweight("bold")


def _finalize_journal_figure(fig):
    for ax in fig.axes:
        is_3d = hasattr(ax, "zaxis")
        is_cbar = ax.get_label() == "<colorbar>"

        if not is_3d and ax.axison:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#111111")
                spine.set_linewidth(1.30)
            if is_cbar:
                ax.tick_params(direction="in", length=3.5, width=1.1, colors="#111111")
            else:
                ax.tick_params(axis="both", which="major", direction="in", top=True, right=True,
                               bottom=True, left=True, length=4.5, width=1.2, colors="#111111")
                ax.tick_params(axis="both", which="minor", direction="in", top=True, right=True,
                               bottom=True, left=True, length=2.5, width=0.9, colors="#111111")
        elif is_3d:
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.fill = True
                axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
                axis.pane.set_edgecolor("#CBD5E1")
                axis._axinfo["grid"]["color"] = (0.58, 0.64, 0.72, 0.35)
                axis._axinfo["grid"]["linestyle"] = ":"
                axis._axinfo["grid"]["linewidth"] = 0.7

        ax.title.set_fontfamily("Times New Roman")
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_fontfamily("Times New Roman")
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontfamily("Times New Roman")
        ax.yaxis.label.set_fontweight("bold")
        ax.xaxis.get_offset_text().set_fontfamily("Times New Roman")
        ax.xaxis.get_offset_text().set_fontweight("bold")
        ax.yaxis.get_offset_text().set_fontfamily("Times New Roman")
        ax.yaxis.get_offset_text().set_fontweight("bold")

        if is_3d:
            ax.zaxis.label.set_fontfamily("Times New Roman")
            ax.zaxis.label.set_fontweight("bold")
            ax.zaxis.get_offset_text().set_fontfamily("Times New Roman")
            ax.zaxis.get_offset_text().set_fontweight("bold")

        for tick in list(ax.xaxis.get_major_ticks()) + list(ax.yaxis.get_major_ticks()):
            tick.label1.set_fontfamily("Times New Roman")
            tick.label1.set_fontweight("bold")
            tick.label2.set_fontfamily("Times New Roman")
            tick.label2.set_fontweight("bold")

        for lbl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            lbl.set_fontfamily("Times New Roman")
            lbl.set_fontweight("bold")
        if is_3d:
            for lbl in ax.get_zticklabels():
                lbl.set_fontfamily("Times New Roman")
                lbl.set_fontweight("bold")

        for txt in ax.texts:
            txt.set_fontfamily("Times New Roman")
            txt.set_fontweight("bold")

        _style_legend(ax.get_legend())

    for txt in fig.texts:
        txt.set_fontfamily("Times New Roman")
        txt.set_fontweight("bold")
    for leg in fig.legends:
        _style_legend(leg)


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
    _configure_journal_style()

    out_dir.mkdir(parents=True, exist_ok=True)
    xs = np.linspace(lb[0], ub[0], 220)
    ys = np.linspace(lb[1], ub[1], 220)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.vectorize(lambda a, b: func(np.array([a, b])))(xx, yy)

    best_path = np.array(history["best_x"], dtype=float)
    x_opt = best_path[-1]
    f_opt = float(history["best_f"][-1])

    # 1. 3D Multimodal Surface
    fig = plt.figure(figsize=(8.6, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        xx, yy, zz,
        cmap="YlGnBu_r", rcount=160, ccount=160,
        linewidth=0.12, edgecolor=(0.12, 0.18, 0.26, 0.10),
        antialiased=True, alpha=0.94,
    )
    ax.scatter(
        [x_opt[0]], [x_opt[1]], [f_opt],
        marker="*", s=180, color=JOURNAL_PALETTE["gold"],
        edgecolor="#111111", linewidth=1.1, zorder=10,
        label=r"Global Optimum $\mathbf{x}^*$",
    )
    ax.view_init(elev=29, azim=-52)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.66, aspect=18, pad=0.13)
    cbar.set_label(r"Objective Value $f(\mathbf{x})$", fontsize=11.0)
    ax.set_title(rf"{title}: 3D Multimodal Landscape $f(\mathbf{{x}})$", fontsize=12.5, pad=12)
    ax.set_xlabel(r"$x_1$", fontsize=11.5, labelpad=8)
    ax.set_ylabel(r"$x_2$", fontsize=11.5, labelpad=8)
    ax.set_zlabel(r"$f(\mathbf{x})$", fontsize=11.5, labelpad=8)
    ax.legend(loc="upper left", bbox_to_anchor=(0.03, 0.95), fontsize=9.2)
    fig.subplots_adjust(left=0.05, right=0.88, bottom=0.06, top=0.91)
    _finalize_journal_figure(fig)
    fig.savefig(out_dir / "surface.png", dpi=300)
    plt.close(fig)

    # 2. Contour + Representative Trajectory Snapshots
    fig, ax = plt.subplots(figsize=(8.0, 6.4))
    cs = ax.contourf(xx, yy, zz, levels=44, cmap="YlGnBu_r", alpha=0.88)
    ax.contour(xx, yy, zz, levels=12, colors="#1E293B", linewidths=0.45, alpha=0.32)
    cbar = fig.colorbar(cs, ax=ax, shrink=0.88, pad=0.03)
    cbar.set_label(r"Objective Value $f(\mathbf{x})$", fontsize=11.0)

    snaps = history["snapshots"]
    if snaps:
        n_s = len(snaps)
        raw_idxs = [0, min(3, n_s - 1), n_s // 2, n_s - 1]
        sel_idxs = []
        for idx_s in raw_idxs:
            if idx_s not in sel_idxs:
                sel_idxs.append(idx_s)
        snap_styles = [
            (JOURNAL_PALETTE["ochre"], "^", 38),
            (JOURNAL_PALETTE["plum"], "s", 34),
            (JOURNAL_PALETTE["crimson"], "D", 32),
            (JOURNAL_PALETTE["emerald"], "o", 32),
        ]
        for k, idx_s in enumerate(sel_idxs):
            t_s, x_s, _, stage_s = snaps[idx_s]
            col, mk, sz = snap_styles[k % len(snap_styles)]
            st_lbl = "Explore" if stage_s == "exploration" else "Develop"
            ax.scatter(
                x_s[:, 0], x_s[:, 1], s=sz, alpha=0.86,
                color=col, marker=mk, edgecolor="#FFFFFF", linewidth=0.55,
                label=rf"$t = {t_s}$ ({st_lbl})", zorder=4 + k,
            )

    ax.plot(best_path[:, 0], best_path[:, 1], color="#FFFFFF", linewidth=2.3, alpha=0.85, zorder=8)
    ax.plot(
        best_path[:, 0], best_path[:, 1],
        color=JOURNAL_PALETTE["crimson"], linestyle="--", linewidth=1.5,
        label=r"Best Path $\mathbf{x}_{\mathrm{best}}(t)$", zorder=9,
    )
    ax.scatter(
        x_opt[0], x_opt[1],
        color=JOURNAL_PALETTE["gold"], edgecolor="#111111", linewidth=1.2,
        s=225, marker="*", label=r"Global Optimum $\mathbf{x}^*$", zorder=10,
    )
    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim(lb[1], ub[1])
    ax.set_title(rf"{title}: Population Search Trajectory", fontsize=12.5, pad=10)
    ax.set_xlabel(r"Decision Variable $x_1$", fontsize=11.5)
    ax.set_ylabel(r"Decision Variable $x_2$", fontsize=11.5)
    ax.legend(loc="upper right", fontsize=8.8)
    fig.tight_layout()
    _finalize_journal_figure(fig)
    fig.savefig(out_dir / "trajectory.png", dpi=300)
    plt.close(fig)

    # 3. Log-Scale Convergence Curve (clamped at machine precision 1e-16)
    eps_floor = 1e-16
    best_c = np.maximum(np.array(history["best_f"], dtype=float), eps_floor)
    mean_c = np.maximum(np.array(history["mean_f"], dtype=float), eps_floor)
    iters = np.arange(1, len(best_c) + 1)
    switch = len(iters) / 4.0

    fig, ax = plt.subplots(figsize=(7.8, 4.9))
    ax.axvspan(1, switch, color=JOURNAL_PALETTE["navy"], alpha=0.045, zorder=0)
    ax.axvspan(switch, len(iters), color=JOURNAL_PALETTE["emerald"], alpha=0.035, zorder=0)
    ax.axvline(switch, color=JOURNAL_PALETTE["emerald"], linestyle="-.", linewidth=1.35,
               label=rf"Stage Switch ($t = {int(switch)}$)", zorder=2)
    ax.axhline(eps_floor, color=JOURNAL_PALETTE["slate"], linestyle=":", linewidth=1.15,
               label=r"Machine Floor ($10^{-16}$)", zorder=2)
    ax.plot(
        iters, best_c,
        color=JOURNAL_PALETTE["navy"], linestyle="-", linewidth=2.25,
        marker="o", markevery=25, markersize=5.2,
        markerfacecolor="#FFFFFF", markeredgewidth=1.6,
        label=r"Best Fitness $f(\mathbf{x}_{\mathrm{best}})$", zorder=4,
    )
    ax.plot(
        iters, mean_c,
        color=JOURNAL_PALETTE["crimson"], linestyle="--", linewidth=1.85,
        marker="s", markevery=25, markersize=4.6,
        markerfacecolor="#FFFFFF", markeredgewidth=1.4, alpha=0.92,
        label=r"Mean Fitness $\bar{f}(t)$", zorder=3,
    )
    ax.set_yscale("log")
    ax.set_xlim(1, len(iters))
    ax.set_ylim(2e-18, max(float(mean_c.max()), 10.0) * 18.0)
    ax.set_title(rf"{title}: Logarithmic Convergence Profile", fontsize=12.5, pad=10)
    ax.set_xlabel(r"Iteration $t$", fontsize=11.5)
    ax.set_ylabel(r"Objective $\max(f(\mathbf{x}), 10^{-16})$", fontsize=11.5)
    ax.grid(True)
    ax.legend(loc="center right", bbox_to_anchor=(0.975, 0.44), fontsize=9.2)
    fig.tight_layout()
    _finalize_journal_figure(fig)
    fig.savefig(out_dir / "convergence.png", dpi=300)
    plt.close(fig)

    # 4. Population Diversity + Stratified Role Allocation
    div_arr = np.array(history["diversity"], dtype=float)
    roles = np.array(history["roles"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.5), sharex=True)

    axes[0].axvspan(1, switch, color=JOURNAL_PALETTE["navy"], alpha=0.045, zorder=0)
    axes[0].axvspan(switch, len(iters), color=JOURNAL_PALETTE["emerald"], alpha=0.035, zorder=0)
    axes[0].axvline(switch, color=JOURNAL_PALETTE["crimson"], linestyle="--", linewidth=1.3)
    axes[0].plot(
        iters, div_arr,
        color=JOURNAL_PALETTE["navy"], linewidth=2.1,
        label=r"Spatial Diversity $D(t)$",
    )
    div_top = max(float(div_arr.max()), 1.0) * 1.22
    axes[0].set_ylim(-0.04 * div_top, div_top)
    axes[0].text(
        switch * 0.50, div_top * 0.88, r"Exploration ($t \leq 75$)",
        fontsize=9.2, color=JOURNAL_PALETTE["navy"], ha="center", va="center",
    )
    axes[0].text(
        switch + (len(iters) - switch) * 0.50, div_top * 0.88,
        r"Development ($t > 75$)",
        fontsize=9.2, color=JOURNAL_PALETTE["emerald"], ha="center", va="center",
    )
    axes[0].set_title(r"(a) Population Spatial Diversity $D(t)$ Evolution", fontsize=11.8, pad=7)
    axes[0].set_ylabel(r"Mean Distance $D(t)$", fontsize=11.0)
    axes[0].grid(True)
    axes[0].legend(loc="lower right", fontsize=9.0)

    if roles.size > 0:
        n_total = int(np.max(np.sum(roles, axis=1)))
        if n_total <= 0:
            n_total = 50
        is_explore = np.sum(roles, axis=1) == 0
        explore_band = np.where(is_explore, float(n_total), 0.0)
        c0 = roles[:, 0]
        c1 = c0 + roles[:, 1]
        c2 = c1 + roles[:, 2]
        c3 = c2 + roles[:, 3]

        exp_mask = iters <= switch
        dev_mask = iters > switch

        axes[1].fill_between(
            iters[exp_mask], 0, explore_band[exp_mask], step="post",
            color=JOURNAL_PALETTE["slate"], alpha=0.28, edgecolor=JOURNAL_PALETTE["navy"],
            linewidth=1.1,
        )
        axes[1].fill_between(
            iters[dev_mask], 0, c0[dev_mask], step="pre",
            color=JOURNAL_PALETTE["crimson"], alpha=0.80, edgecolor="#FFFFFF", linewidth=0.9,
        )
        axes[1].fill_between(
            iters[dev_mask], c0[dev_mask], c1[dev_mask], step="pre",
            color=JOURNAL_PALETTE["navy"], alpha=0.80, edgecolor="#FFFFFF", linewidth=0.9,
        )
        axes[1].fill_between(
            iters[dev_mask], c1[dev_mask], c2[dev_mask], step="pre",
            color=JOURNAL_PALETTE["emerald"], alpha=0.84, edgecolor="#FFFFFF", linewidth=0.9,
        )
        axes[1].fill_between(
            iters[dev_mask], c2[dev_mask], c3[dev_mask], step="pre",
            color=JOURNAL_PALETTE["ochre"], alpha=0.84, edgecolor="#FFFFFF", linewidth=0.9,
        )
        axes[1].axvline(switch, color="#111111", linestyle="--", linewidth=1.35, zorder=5)

        axes[1].text(
            switch * 0.50, n_total * 0.50, "Unified\nExploration\n(100%)",
            fontsize=8.8, color="#1E293B", ha="center", va="center",
        )
        mid_dev = switch + (len(iters) - switch) * 0.52
        if np.any(dev_mask):
            l_val = float(roles[dev_mask, 0][0])
            e_val = float(roles[dev_mask, 1][0])
            axes[1].text(mid_dev, l_val * 0.50, rf"Leaders ($40\%, n={int(l_val)}$)",
                         fontsize=8.8, color="#FFFFFF", ha="center", va="center")
            axes[1].text(mid_dev, l_val + e_val * 0.50, rf"Explorers ($40\%, n={int(e_val)}$)",
                         fontsize=8.8, color="#FFFFFF", ha="center", va="center")

        role_handles = [
            Patch(facecolor=JOURNAL_PALETTE["slate"], edgecolor=JOURNAL_PALETTE["navy"], alpha=0.40,
                  label="Unified Cohort"),
            Patch(facecolor=JOURNAL_PALETTE["crimson"], alpha=0.82, label=r"Leaders ($40\%$)"),
            Patch(facecolor=JOURNAL_PALETTE["navy"], alpha=0.82, label=r"Explorers ($40\%$)"),
            Patch(facecolor=JOURNAL_PALETTE["emerald"], alpha=0.86, label=r"Followers ($10\%$)"),
            Patch(facecolor=JOURNAL_PALETTE["ochre"], alpha=0.86, label=r"Losers ($10\%$)"),
        ]
        axes[1].set_xlim(1, len(iters))
        axes[1].set_ylim(0, n_total * 1.32)
        axes[1].set_title(rf"(b) Two-Stage Cohort Role Stratification ($N = {n_total}$)",
                          fontsize=11.8, pad=7)
        axes[1].set_ylabel(r"Cohort Allocation $N_k$", fontsize=11.0)
        axes[1].set_xlabel(r"Iteration $t$", fontsize=11.5)
        axes[1].grid(True)
        axes[1].legend(handles=role_handles, loc="upper center", bbox_to_anchor=(0.5, 0.985),
                       ncol=3, fontsize=8.2, columnspacing=0.9, handletextpad=0.4)

    fig.tight_layout()
    _finalize_journal_figure(fig)
    fig.savefig(out_dir / "diversity_roles.png", dpi=300)
    plt.close(fig)


def plot_results_nd(history: Dict, out_dir: Path, title: str):
    if plt is None:
        raise RuntimeError("matplotlib is required for visualization.")
    _configure_journal_style()

    out_dir.mkdir(parents=True, exist_ok=True)
    eps_floor = 1e-16
    best_c = np.maximum(np.array(history["best_f"], dtype=float), eps_floor)
    mean_c = np.maximum(np.array(history["mean_f"], dtype=float), eps_floor)
    iters = np.arange(1, len(best_c) + 1)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(iters, best_c, color=JOURNAL_PALETTE["navy"], linewidth=2.2,
            label=r"Best Fitness $f(\mathbf{x}_{\mathrm{best}})$")
    ax.plot(iters, mean_c, color=JOURNAL_PALETTE["crimson"], linestyle="--", linewidth=1.8,
            alpha=0.9, label=r"Mean Fitness $\bar{f}(t)$")
    ax.set_yscale("log")
    ax.set_xlim(1, len(iters))
    ax.set_title(rf"{title}: Convergence Curve", fontsize=12.5, pad=10)
    ax.set_xlabel(r"Iteration $t$", fontsize=11.5)
    ax.set_ylabel(r"Objective $\max(f(\mathbf{x}), 10^{-16})$", fontsize=11.5)
    ax.grid(True)
    ax.legend(fontsize=9.5)
    fig.tight_layout()
    _finalize_journal_figure(fig)
    fig.savefig(out_dir / "convergence.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(iters, history["diversity"], color=JOURNAL_PALETTE["navy"], linewidth=2.1,
            label=r"Spatial Diversity $D(t)$")
    ax.set_xlim(1, len(iters))
    ax.set_title(rf"{title}: Population Spatial Diversity", fontsize=12.5, pad=10)
    ax.set_xlabel(r"Iteration $t$", fontsize=11.5)
    ax.set_ylabel(r"Mean Distance $D(t)$", fontsize=11.5)
    ax.grid(True)
    ax.legend(fontsize=9.5)
    fig.tight_layout()
    _finalize_journal_figure(fig)
    fig.savefig(out_dir / "diversity.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="HEOA Benchmark Reproduction and Visualization")
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
    title = rf"HEOA on {args.func.capitalize()} ($D={args.dim}$)"
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