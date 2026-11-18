import json
import numpy as np
from math import gamma, sin, pi, sqrt
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ImportError:
    plt = None
    Line2D = None
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
        is_twinx = getattr(ax, "_is_twinx", False)
        has_twinx = getattr(ax, "_has_twinx", False)
        is_cbar = ax.get_label() == "<colorbar>"

        if not is_3d and ax.axison:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#111111")
                spine.set_linewidth(1.30)
            if is_cbar:
                ax.tick_params(direction="in", length=3.5, width=1.1, colors="#111111")
            elif is_twinx:
                ax.tick_params(axis="y", which="both", direction="in", left=False, right=True,
                               length=4.5, width=1.2, colors="#111111")
                ax.tick_params(axis="x", which="both", bottom=False, top=False)
            elif has_twinx:
                ax.tick_params(axis="x", which="both", direction="in", top=True, bottom=True,
                               length=4.5, width=1.2, colors="#111111")
                ax.tick_params(axis="y", which="both", direction="in", left=True, right=False,
                               length=4.5, width=1.2, colors="#111111")
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

TRUSS_L = 100.0
TRUSS_P = 2.0
TRUSS_SIGMA = 2.0


def levy_flight(dim, gamma_val=1.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # Mantegna's algorithm
    sigma = (gamma(1 + gamma_val) * sin(pi * gamma_val / 2) /
             (gamma((1 + gamma_val) / 2) * gamma_val * 2 ** ((gamma_val - 1) / 2))) ** (1 / gamma_val)
    u = rng.normal(0, sigma, size=dim)
    v = rng.normal(0, 1, size=dim)
    return u / (np.abs(v) ** (1 / gamma_val))


def logistic_chaos_init(N, dim, lb, ub, alpha=4.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    # Generate a chaotic sequence per dimension, length N
    x = rng.random(dim)
    pop = np.empty((N, dim), dtype=float)
    for i in range(N):
        x = alpha * x * (1.0 - x)
        pop[i] = x
    return lb + (ub - lb) * pop


def truss_objective(x):
    x1, x2 = x[0], x[1]
    return (2.0 * sqrt(2.0) * x1 + x2) * TRUSS_L


def truss_constraints(x):
    x1, x2 = x[0], x[1]
    s2 = sqrt(2.0)
    eps = 1e-12
    denom = s2 * x1 * x1 + 2.0 * x1 * x2 + eps
    g1 = (s2 * x1 + x2) / denom * TRUSS_P - TRUSS_SIGMA
    g2 = (x2) / denom * TRUSS_P - TRUSS_SIGMA
    g3 = (1.0) / (s2 * x2 + x1 + eps) * TRUSS_P - TRUSS_SIGMA
    return np.array([g1, g2, g3])


def fitness_with_penalty(x, penalty=1e8):
    f = truss_objective(x)
    g = truss_constraints(x)
    viol = np.maximum(0.0, g).sum()
    return f + penalty * viol


def evaluate_population(X, penalty=1e8):
    f_raw = np.array([truss_objective(x) for x in X])
    g = np.array([truss_constraints(x) for x in X])
    violations = np.maximum(0.0, g).sum(axis=1)
    fit = f_raw + penalty * violations
    return f_raw, violations, fit


def rank_population(f_raw, violations, penalty, eps=0.0):
    fit = f_raw + penalty * violations
    feasible = violations <= eps
    if np.any(feasible):
        feasible_idx = np.where(feasible)[0]
        infeasible_idx = np.where(~feasible)[0]
        order_feasible = feasible_idx[np.argsort(f_raw[feasible_idx])]
        order_infeasible = infeasible_idx[np.argsort(fit[infeasible_idx])]
        return np.concatenate([order_feasible, order_infeasible])
    return np.argsort(fit)


def is_better(viol_a, f_a, viol_b, f_b, eps=0.0, tol=1e-12):
    a_feas = viol_a <= eps
    b_feas = viol_b <= eps
    if a_feas and not b_feas:
        return True
    if not a_feas and b_feas:
        return False
    if a_feas and b_feas:
        return f_a < f_b - tol
    return viol_a < viol_b - tol


def epsilon_schedule(t, Maxiter, eps0, power=2.0, eps_min=0.0):
    if Maxiter <= 0:
        return eps_min
    ratio = max(0.0, 1.0 - t / Maxiter)
    return max(eps_min, eps0 * (ratio ** power))


def update_penalty(penalty, feasible_ratio, target=0.2, increase=1.05, decrease=1.02,
                   p_min=1e6, p_max=1e12):
    if feasible_ratio < target:
        return min(penalty * increase, p_max)
    return max(penalty / decrease, p_min)


def constraint_violation(x):
    return float(np.maximum(0.0, truss_constraints(x)).sum())


def repair_candidate(x, xbest, lb, ub, rng, eps=0.0, max_tries=6, sigma=0.12):
    best = x.copy()
    best_v = constraint_violation(best)
    if best_v <= eps:
        return best

    for frac in (0.25, 0.5, 0.75, 1.0):
        cand = x + frac * (xbest - x)
        cand = np.clip(cand, lb, ub)
        v = constraint_violation(cand)
        if v < best_v:
            best, best_v = cand, v
            if best_v <= eps:
                return best

    scale = (ub - lb) * sigma
    for _ in range(max_tries):
        w = rng.uniform(0.0, 1.0)
        cand = w * best + (1.0 - w) * xbest
        cand = cand + rng.normal(0.0, 1.0, size=best.shape) * scale
        cand = np.clip(cand, lb, ub)
        v = constraint_violation(cand)
        if v < best_v:
            best, best_v = cand, v
            if best_v <= eps:
                return best
    return best


def sample_feasible(feasible_pool, lb, ub, rng, sigma=0.08):
    idx = int(rng.integers(0, feasible_pool.shape[0]))
    base = feasible_pool[idx]
    noise = rng.normal(0.0, 1.0, size=base.shape) * (ub - lb) * sigma
    return np.clip(base + noise, lb, ub)


def split_roles(rank_idx, N):
    idx = np.array(rank_idx, dtype=int)
    n_leader = int(0.4 * N)
    n_explorer = int(0.4 * N)
    n_follower = int(0.1 * N)
    leaders = idx[:n_leader]
    explorers = idx[n_leader:n_leader + n_explorer]
    followers = idx[n_leader + n_explorer:n_leader + n_explorer + n_follower]
    losers = idx[n_leader + n_explorer + n_follower:]
    return idx, leaders, explorers, followers, losers


def default_snapshot_iters(Maxiter):
    switch = max(1, int(Maxiter / 4))
    candidates = [1, switch, min(Maxiter, switch + 1), Maxiter]
    seen = set()
    iters = []
    for t in candidates:
        if 1 <= t <= Maxiter and t not in seen:
            seen.add(t)
            iters.append(t)
    return iters


def heoa_truss(N=50, Maxiter=300, A=0.6, seed=0, trace=False, snapshot_iters=None,
               alpha=4.0, gamma_val=1.5, penalty=1e8, eps_percentile=70.0, eps_power=2.0,
               penalty_target=0.3, penalty_increase=1.05, penalty_decrease=1.02,
               penalty_min=1e6, penalty_max=1e12, repair=True, repair_tries=6,
               repair_sigma=0.12, explore_feasible_prob=0.7, explore_feasible_sigma=0.05):
    rng = np.random.default_rng(seed)
    dim = 2
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    X = logistic_chaos_init(N, dim, lb, ub, alpha=alpha, rng=rng)
    penalty_t = float(penalty)
    f_raw, violations, fit = evaluate_population(X, penalty=penalty_t)
    eps0 = float(np.percentile(violations, eps_percentile)) if violations.size else 0.0
    eps = epsilon_schedule(0, Maxiter, eps0, power=eps_power)
    order = rank_population(f_raw, violations, penalty_t, eps=eps)
    best_idx = int(order[0])
    Xbest = X[best_idx].copy()
    best_f_raw = float(f_raw[best_idx])
    best_viol = float(violations[best_idx])
    fbest = float(fit[best_idx])

    history = None
    snapshots = None
    snapshot_set = set()
    if trace:
        history = {
            "best_f": [],
            "mean_f": [],
            "best_x": [],
            "stage": [],
            "diversity": [],
            "mean_violation": [],
            "best_violation": [],
            "feasible_ratio": [],
            "epsilon": [],
            "penalty": [],
        }
        snapshots = {}
        if snapshot_iters is None:
            snapshot_set = set(default_snapshot_iters(Maxiter))
        else:
            snapshot_set = {int(t) for t in snapshot_iters if 1 <= int(t) <= Maxiter}

    for t in range(1, Maxiter + 1):
        f_raw, violations, fit = evaluate_population(X, penalty=penalty_t)
        eps = epsilon_schedule(t, Maxiter, eps0, power=eps_power)
        feasible_mask = violations <= eps
        feasible_pool = X[feasible_mask]
        order = rank_population(f_raw, violations, penalty_t, eps=eps)
        best_idx = int(order[0])
        if is_better(violations[best_idx], f_raw[best_idx], best_viol, best_f_raw, eps=eps):
            best_viol = float(violations[best_idx])
            best_f_raw = float(f_raw[best_idx])
            fbest = float(fit[best_idx])
            Xbest = X[best_idx].copy()
        Xmean = X.mean(axis=0)
        stage = "exploration" if t <= Maxiter / 4 else "development"
        roles = None
        leaders = explorers = followers = losers = None

        if stage == "development":
            # Human development stage
            idx, leaders, explorers, followers, losers = split_roles(order, N)
            roles = {
                "leaders": leaders,
                "explorers": explorers,
                "followers": followers,
                "losers": losers,
            }

        if trace and t in snapshot_set:
            snapshots[t] = {
                "X": X.copy(),
                "fit": fit.copy(),
                "f_raw": f_raw.copy(),
                "violations": violations.copy(),
                "epsilon": float(eps),
                "penalty": float(penalty_t),
                "roles": roles,
                "stage": stage,
            }

        if stage == "development":
            omega = 0.2 * np.cos((pi / 2.0) * (1.0 - t / Maxiter))

            # Leaders (Eq. 8)
            for i in leaders:
                R = rng.random()
                if R < A:
                    Xnew = omega * X[i] * np.exp(-t / (rng.random() * Maxiter))
                else:
                    Rn = rng.normal(0.0, 1.0, size=dim)
                    Xnew = omega * X[i] + Rn * np.ones(dim)
                Xnew = np.clip(Xnew, lb, ub)
                if repair and constraint_violation(Xnew) > eps:
                    Xnew = repair_candidate(Xnew, Xbest, lb, ub, rng, eps=eps,
                                            max_tries=repair_tries, sigma=repair_sigma)
                if feasible_pool.size and constraint_violation(Xnew) > eps and rng.random() < explore_feasible_prob:
                    Xnew = sample_feasible(feasible_pool, lb, ub, rng, sigma=explore_feasible_sigma)
                X[i] = Xnew

            # Explorers (Eq. 10)
            Xworst = X[idx[-1]]
            for rank, i in enumerate(explorers, start=1):
                Rn = rng.normal(0.0, 1.0, size=dim)
                Xnew = Rn * np.exp((Xworst ** 2 - X[i] ** 2) / (rank ** 2))
                Xnew = np.clip(Xnew, lb, ub)
                if repair and constraint_violation(Xnew) > eps:
                    Xnew = repair_candidate(Xnew, Xbest, lb, ub, rng, eps=eps,
                                            max_tries=repair_tries, sigma=repair_sigma)
                if feasible_pool.size and constraint_violation(Xnew) > eps and rng.random() < explore_feasible_prob:
                    Xnew = sample_feasible(feasible_pool, lb, ub, rng, sigma=explore_feasible_sigma)
                X[i] = Xnew

            # Followers (Eq. 11)
            for i in followers:
                Rd = rng.uniform(1.0, dim)
                Xnew = X[i] + omega * Rd * (Xbest - X[i])
                Xnew = np.clip(Xnew, lb, ub)
                if repair and constraint_violation(Xnew) > eps:
                    Xnew = repair_candidate(Xnew, Xbest, lb, ub, rng, eps=eps,
                                            max_tries=repair_tries, sigma=repair_sigma)
                if feasible_pool.size and constraint_violation(Xnew) > eps and rng.random() < explore_feasible_prob:
                    Xnew = sample_feasible(feasible_pool, lb, ub, rng, sigma=explore_feasible_sigma)
                X[i] = Xnew

            # Losers (Eq. 12)
            for i in losers:
                Rn = rng.normal(0.0, 1.0, size=dim)
                Xnew = Xbest + (Xbest - X[i]) * Rn
                Xnew = np.clip(Xnew, lb, ub)
                if repair and constraint_violation(Xnew) > eps:
                    Xnew = repair_candidate(Xnew, Xbest, lb, ub, rng, eps=eps,
                                            max_tries=repair_tries, sigma=repair_sigma)
                if feasible_pool.size and constraint_violation(Xnew) > eps and rng.random() < explore_feasible_prob:
                    Xnew = sample_feasible(feasible_pool, lb, ub, rng, sigma=explore_feasible_sigma)
                X[i] = Xnew
        else:
            # Human exploration stage
            for i in range(N):
                beta = 0.2 * (1.0 - t / Maxiter) * (X[i] - Xmean)
                delta = rng.uniform(100.0, 2000.0)
                fjump = (ub - lb) / delta
                fjump = np.where(fjump == 0.0, 1e-6, fjump)
                jump = np.floor(rng.random(dim) / fjump) * fjump
                jump *= rng.choice([-1.0, 1.0], size=dim)
                Xnew = (beta * (1.0 - t / Maxiter) * (X[i] - Xbest)
                        * levy_flight(dim, gamma_val=gamma_val, rng=rng)
                        + Xbest * (1.0 - t / Maxiter)
                        + (Xmean - Xbest) * jump)
                Xnew = np.clip(Xnew, lb, ub)
                viol_new = constraint_violation(Xnew)
                if feasible_pool.size and viol_new > eps and rng.random() < explore_feasible_prob:
                    Xnew = sample_feasible(feasible_pool, lb, ub, rng, sigma=explore_feasible_sigma)
                    viol_new = constraint_violation(Xnew)
                if repair and viol_new > eps:
                    Xnew = repair_candidate(Xnew, Xbest, lb, ub, rng, eps=eps,
                                            max_tries=repair_tries, sigma=repair_sigma)
                X[i] = Xnew

        # Update best
        f_raw, violations, fit = evaluate_population(X, penalty=penalty_t)
        eps = epsilon_schedule(t, Maxiter, eps0, power=eps_power)
        order = rank_population(f_raw, violations, penalty_t, eps=eps)
        best_idx = int(order[0])
        if is_better(violations[best_idx], f_raw[best_idx], best_viol, best_f_raw, eps=eps):
            best_viol = float(violations[best_idx])
            best_f_raw = float(f_raw[best_idx])
            fbest = float(fit[best_idx])
            Xbest = X[best_idx].copy()

        if trace:
            Xmean_after = X.mean(axis=0)
            diversity = float(np.mean(np.linalg.norm(X - Xmean_after, axis=1)))
            history["best_f"].append(float(fbest))
            history["mean_f"].append(float(fit.mean()))
            history["best_x"].append(Xbest.tolist())
            history["stage"].append(stage)
            history["diversity"].append(diversity)
            history["mean_violation"].append(float(violations.mean()))
            history["best_violation"].append(float(violations[best_idx]))
            history["feasible_ratio"].append(float(np.mean(violations <= 1e-9)))
            history["epsilon"].append(float(eps))
            history["penalty"].append(float(penalty_t))

        feasible_ratio = float(np.mean(violations <= 1e-9))
        penalty_t = update_penalty(penalty_t, feasible_ratio, target=penalty_target,
                                   increase=penalty_increase, decrease=penalty_decrease,
                                   p_min=penalty_min, p_max=penalty_max)

    # Final evaluation
    f_raw = truss_objective(Xbest)
    g = truss_constraints(Xbest)
    viol = np.maximum(0.0, g).sum()
    feasible = viol <= 1e-9

    result = {
        "x": Xbest.tolist(),
        "f": float(f_raw),
        "g": g.tolist(),
        "feasible": bool(feasible)
    }
    if trace:
        result["history"] = history
        result["snapshots"] = snapshots
        result["bounds"] = {"lb": lb.tolist(), "ub": ub.tolist()}
        result["params"] = {
            "N": int(N),
            "Maxiter": int(Maxiter),
            "A": float(A),
            "seed": int(seed),
            "alpha": float(alpha),
            "gamma": float(gamma_val),
            "penalty": float(penalty),
            "eps_percentile": float(eps_percentile),
            "eps_power": float(eps_power),
            "eps0": float(eps0),
            "penalty_target": float(penalty_target),
            "penalty_increase": float(penalty_increase),
            "penalty_decrease": float(penalty_decrease),
            "penalty_min": float(penalty_min),
            "penalty_max": float(penalty_max),
            "repair": bool(repair),
            "repair_tries": int(repair_tries),
            "repair_sigma": float(repair_sigma),
            "explore_feasible_prob": float(explore_feasible_prob),
            "explore_feasible_sigma": float(explore_feasible_sigma),
            "lb": lb.tolist(),
            "ub": ub.tolist(),
            "stage_switch": int(Maxiter / 4),
            "role_fractions": {
                "leaders": 0.4,
                "explorers": 0.4,
                "followers": 0.1,
                "losers": 0.1,
            },
            "truss": {
                "P": TRUSS_P,
                "sigma": TRUSS_SIGMA,
                "L": TRUSS_L,
            },
        }
    return result


def _objective_and_constraint_grids(bounds, grid_n=240):
    x1 = np.linspace(bounds["lb"][0], bounds["ub"][0], grid_n)
    x2 = np.linspace(bounds["lb"][1], bounds["ub"][1], grid_n)
    X1, X2 = np.meshgrid(x1, x2)
    s2 = np.sqrt(2.0)
    denom = s2 * X1 * X1 + 2.0 * X1 * X2 + 1e-12
    g1 = (s2 * X1 + X2) / denom * TRUSS_P - TRUSS_SIGMA
    g2 = X2 / denom * TRUSS_P - TRUSS_SIGMA
    g3 = 1.0 / (s2 * X2 + X1 + 1e-12) * TRUSS_P - TRUSS_SIGMA
    gmax = np.maximum.reduce([g1, g2, g3])
    F = (2.0 * s2 * X1 + X2) * TRUSS_L
    return X1, X2, F, gmax


def _constraint_grid(bounds, grid_n=200):
    X1, X2, _, gmax = _objective_and_constraint_grids(bounds, grid_n=grid_n)
    return X1, X2, gmax


def plot_convergence(history, Maxiter, out_path):
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")
    _configure_journal_style()

    iters = np.arange(1, Maxiter + 1)
    best = np.maximum(np.array(history["best_f"], dtype=float), 1.0)
    mean = np.maximum(np.array(history["mean_f"], dtype=float), 1.0)
    raw_best = None
    if "best_x" in history and len(history["best_x"]) == len(iters):
        raw_best = np.maximum(
            np.array([truss_objective(x) for x in history["best_x"]], dtype=float), 1.0
        )

    fig, ax = plt.subplots(figsize=(7.8, 5.1))
    switch = Maxiter / 4.0

    ax.axvspan(1, switch, color=JOURNAL_PALETTE["navy"], alpha=0.045, zorder=0)
    ax.axvspan(switch, Maxiter, color=JOURNAL_PALETTE["emerald"], alpha=0.035, zorder=0)

    ax.plot(
        iters, best,
        color=JOURNAL_PALETTE["navy"], linestyle="-", linewidth=2.25,
        marker="o", markevery=25, markersize=5.2,
        markerfacecolor="#FFFFFF", markeredgewidth=1.6,
        label=r"Best Penalized $\tilde{f}_{\mathrm{best}}(t)$", zorder=4,
    )
    ax.plot(
        iters, mean,
        color=JOURNAL_PALETTE["ochre"], linestyle="--", linewidth=1.85,
        marker="s", markevery=25, markersize=4.6,
        markerfacecolor="#FFFFFF", markeredgewidth=1.4, alpha=0.92,
        label=r"Mean Penalized $\tilde{f}_{\mathrm{mean}}(t)$", zorder=3,
    )
    if raw_best is not None:
        ax.plot(
            iters, raw_best,
            color=JOURNAL_PALETTE["emerald"], linestyle="-.", linewidth=2.05,
            marker="^", markevery=30, markersize=5.0,
            markerfacecolor="#FFFFFF", markeredgewidth=1.5,
            label=r"Structural Volume $f(\mathbf{x}_{\mathrm{best}})$", zorder=4,
        )

    ax.axvline(switch, color=JOURNAL_PALETTE["crimson"], linestyle="--", linewidth=1.35, zorder=2)
    final_opt = float(best[-1])
    ax.axhline(final_opt, color=JOURNAL_PALETTE["slate"], linestyle=":", linewidth=1.15, zorder=2)

    ax.set_yscale("log")
    ymax = max(float(mean.max()), float(best.max())) * 35.0
    ax.set_xlim(1, Maxiter)
    ax.set_ylim(25.0, ymax)

    ax.text(
        switch * 0.50, ymax * 0.36,
        r"Exploration ($t \leq 75$)",
        fontsize=9.5, color=JOURNAL_PALETTE["navy"], ha="center", va="center",
    )
    ax.text(
        switch + (Maxiter - switch) * 0.52, ymax * 0.36,
        r"Development ($t > 75$)",
        fontsize=9.5, color=JOURNAL_PALETTE["emerald"], ha="center", va="center",
    )
    ax.annotate(
        rf"$f(\mathbf{{x}}^*) = {final_opt:.4f}$",
        xy=(Maxiter, final_opt),
        xytext=(Maxiter - 118, final_opt * 6.5),
        fontsize=9.5,
        color=JOURNAL_PALETTE["emerald"],
        bbox=dict(boxstyle="square,pad=0.25", facecolor="#FFFFFF",
                  edgecolor=JOURNAL_PALETTE["emerald"], linewidth=0.9, alpha=0.95),
        arrowprops=dict(arrowstyle="->", color=JOURNAL_PALETTE["emerald"], lw=1.25),
    )

    ax.set_xlabel(r"Iteration $t$", fontsize=11.5)
    ax.set_ylabel(r"Objective / Penalized Fitness $\tilde{f}(\mathbf{x})$", fontsize=11.5)
    ax.set_title(r"HEOA Convergence on Constrained 3-Bar Truss Design", fontsize=12.5, pad=10)
    ax.grid(True)
    ax.legend(loc="center left", bbox_to_anchor=(0.025, 0.43), fontsize=9.5)

    fig.tight_layout()
    _finalize_journal_figure(fig)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_population_snapshots(snapshots, bounds, out_path):
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")
    if not snapshots:
        return
    _configure_journal_style()

    role_styles = {
        "leaders": {
            "color": JOURNAL_PALETTE["crimson"], "marker": "o", "size": 38,
            "label": r"Leaders ($40\%$)",
        },
        "explorers": {
            "color": JOURNAL_PALETTE["navy"], "marker": "^", "size": 38,
            "label": r"Explorers ($40\%$)",
        },
        "followers": {
            "color": JOURNAL_PALETTE["emerald"], "marker": "s", "size": 36,
            "label": r"Followers ($10\%$)",
        },
        "losers": {
            "color": JOURNAL_PALETTE["plum"], "marker": "D", "size": 34,
            "label": r"Losers ($10\%$)",
        },
    }
    iters = sorted(snapshots.keys())
    n = len(iters)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.2, 8.8), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    X1, X2, F, gmax = _objective_and_constraint_grids(bounds, grid_n=240)
    gmin = float(np.nanmin(gmax))
    panel_tags = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for idx_panel, (ax, t) in enumerate(zip(axes, iters)):
        snap = snapshots[t]
        X = snap["X"]
        fit = snap["fit"]
        roles = snap["roles"]
        stage = snap["stage"]

        ax.contourf(X1, X2, gmax, levels=[gmin, 0.0], colors=["#D1FAE5"], alpha=0.34, zorder=1)
        ax.contour(X1, X2, F, levels=8, colors=["#94A3B8"], linewidths=0.75,
                   linestyles="--", alpha=0.55, zorder=1)
        ax.contour(X1, X2, gmax, levels=[0.0], colors=[JOURNAL_PALETTE["emerald"]],
                   linewidths=1.55, zorder=2)

        if roles is None:
            ax.scatter(
                X[:, 0], X[:, 1], s=36, alpha=0.86,
                color=JOURNAL_PALETTE["navy"], marker="o",
                edgecolor="#FFFFFF", linewidth=0.55, zorder=4,
            )
        else:
            for role in ("losers", "followers", "explorers", "leaders"):
                idxs = roles.get(role, [])
                if len(idxs) == 0:
                    continue
                st = role_styles[role]
                ax.scatter(
                    X[idxs, 0], X[idxs, 1], s=st["size"], alpha=0.88,
                    color=st["color"], marker=st["marker"],
                    edgecolor="#FFFFFF", linewidth=0.55, zorder=4,
                )

        if "violations" in snap and "f_raw" in snap:
            eps = float(snap.get("epsilon", 0.0))
            penalty_val = float(snap.get("penalty", 1.0))
            best_idx = int(rank_population(snap["f_raw"], snap["violations"],
                                           penalty_val, eps=eps)[0])
        else:
            best_idx = int(np.argmin(fit))
        ax.scatter(
            X[best_idx, 0], X[best_idx, 1], marker="*", s=195,
            color=JOURNAL_PALETTE["gold"], edgecolor="#111111", linewidth=1.1, zorder=6,
        )

        tag = panel_tags[idx_panel] if idx_panel < len(panel_tags) else ""
        if t == 1:
            desc = "Chaotic Initialization"
        elif stage == "exploration":
            desc = "Exploration Stage"
        elif t == iters[-1]:
            desc = "Converged Optimum"
        else:
            desc = "Role Stratification"
        ax.set_title(rf"{tag} $t = {t}$ ({desc})", fontsize=11.5, pad=7)
        ax.set_xlim(bounds["lb"][0], bounds["ub"][0])
        ax.set_ylim(bounds["lb"][1], bounds["ub"][1])
        if idx_panel // ncols == nrows - 1:
            ax.set_xlabel(r"Design Variable $x_1$ ($A_1$)", fontsize=11.0)
        if idx_panel % ncols == 0:
            ax.set_ylabel(r"Design Variable $x_2$ ($A_2$)", fontsize=11.0)
        ax.grid(True)

    for ax in axes[len(iters):]:
        ax.axis("off")

    legend_items = [
        Line2D([0], [0], marker="o", linestyle="None",
               label=r"Unified Cohort ($t \leq 75$)",
               markerfacecolor=JOURNAL_PALETTE["navy"], markeredgecolor="#FFFFFF", markersize=7.5),
        Line2D([0], [0], marker="o", linestyle="None",
               label=role_styles["leaders"]["label"],
               markerfacecolor=role_styles["leaders"]["color"], markeredgecolor="#FFFFFF", markersize=7.5),
        Line2D([0], [0], marker="^", linestyle="None",
               label=role_styles["explorers"]["label"],
               markerfacecolor=role_styles["explorers"]["color"], markeredgecolor="#FFFFFF", markersize=7.5),
        Line2D([0], [0], marker="s", linestyle="None",
               label=role_styles["followers"]["label"],
               markerfacecolor=role_styles["followers"]["color"], markeredgecolor="#FFFFFF", markersize=7.2),
        Line2D([0], [0], marker="D", linestyle="None",
               label=role_styles["losers"]["label"],
               markerfacecolor=role_styles["losers"]["color"], markeredgecolor="#FFFFFF", markersize=6.8),
        Line2D([0], [0], marker="*", linestyle="None",
               label=r"Best Candidate $\mathbf{x}_{\mathrm{best}}$",
               markerfacecolor=JOURNAL_PALETTE["gold"], markeredgecolor="#111111", markersize=12.5),
        Line2D([0], [0], color=JOURNAL_PALETTE["emerald"], linewidth=1.6,
               label=r"Active Boundary $g_1(\mathbf{x}) = 0$"),
        Patch(facecolor="#D1FAE5", edgecolor=JOURNAL_PALETTE["emerald"], alpha=0.55,
              label=r"Feasible Domain $\Omega_{\mathrm{feas}}$"),
    ]
    fig.legend(
        handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, 0.985),
        ncol=4, fontsize=9.2, frameon=True, columnspacing=1.2, handletextpad=0.5,
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.07, top=0.875, hspace=0.24, wspace=0.14)
    _finalize_journal_figure(fig)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_feasibility_diagnostics(snapshots, history, bounds, out_path):
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")
    if not snapshots or history is None:
        return
    _configure_journal_style()

    iters = sorted(snapshots.keys())
    n = len(iters)
    ncols = 2
    nrows_snap = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(10.5, 11.4))
    gs = fig.add_gridspec(
        nrows_snap + 1, ncols,
        height_ratios=[1.0] * nrows_snap + [0.95],
        hspace=0.31, wspace=0.18,
    )

    X1, X2, F, gmax = _objective_and_constraint_grids(bounds, grid_n=240)
    gmin = float(np.nanmin(gmax))
    panel_tags = ["(a)", "(b)", "(c)", "(d)"]

    axes = []
    for i, t in enumerate(iters):
        r = i // ncols
        c = i % ncols
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)
        snap = snapshots[t]
        X = snap["X"]
        fit = snap["fit"]
        stage = snap["stage"]
        if "violations" in snap:
            violations = np.array(snap["violations"], dtype=float)
        else:
            violations = np.array([np.maximum(0.0, truss_constraints(x)).sum() for x in X])
        feasible = violations <= 1e-9
        eps = float(snap.get("epsilon", 0.0))

        ax.contourf(X1, X2, gmax, levels=[gmin, 0.0], colors=["#D1FAE5"], alpha=0.38, zorder=1)
        ax.contour(X1, X2, F, levels=7, colors=["#CBD5E1"], linewidths=0.7,
                   linestyles="--", alpha=0.6, zorder=1)
        ax.contour(X1, X2, gmax, levels=[0.0], colors=[JOURNAL_PALETTE["emerald"]],
                   linewidths=1.65, zorder=2)
        if 1e-4 < eps < 25.0:
            ax.contour(X1, X2, gmax, levels=[eps], colors=[JOURNAL_PALETTE["ochre"]],
                       linewidths=1.2, linestyles="-.", alpha=0.85, zorder=2)

        if np.any(~feasible):
            ax.scatter(
                X[~feasible, 0], X[~feasible, 1], s=34, alpha=0.84,
                color=JOURNAL_PALETTE["crimson"], marker="X",
                edgecolor="#FFFFFF", linewidth=0.4, zorder=4,
            )
        if np.any(feasible):
            ax.scatter(
                X[feasible, 0], X[feasible, 1], s=38, alpha=0.90,
                color=JOURNAL_PALETTE["emerald"], marker="o",
                edgecolor="#FFFFFF", linewidth=0.55, zorder=5,
            )

        if "violations" in snap and "f_raw" in snap:
            penalty_val = float(snap.get("penalty", 1.0))
            best_idx = int(rank_population(snap["f_raw"], violations,
                                           penalty_val, eps=eps)[0])
        else:
            best_idx = int(np.argmin(fit))
        ax.scatter(
            X[best_idx, 0], X[best_idx, 1], marker="*", s=190,
            color=JOURNAL_PALETTE["gold"], edgecolor="#111111", linewidth=1.1, zorder=6,
        )

        tag = panel_tags[i] if i < len(panel_tags) else ""
        stage_desc = "Chaotic Init" if t == 1 else ("Exploration" if stage == "exploration" else "Development")
        ax.set_title(rf"{tag} $t = {t}$ ({stage_desc}, $\epsilon = {eps:.2f}$)", fontsize=11.0, pad=6)
        ax.set_xlim(bounds["lb"][0], bounds["ub"][0])
        ax.set_ylim(bounds["lb"][1], bounds["ub"][1])
        ax.set_xlabel(r"$x_1$", fontsize=10.5)
        ax.set_ylabel(r"$x_2$", fontsize=10.5)
        ax.grid(True)

    ax_curve = fig.add_subplot(gs[-1, :])
    ax_curve._has_twinx = True
    iters_full = np.arange(1, len(history["best_f"]) + 1)
    feas_ratio = np.array(history["feasible_ratio"], dtype=float)
    mean_viol = np.array(history["mean_violation"], dtype=float)
    best_viol = np.array(history["best_violation"], dtype=float)
    eps_hist = np.array(history.get("epsilon", np.zeros_like(iters_full)), dtype=float)
    switch = len(iters_full) / 4.0

    ax_curve.axvline(switch, color=JOURNAL_PALETTE["slate"], linestyle="--", linewidth=1.25, zorder=1)
    ax_curve.plot(
        iters_full, feas_ratio,
        color=JOURNAL_PALETTE["emerald"], linewidth=2.15, linestyle="-",
        label=r"Feasible Ratio $\rho_{\mathrm{feas}}(t)$", zorder=3,
    )
    ax_curve.set_xlim(1, len(iters_full))
    ax_curve.set_ylim(-0.04, 1.18)
    ax_curve.set_xlabel(r"Iteration $t$", fontsize=11.0)
    ax_curve.set_ylabel(r"Feasible Ratio $\rho_{\mathrm{feas}}$", fontsize=11.0,
                        color=JOURNAL_PALETTE["emerald"])
    ax_curve.set_title(
        r"(e) Evolution of Strict Feasibility Ratio and Constraint Violation $\nu(\mathbf{x})$",
        fontsize=11.5, pad=7,
    )
    ax_curve.grid(True)

    ax_curve_b = ax_curve.twinx()
    ax_curve_b._is_twinx = True
    ax_curve_b.plot(
        iters_full, mean_viol,
        color=JOURNAL_PALETTE["crimson"], linewidth=1.85, linestyle="-",
        label=r"Mean Violation $\bar{\nu}(t)$", zorder=2,
    )
    ax_curve_b.plot(
        iters_full, best_viol,
        color=JOURNAL_PALETTE["plum"], linewidth=1.75, linestyle="--",
        label=r"Best Violation $\nu(\mathbf{x}_{\mathrm{best}})$", zorder=2,
    )
    if np.any(eps_hist > 0):
        ax_curve_b.plot(
            iters_full, eps_hist,
            color=JOURNAL_PALETTE["ochre"], linewidth=1.60, linestyle="-.",
            label=r"Relaxation Threshold $\epsilon(t)$", zorder=2,
        )
    vmax = max(float(mean_viol.max()), float(best_viol.max()), float(eps_hist.max()), 1.0)
    ax_curve_b.set_ylim(-0.03 * vmax, vmax * 1.24)
    ax_curve_b.set_ylabel(r"Constraint Violation $\nu(\mathbf{x})$", fontsize=11.0,
                          color=JOURNAL_PALETTE["crimson"])

    lines0, labels0 = ax_curve.get_legend_handles_labels()
    lines1, labels1 = ax_curve_b.get_legend_handles_labels()
    ax_curve_b.legend(
        lines0 + lines1, labels0 + labels1,
        fontsize=8.8, loc="upper center", bbox_to_anchor=(0.54, 0.98), ncol=2,
    )

    legend_items = [
        Line2D([0], [0], marker="o", linestyle="None",
               label=r"Feasible ($\nu(\mathbf{x}) \leq 10^{-9}$)",
               markerfacecolor=JOURNAL_PALETTE["emerald"], markeredgecolor="#FFFFFF", markersize=7.5),
        Line2D([0], [0], marker="X", linestyle="None",
               label=r"Infeasible ($\nu(\mathbf{x}) > 0$)",
               markerfacecolor=JOURNAL_PALETTE["crimson"], markeredgecolor="#FFFFFF", markersize=7.5),
        Line2D([0], [0], marker="*", linestyle="None",
               label=r"Best Candidate $\mathbf{x}_{\mathrm{best}}$",
               markerfacecolor=JOURNAL_PALETTE["gold"], markeredgecolor="#111111", markersize=12.5),
        Line2D([0], [0], color=JOURNAL_PALETTE["emerald"], linewidth=1.65,
               label=r"Active Boundary $g_1(\mathbf{x}) = 0$"),
        Line2D([0], [0], color=JOURNAL_PALETTE["ochre"], linewidth=1.3, linestyle="-.",
               label=r"$\epsilon(t)$-Relaxed Boundary"),
    ]
    fig.legend(
        handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, 0.988),
        ncol=3, fontsize=9.2, frameon=True, columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.08, right=0.915, bottom=0.055, top=0.905)
    _finalize_journal_figure(fig)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_heoa_report(history, Maxiter, bounds, params, out_path):
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")
    _configure_journal_style()

    iters = np.arange(1, Maxiter + 1)
    best = np.maximum(np.array(history["best_f"], dtype=float), 1.0)
    mean = np.maximum(np.array(history["mean_f"], dtype=float), 1.0)
    diversity = np.array(history["diversity"], dtype=float)
    mean_viol = np.array(history["mean_violation"], dtype=float)
    best_viol = np.array(history["best_violation"], dtype=float)
    eps_hist = np.array(history.get("epsilon", np.zeros_like(iters)), dtype=float)
    pen_hist = np.maximum(np.array(history.get("penalty", np.ones_like(iters)), dtype=float), 1.0)
    best_x = np.array(history["best_x"], dtype=float)
    raw_best = np.maximum(np.array([truss_objective(x) for x in best_x], dtype=float), 1.0)
    stages = np.array([1 if s == "development" else 0 for s in history["stage"]])
    switch = int(params.get("stage_switch", Maxiter / 4))
    lb = params.get("lb", bounds["lb"])
    ub = params.get("ub", bounds["ub"])

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2))
    ax0, ax1, ax2, ax3 = axes.ravel()

    # Panel (a): Log-scale Convergence
    ax0.axvspan(1, switch, color=JOURNAL_PALETTE["navy"], alpha=0.045, zorder=0)
    ax0.axvspan(switch, Maxiter, color=JOURNAL_PALETTE["emerald"], alpha=0.035, zorder=0)
    ax0.plot(iters, best, color=JOURNAL_PALETTE["navy"], linewidth=2.1,
             marker="o", markevery=30, markersize=4.8, markerfacecolor="#FFFFFF", markeredgewidth=1.4,
             label=r"Best Penalized $\tilde{f}_{\mathrm{best}}$")
    ax0.plot(iters, mean, color=JOURNAL_PALETTE["ochre"], linewidth=1.7, linestyle="--",
             marker="s", markevery=30, markersize=4.4, markerfacecolor="#FFFFFF", markeredgewidth=1.3,
             label=r"Mean Penalized $\tilde{f}_{\mathrm{mean}}$")
    ax0.plot(iters, raw_best, color=JOURNAL_PALETTE["emerald"], linewidth=1.9, linestyle="-.",
             label=r"Structural Volume $f(\mathbf{x}_{\mathrm{best}})$")
    ax0.axvline(switch, color=JOURNAL_PALETTE["crimson"], linestyle="--", linewidth=1.25)
    ax0.set_yscale("log")
    ax0.set_xlim(1, Maxiter)
    ax0.set_ylim(60.0, max(float(mean.max()), float(best.max())) * 30.0)
    ax0.set_xlabel(r"Iteration $t$", fontsize=10.8)
    ax0.set_ylabel(r"Fitness / Volume (Log Scale)", fontsize=10.8)
    ax0.set_title(r"(a) Penalized Fitness & Structural Volume Convergence", fontsize=11.5, pad=7)
    ax0.grid(True)
    ax0.legend(loc="center left", bbox_to_anchor=(0.02, 0.44), fontsize=8.6)

    # Panel (b): Dynamic epsilon-Relaxation & Adaptive Penalty Schedule + Summary Callout
    ax1._has_twinx = True
    ax1.axvline(switch, color=JOURNAL_PALETTE["slate"], linestyle="--", linewidth=1.2, zorder=1)
    ax1.plot(iters, eps_hist, color=JOURNAL_PALETTE["ochre"], linewidth=2.15, linestyle="-",
             label=r"$\epsilon(t)$ Threshold")
    ax1.set_xlim(1, Maxiter)
    eps_top = max(float(eps_hist.max()), 1.0) * 1.22
    ax1.set_ylim(-0.03 * eps_top, eps_top)
    ax1.set_xlabel(r"Iteration $t$", fontsize=10.8)
    ax1.set_ylabel(r"Relaxation Threshold $\epsilon(t)$", fontsize=10.8,
                   color=JOURNAL_PALETTE["ochre"])
    ax1.set_title(r"(b) Dynamic $\epsilon$-Constraint & Penalty Schedule", fontsize=11.5, pad=7)
    ax1.grid(True)

    ax1b = ax1.twinx()
    ax1b._is_twinx = True
    ax1b.plot(iters, pen_hist, color=JOURNAL_PALETTE["plum"], linewidth=2.0, linestyle="--",
              label=r"Penalty Factor $\lambda_t$")
    ax1b.set_yscale("log")
    ax1b.set_ylim(
        max(float(pen_hist.min()) * 0.4, 1e5),
        float(pen_hist.max()) * 18.0,
    )
    ax1b.set_ylabel(r"Adaptive Penalty $\lambda_t$ (Log Scale)", fontsize=10.8,
                    color=JOURNAL_PALETTE["plum"])

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1b.get_legend_handles_labels()
    ax1b.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=8.6)

    opt_f = float(raw_best[-1])
    opt_x1, opt_x2 = float(best_x[-1, 0]), float(best_x[-1, 1])
    summary_box = "\n".join([
        rf"$N = {params.get('N')},\; T = {params.get('Maxiter')},\; \alpha = {params.get('alpha'):.1f},\; \gamma = {params.get('gamma'):.1f}$",
        rf"$\epsilon_0 = {params.get('eps0', 0.0):.2f}\;(p={params.get('eps_power', 2.0):.1f}),\; \mathrm{{ Seed}} = {params.get('seed')}$",
        rf"$\mathbf{{x}}^* = [{opt_x1:.4f},\; {opt_x2:.4f}]$",
        rf"$f(\mathbf{{x}}^*) = {opt_f:.4f}\; (g_1 \leq 0)$",
    ])
    ax1.text(
        0.97, 0.25, summary_box,
        transform=ax1.transAxes, va="bottom", ha="right", fontsize=8.3,
        bbox=dict(boxstyle="square,pad=0.40", facecolor="#F8FAFC",
                  edgecolor="#111111", linewidth=1.0, alpha=0.96),
        zorder=6,
    )

    # Panel (c): Population Diversity & Constraint Violation
    ax2._has_twinx = True
    ax2.axvspan(1, switch, color=JOURNAL_PALETTE["navy"], alpha=0.045, zorder=0)
    ax2.axvspan(switch, Maxiter, color=JOURNAL_PALETTE["ochre"], alpha=0.035, zorder=0)
    ax2.axvline(switch, color=JOURNAL_PALETTE["slate"], linestyle="--", linewidth=1.2)
    ax2.plot(iters, diversity, color=JOURNAL_PALETTE["navy"], linewidth=1.9,
             label=r"Diversity $D(t)$")
    ax2.set_xlim(1, Maxiter)
    ax2.set_ylim(0.0, max(float(diversity.max()), 0.1) * 1.22)
    ax2.set_xlabel(r"Iteration $t$", fontsize=10.8)
    ax2.set_ylabel(r"Population Diversity $D(t)$", fontsize=10.8,
                   color=JOURNAL_PALETTE["navy"])
    ax2.set_title(r"(c) Spatial Diversity & Constraint Violation Decay", fontsize=11.5, pad=7)
    ax2.grid(True)

    ax2b = ax2.twinx()
    ax2b._is_twinx = True
    ax2b.plot(iters, mean_viol, color=JOURNAL_PALETTE["crimson"], linewidth=1.8,
              label=r"Mean Violation $\bar{\nu}(t)$")
    ax2b.plot(iters, best_viol, color=JOURNAL_PALETTE["emerald"], linewidth=1.65,
              linestyle="--", label=r"Best Violation $\nu_{\mathrm{best}}(t)$")
    vmax2 = max(float(mean_viol.max()), float(best_viol.max()), 1.0)
    ax2b.set_ylim(-0.02 * vmax2, vmax2 * 1.22)
    ax2b.set_ylabel(r"Constraint Violation $\nu(\mathbf{x})$", fontsize=10.8,
                    color=JOURNAL_PALETTE["crimson"])

    lines0, labels0 = ax2.get_legend_handles_labels()
    lines1, labels1 = ax2b.get_legend_handles_labels()
    ax2b.legend(lines0 + lines1, labels0 + labels1, fontsize=8.5, loc="upper right")

    # Panel (d): Optimal Trajectory on Constrained Design Space
    X1, X2, F, gmax = _objective_and_constraint_grids(bounds, grid_n=240)
    gmin = float(np.nanmin(gmax))
    ax3.contourf(X1, X2, gmax, levels=[gmin, 0.0], colors=["#D1FAE5"], alpha=0.38, zorder=1)
    cs = ax3.contour(X1, X2, F, levels=8, colors=["#94A3B8"], linewidths=0.75,
                     linestyles="--", alpha=0.65, zorder=1)
    ax3.clabel(cs, inline=True, fontsize=7.5, fmt="%.0f")
    ax3.contour(X1, X2, gmax, levels=[0.0], colors=[JOURNAL_PALETTE["emerald"]],
                linewidths=1.65, zorder=2)

    x1 = best_x[:, 0]
    x2 = best_x[:, 1]
    ax3.plot(x1, x2, color=JOURNAL_PALETTE["slate"], linewidth=1.15, alpha=0.72, zorder=3)
    ax3.scatter(
        x1[stages == 0], x2[stages == 0], s=26, marker="o",
        color=JOURNAL_PALETTE["navy"], edgecolor="#FFFFFF", linewidth=0.45,
        label=r"Exploration ($t \leq 75$)", zorder=4,
    )
    ax3.scatter(
        x1[stages == 1], x2[stages == 1], s=26, marker="s",
        color=JOURNAL_PALETTE["ochre"], edgecolor="#FFFFFF", linewidth=0.45,
        label=r"Development ($t > 75$)", zorder=4,
    )
    ax3.scatter(
        x1[-1], x2[-1], marker="*", s=210,
        color=JOURNAL_PALETTE["gold"], edgecolor="#111111", linewidth=1.15,
        label=rf"Optimum $\mathbf{{x}}^* ({x1[-1]:.3f}, {x2[-1]:.3f})$", zorder=6,
    )
    ax3.plot([], [], color=JOURNAL_PALETTE["emerald"], linewidth=1.65,
             label=r"Active Boundary $g_1(\mathbf{x}) = 0$")
    ax3.set_xlim(lb[0], ub[0])
    ax3.set_ylim(lb[1], ub[1])
    ax3.set_xlabel(r"Design Variable $x_1$ ($A_1$)", fontsize=10.8)
    ax3.set_ylabel(r"Design Variable $x_2$ ($A_2$)", fontsize=10.8)
    ax3.set_title(r"(d) Best-So-Far Trajectory on Constrained Design Space", fontsize=11.5, pad=7)
    ax3.grid(True)
    ax3.legend(fontsize=8.5, loc="upper left")

    fig.subplots_adjust(left=0.075, right=0.92, bottom=0.07, top=0.94, hspace=0.28, wspace=0.28)
    _finalize_journal_figure(fig)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_experiments(runs=30, N=50, Maxiter=300, A=0.6, alpha=4.0, gamma_val=1.5,
                    penalty=1e8, eps_percentile=70.0, eps_power=2.0, penalty_target=0.3,
                    penalty_increase=1.05, penalty_decrease=1.02, penalty_min=1e6,
                    penalty_max=1e12, repair=True, repair_tries=10, repair_sigma=0.12,
                    explore_feasible_prob=0.7, explore_feasible_sigma=0.05):
    results = []
    for seed in range(runs):
        res = heoa_truss(N=N, Maxiter=Maxiter, A=A, seed=seed,
                         alpha=alpha, gamma_val=gamma_val, penalty=penalty,
                         eps_percentile=eps_percentile, eps_power=eps_power,
                         penalty_target=penalty_target, penalty_increase=penalty_increase,
                         penalty_decrease=penalty_decrease, penalty_min=penalty_min,
                         penalty_max=penalty_max, repair=repair, repair_tries=repair_tries,
                         repair_sigma=repair_sigma,
                         explore_feasible_prob=explore_feasible_prob,
                         explore_feasible_sigma=explore_feasible_sigma)
        results.append(res)

    f_vals = np.array([r["f"] for r in results])
    best_idx = int(np.argmin(f_vals))

    summary = {
        "runs": runs,
        "N": N,
        "Maxiter": Maxiter,
        "best": float(f_vals.min()),
        "worst": float(f_vals.max()),
        "mean": float(f_vals.mean()),
        "std": float(f_vals.std(ddof=0)),
        "median": float(np.median(f_vals)),
        "best_x": results[best_idx]["x"],
        "best_g": results[best_idx]["g"],
        "best_feasible": results[best_idx]["feasible"],
    }

    return summary, results


def main():
    out_dir = Path(__file__).resolve().parent
    runs = 30
    N = 50
    Maxiter = 300
    A = 0.6
    alpha = 4.0
    gamma_val = 1.5
    penalty = 1e8
    eps_percentile = 70.0
    eps_power = 2.0
    penalty_target = 0.3
    penalty_increase = 1.05
    penalty_decrease = 1.02
    penalty_min = 1e6
    penalty_max = 1e12
    repair = True
    repair_tries = 10
    repair_sigma = 0.12
    explore_feasible_prob = 0.7
    explore_feasible_sigma = 0.05
    summary, results = run_experiments(runs=runs, N=N, Maxiter=Maxiter, A=A,
                                       alpha=alpha, gamma_val=gamma_val, penalty=penalty,
                                       eps_percentile=eps_percentile, eps_power=eps_power,
                                       penalty_target=penalty_target,
                                       penalty_increase=penalty_increase,
                                       penalty_decrease=penalty_decrease,
                                       penalty_min=penalty_min, penalty_max=penalty_max,
                                       repair=repair, repair_tries=repair_tries,
                                       repair_sigma=repair_sigma,
                                       explore_feasible_prob=explore_feasible_prob,
                                       explore_feasible_sigma=explore_feasible_sigma)

    # Save detailed results
    (out_dir / "heoa_truss_results.json").write_text(json.dumps({
        "summary": summary,
        "results": results
    }, indent=2))

    # Save a simple CSV
    csv_lines = ["run,x1,x2,f,g1,g2,g3,feasible"]
    for i, r in enumerate(results):
        x1, x2 = r["x"]
        g1, g2, g3 = r["g"]
        csv_lines.append(f"{i},{x1},{x2},{r['f']},{g1},{g2},{g3},{r['feasible']}")
    (out_dir / "heoa_truss_results.csv").write_text("\n".join(csv_lines))

    # Print summary for CLI
    print("Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    # Visualization for the best run
    best_seed = int(np.argmin([r["f"] for r in results]))
    trace_result = heoa_truss(N=N, Maxiter=Maxiter, A=A, seed=best_seed, trace=True,
                              alpha=alpha, gamma_val=gamma_val, penalty=penalty,
                              eps_percentile=eps_percentile, eps_power=eps_power,
                              penalty_target=penalty_target,
                              penalty_increase=penalty_increase,
                              penalty_decrease=penalty_decrease,
                              penalty_min=penalty_min, penalty_max=penalty_max,
                              repair=repair, repair_tries=repair_tries,
                              repair_sigma=repair_sigma,
                              explore_feasible_prob=explore_feasible_prob,
                              explore_feasible_sigma=explore_feasible_sigma)

    if plt is None:
        print("matplotlib not installed; skipping plots. Install with: python -m pip install matplotlib")
        return

    conv_path = out_dir / "heoa_truss_convergence.png"
    pop_path = out_dir / "heoa_truss_population.png"
    feas_path = out_dir / "heoa_truss_feasibility.png"
    report_path = out_dir / "heoa_truss_report.png"
    plot_convergence(trace_result["history"], Maxiter, conv_path)
    plot_population_snapshots(trace_result["snapshots"], trace_result["bounds"], pop_path)
    plot_feasibility_diagnostics(trace_result["snapshots"], trace_result["history"],
                                 trace_result["bounds"], feas_path)
    plot_heoa_report(trace_result["history"], Maxiter, trace_result["bounds"],
                     trace_result["params"], report_path)
    print(f"Saved plots: {conv_path.name}, {pop_path.name}, {feas_path.name}, {report_path.name}")


if __name__ == "__main__":
    main()
