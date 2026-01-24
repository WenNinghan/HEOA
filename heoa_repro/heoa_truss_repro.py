import json
import numpy as np
from math import gamma, sin, pi, sqrt
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:
    plt = None
    Line2D = None

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


def plot_convergence(history, Maxiter, out_path):
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")
    iters = np.arange(1, Maxiter + 1)
    best = np.array(history["best_f"])
    mean = np.array(history["mean_f"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(iters, best, label="best fitness", linewidth=2.0)
    ax.plot(iters, mean, label="mean fitness", linewidth=1.5, alpha=0.8)
    switch = Maxiter / 4.0
    ax.axvline(switch, color="#444444", linestyle="--", linewidth=1.0)
    ax.text(switch + 1, max(mean.max(), best.max()) * 0.95,
            "exploration -> development", fontsize=9, color="#444444")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Penalized fitness")
    ax.set_title("HEOA convergence on truss optimization")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_population_snapshots(snapshots, bounds, out_path):
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")
    if not snapshots:
        return

    role_styles = {
        "leaders": {"color": "#d62728", "label": "leaders"},
        "explorers": {"color": "#1f77b4", "label": "explorers"},
        "followers": {"color": "#2ca02c", "label": "followers"},
        "losers": {"color": "#7f7f7f", "label": "losers"},
    }
    iters = sorted(snapshots.keys())
    n = len(iters)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, t in zip(axes, iters):
        snap = snapshots[t]
        X = snap["X"]
        fit = snap["fit"]
        roles = snap["roles"]
        stage = snap["stage"]

        if roles is None:
            ax.scatter(X[:, 0], X[:, 1], s=18, alpha=0.7, color="#1f77b4", label="population")
        else:
            for role, idxs in roles.items():
                if len(idxs) == 0:
                    continue
                style = role_styles[role]
                ax.scatter(X[idxs, 0], X[idxs, 1], s=18, alpha=0.7, color=style["color"],
                           label=style["label"])

        if "violations" in snap and "f_raw" in snap:
            eps = float(snap.get("epsilon", 0.0))
            penalty_val = float(snap.get("penalty", 1.0))
            best_idx = int(rank_population(snap["f_raw"], snap["violations"],
                                           penalty_val, eps=eps)[0])
        else:
            best_idx = int(np.argmin(fit))
        ax.scatter(X[best_idx, 0], X[best_idx, 1], marker="*", s=120, color="gold",
                   edgecolor="#222222", label="best")
        stage_label = stage
        if t == 1:
            stage_label = f"{stage}, chaotic init"
        ax.set_title(f"Iter {t} ({stage_label})")
        ax.set_xlim(bounds["lb"][0], bounds["ub"][0])
        ax.set_ylim(bounds["lb"][1], bounds["ub"][1])
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True, alpha=0.2)

    for ax in axes[len(iters):]:
        ax.axis("off")

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="population",
               markerfacecolor="#1f77b4", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="leaders",
               markerfacecolor=role_styles["leaders"]["color"], markersize=7),
        Line2D([0], [0], marker="o", color="w", label="explorers",
               markerfacecolor=role_styles["explorers"]["color"], markersize=7),
        Line2D([0], [0], marker="o", color="w", label="followers",
               markerfacecolor=role_styles["followers"]["color"], markersize=7),
        Line2D([0], [0], marker="o", color="w", label="losers",
               markerfacecolor=role_styles["losers"]["color"], markersize=7),
        Line2D([0], [0], marker="*", color="w", label="best",
               markerfacecolor="gold", markeredgecolor="#222222", markersize=12),
    ]
    fig.legend(handles=legend_items, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _constraint_grid(bounds, grid_n=200):
    x1 = np.linspace(bounds["lb"][0], bounds["ub"][0], grid_n)
    x2 = np.linspace(bounds["lb"][1], bounds["ub"][1], grid_n)
    X1, X2 = np.meshgrid(x1, x2)
    s2 = np.sqrt(2.0)
    denom = s2 * X1 * X1 + 2.0 * X1 * X2 + 1e-12
    g1 = (s2 * X1 + X2) / denom * TRUSS_P - TRUSS_SIGMA
    g2 = X2 / denom * TRUSS_P - TRUSS_SIGMA
    g3 = 1.0 / (s2 * X2 + X1 + 1e-12) * TRUSS_P - TRUSS_SIGMA
    gmax = np.maximum.reduce([g1, g2, g3])
    return X1, X2, gmax


def plot_feasibility_diagnostics(snapshots, history, bounds, out_path):
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")
    if not snapshots or history is None:
        return

    iters = sorted(snapshots.keys())
    n = len(iters)
    ncols = 2
    nrows_snap = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(6 * ncols, 4.5 * (nrows_snap + 0.9)))
    gs = fig.add_gridspec(nrows_snap + 1, ncols, height_ratios=[1] * nrows_snap + [0.9])

    X1, X2, gmax = _constraint_grid(bounds, grid_n=220)
    gmin = float(np.nanmin(gmax))

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

        ax.contourf(X1, X2, gmax, levels=[gmin, 0.0], colors=["#c7f0d8"], alpha=0.12)
        ax.contour(X1, X2, gmax, levels=[0.0], colors=["#2ca02c"], linewidths=1.0, alpha=0.7)

        if np.any(~feasible):
            ax.scatter(X[~feasible, 0], X[~feasible, 1], s=18, alpha=0.7,
                       color="#d62728", label="infeasible")
        if np.any(feasible):
            ax.scatter(X[feasible, 0], X[feasible, 1], s=18, alpha=0.7,
                       color="#2ca02c", label="feasible")

        if "violations" in snap and "f_raw" in snap:
            penalty_val = float(snap.get("penalty", 1.0))
            eps = float(snap.get("epsilon", 0.0))
            best_idx = int(rank_population(snap["f_raw"], violations,
                                           penalty_val, eps=eps)[0])
        else:
            best_idx = int(np.argmin(fit))
        ax.scatter(X[best_idx, 0], X[best_idx, 1], marker="*", s=120, color="gold",
                   edgecolor="#222222", label="best")
        stage_label = stage
        if t == 1:
            stage_label = f"{stage}, chaotic init"
        ax.set_title(f"Iter {t} ({stage_label})")
        ax.set_xlim(bounds["lb"][0], bounds["ub"][0])
        ax.set_ylim(bounds["lb"][1], bounds["ub"][1])
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.grid(True, alpha=0.2)

    for ax in axes[len(iters):]:
        ax.axis("off")

    ax_curve = fig.add_subplot(gs[-1, :])
    iters_full = np.arange(1, len(history["best_f"]) + 1)
    feas_ratio = np.array(history["feasible_ratio"])
    mean_viol = np.array(history["mean_violation"])
    best_viol = np.array(history["best_violation"])

    ax_curve.plot(iters_full, feas_ratio, color="#2ca02c", linewidth=1.7, label="feasible ratio")
    ax_curve.set_xlabel("Iteration")
    ax_curve.set_ylabel("Feasible ratio")
    ax_curve.set_title("Feasibility ratio and constraint violation")
    ax_curve.grid(True, alpha=0.2)

    ax_curve_b = ax_curve.twinx()
    ax_curve_b.plot(iters_full, mean_viol, color="#d62728", linewidth=1.4, label="mean violation")
    ax_curve_b.plot(iters_full, best_viol, color="#9467bd", linewidth=1.2, linestyle="--",
                    label="best violation")
    ax_curve_b.set_ylabel("Violation")

    lines0, labels0 = ax_curve.get_legend_handles_labels()
    lines1, labels1 = ax_curve_b.get_legend_handles_labels()
    ax_curve.legend(lines0 + lines1, labels0 + labels1, fontsize=8, loc="upper right")

    legend_items = [
        Line2D([0], [0], marker="o", color="w", label="feasible",
               markerfacecolor="#2ca02c", markersize=7),
        Line2D([0], [0], marker="o", color="w", label="infeasible",
               markerfacecolor="#d62728", markersize=7),
        Line2D([0], [0], marker="*", color="w", label="best",
               markerfacecolor="gold", markeredgecolor="#222222", markersize=12),
        Line2D([0], [0], color="#2ca02c", label="feasible boundary"),
    ]
    fig.legend(handles=legend_items, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heoa_report(history, Maxiter, bounds, params, out_path):
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting.")

    iters = np.arange(1, Maxiter + 1)
    best = np.array(history["best_f"])
    mean = np.array(history["mean_f"])
    diversity = np.array(history["diversity"])
    mean_viol = np.array(history["mean_violation"])
    best_viol = np.array(history["best_violation"])
    best_x = np.array(history["best_x"])
    stages = np.array([1 if s == "development" else 0 for s in history["stage"]])
    switch = int(params.get("stage_switch", Maxiter / 4))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax0, ax1, ax2, ax3 = axes.ravel()

    ax0.plot(iters, best, label="best fitness", linewidth=2.0)
    ax0.plot(iters, mean, label="mean fitness", linewidth=1.5, alpha=0.8)
    ax0.axvline(switch, color="#444444", linestyle="--", linewidth=1.0)
    ax0.set_xlabel("Iteration")
    ax0.set_ylabel("Penalized fitness")
    ax0.set_title("Convergence")
    ax0.grid(True, alpha=0.2)
    ax0.legend(fontsize=8)

    lb = params.get("lb", bounds["lb"])
    ub = params.get("ub", bounds["ub"])
    role = params.get("role_fractions", {})
    truss = params.get("truss", {})
    lines = [
        "HEOA parameters",
        f"N: {params.get('N')}  Maxiter: {params.get('Maxiter')}",
        f"A: {params.get('A'):.3f}  seed: {params.get('seed')}",
        f"chaos alpha: {params.get('alpha'):.2f}",
        f"levy gamma: {params.get('gamma'):.2f}",
        f"penalty: {params.get('penalty'):.1e}",
        "epsilon: p{0:.0f}% power {1:.2f} eps0 {2:.2e}".format(
            params.get("eps_percentile", 0.0),
            params.get("eps_power", 0.0),
            params.get("eps0", 0.0),
        ),
        "penalty adapt: target {0:.2f} inc {1:.2f} dec {2:.2f}".format(
            params.get("penalty_target", 0.0),
            params.get("penalty_increase", 0.0),
            params.get("penalty_decrease", 0.0),
        ),
        "penalty range: [{0:.1e},{1:.1e}]".format(
            params.get("penalty_min", 0.0),
            params.get("penalty_max", 0.0),
        ),
        "repair: {0} tries {1} sigma {2:.2f}".format(
            params.get("repair", False),
            params.get("repair_tries", 0),
            params.get("repair_sigma", 0.0),
        ),
        "explore feasible: prob {0:.2f} sigma {1:.2f}".format(
            params.get("explore_feasible_prob", 0.0),
            params.get("explore_feasible_sigma", 0.0),
        ),
        f"bounds: x1[{lb[0]:.2f},{ub[0]:.2f}] x2[{lb[1]:.2f},{ub[1]:.2f}]",
        f"stage switch: iter {switch}",
        "roles: leaders {0:.0f}%, explorers {1:.0f}%, followers {2:.0f}%, losers {3:.0f}%".format(
            role.get("leaders", 0.4) * 100,
            role.get("explorers", 0.4) * 100,
            role.get("followers", 0.1) * 100,
            role.get("losers", 0.1) * 100,
        ),
        f"truss: P={truss.get('P', TRUSS_P)}, sigma={truss.get('sigma', TRUSS_SIGMA)}, L={truss.get('L', TRUSS_L)}",
        "objective: (2*sqrt(2)*x1 + x2) * L",
        "features: chaotic init, levy flight, role-based development",
    ]
    ax1.axis("off")
    ax1.text(0.02, 0.98, "\n".join(lines), va="top", ha="left",
             fontsize=9, family="monospace")

    ax2.plot(iters, diversity, color="#1f77b4", linewidth=1.7, label="diversity (mean distance)")
    ax2.axvspan(1, switch, color="#1f77b4", alpha=0.06)
    ax2.axvspan(switch, Maxiter, color="#ff7f0e", alpha=0.04)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Diversity")
    ax2.set_title("Diversity and constraint violation")
    ax2.grid(True, alpha=0.2)

    ax2b = ax2.twinx()
    ax2b.plot(iters, mean_viol, color="#d62728", linewidth=1.5, label="mean violation")
    ax2b.plot(iters, best_viol, color="#9467bd", linewidth=1.2,
              linestyle="--", label="best violation")
    ax2b.set_ylabel("Mean violation")
    lines0, labels0 = ax2.get_legend_handles_labels()
    lines1, labels1 = ax2b.get_legend_handles_labels()
    ax2.legend(lines0 + lines1, labels0 + labels1, fontsize=8, loc="upper right")

    x1 = best_x[:, 0]
    x2 = best_x[:, 1]
    ax3.plot(x1, x2, color="#999999", linewidth=1.0, alpha=0.6)
    ax3.scatter(x1[stages == 0], x2[stages == 0], s=14, color="#1f77b4", label="exploration")
    ax3.scatter(x1[stages == 1], x2[stages == 1], s=14, color="#ff7f0e", label="development")
    ax3.scatter(x1[-1], x2[-1], marker="*", s=120, color="gold",
                edgecolor="#222222", label="final best")
    ax3.set_xlim(lb[0], ub[0])
    ax3.set_ylim(lb[1], ub[1])
    ax3.set_xlabel("x1")
    ax3.set_ylabel("x2")
    ax3.set_title("Best solution path")
    ax3.grid(True, alpha=0.2)
    ax3.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
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
