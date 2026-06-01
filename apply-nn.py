"""
apply-nn.py
===========
Evaluates the trained neural network collision operator against
f_pre / f_post pairs saved from the Kármán vortex street simulation,
and optionally runs a full NN-driven LBM animation.

Usage:
    python apply-nn.py --model-path artifacts-run-all-tensorflow/example_network.keras \
                       --data-dir output \
                       --out-dir eval_results

The script expects pairs of files named:
    fpre_XXXXXX.npy   (pre-collision distribution)
    fpost_XXXXXX.npy  (post-collision distribution, BGK ground truth)
"""

import os
import glob
import logging
import argparse
from typing import cast
import numpy as np
import matplotlib
from matplotlib.artist import Artist
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import keras
from keras import Model, backend as K
from numba import njit
from lbm_ml import LB_stencil, rmsre

log = logging.getLogger(__name__)

c, w, _cs2, _ = LB_stencil()


@njit
def equilibrium(rho, ux, uy):
    Nx, Ny = rho.shape
    feq = np.zeros((Nx, Ny, 9))
    usqr = ux**2 + uy**2
    for i in range(9):
        cu = c[i, 0] * ux + c[i, 1] * uy
        feq[:, :, i] = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usqr)
    return feq


# ──────────────────────────────────────────────
# Helpers — must match training definitions
# ──────────────────────────────────────────────


def normalize(f):
    """Divide each sample by its total density (same as training pipeline)."""
    norm = np.sum(f, axis=1, keepdims=True)
    return f / norm, norm


def f_to_velocity(f, Nx, Ny, Q=9):
    """Calculates velocity and density from the seperate distributions"""
    rho = np.sum(f, axis=1)
    ux = np.einsum("ij,j->i", f, c[:, 0]) / rho
    uy = np.einsum("ij,j->i", f, c[:, 1]) / rho
    return ux.reshape(Nx, Ny), uy.reshape(Nx, Ny), rho.reshape(Nx, Ny)


# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate NN collision operator on Kármán vortex data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-path",
        type=str,
        default="artifacts-run-all-tensorflow/example_network.keras",
        help="Path to the saved Keras model (.keras file)",
    )
    p.add_argument("--data-dir", type=str, default="output", help="Directory containing fpre_*.npy / fpost_*.npy files")
    p.add_argument(
        "--out-dir", type=str, default="eval_results", help="Directory where evaluation plots and CSV are saved"
    )
    p.add_argument("--batch-size", type=int, default=512, help="Batch size for model.predict()")
    p.add_argument(
        "--animate", action="store_true", default=False, help="Produce a GIF of the NN-predicted velocity field"
    )
    p.add_argument("--gif-fps", type=int, default=5, help="Frames per second for the GIF animation")
    p.add_argument("--update-steps", type=int, default=100, help="Update the GIF every N simulation steps")
    p.add_argument(
        "--anim-steps", type=int, default=5000, help="Total number of simulation steps to run for the animation"
    )
    p.add_argument(
        "--preview-path",
        type=str,
        default=None,
        help="If set, save (overwrite) a live preview PNG of the latest frame here during animation",
    )
    p.add_argument("--skip-evaluate", action="store_true", help="Skip snapshot evaluation and all evaluation outputs")
    p.add_argument("--skip-plot", action="store_true", help="Skip time-series metrics plot")
    p.add_argument("--skip-per-direction", action="store_true", help="Skip per-direction error plot")
    p.add_argument(
        "--track-positivity",
        action="store_true",
        help="Record per-step negative-f statistics and save to positivity_stats.csv",
    )
    return p.parse_args()


def render_velocity_frame(ax, ux, uy, obstacle, step, U_max, Nx, Ny, X, Y):
    """Render a single velocity magnitude + streamline frame onto ax."""
    ax.cla()
    speed = np.sqrt(ux**2 + uy**2)
    speed[obstacle] = np.nan
    ax.imshow(speed.T, origin="lower", cmap="jet", vmin=0, vmax=U_max, aspect="auto", extent=(0, Nx, 0, Ny))
    ux_p = ux.copy()
    ux_p[obstacle] = 0.0
    uy_p = uy.copy()
    uy_p[obstacle] = 0.0
    ax.streamplot(X.T, Y.T, ux_p.T, uy_p.T, density=0.5, color="w", linewidth=0.6)
    ax.set_xlim(0, Nx)
    ax.set_ylim(0, Ny)
    ax.set_title(f"NN predicted velocity — step {step}", fontsize=12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────


def load_model(model_path: str) -> Model:
    """Load a saved Keras model with all custom objects for the NN collision operator."""
    log.info("Loading model from: %s", model_path)
    model = cast(
        Model,
        keras.models.load_model(model_path, custom_objects={"rmsre": rmsre}),
    )
    model.summary()
    return model


# ──────────────────────────────────────────────
# Snapshot evaluation
# ──────────────────────────────────────────────


def discover_snapshot_pairs(data_dir: str) -> list[tuple[int, str, str]]:
    """Return sorted (step, fpre_path, fpost_path) tuples for all paired snapshots in data_dir."""
    fpre_files = sorted(glob.glob(os.path.join(data_dir, "fpre_*.npy")))
    pairs = []
    for fpre_path in fpre_files:
        fname = os.path.basename(fpre_path)
        step_str = fname.replace("fpre_", "").replace(".npy", "")
        fpost_path = os.path.join(data_dir, f"fpost_{step_str}.npy")
        if not os.path.exists(fpost_path):
            log.warning("No matching fpost for %s", fname)
            continue
        pairs.append((int(step_str), fpre_path, fpost_path))
    return pairs


def evaluate_snapshot(model: Model, fpre_path: str, fpost_path: str, batch_size: int) -> dict:
    """Evaluate a single fpre/fpost pair.

    Returns a dict with keys: rmsre, mae, max_err, per_dir_mean_err (shape (Q,)).
    """
    # Load and flatten spatial dimensions → (N_cells, 9)
    fpre_raw = np.load(fpre_path)  # shape: (Nx, Ny, 9)
    fpost_raw = np.load(fpost_path)  # shape: (Nx, Ny, 9)

    _Nx, _Ny, Q = fpre_raw.shape

    # Normalize — same as training pipeline
    fpre_norm, _ = normalize(fpre_raw.reshape(-1, Q))  # (Nx*Ny, 9)
    fpost_norm, _ = normalize(fpost_raw.reshape(-1, Q))

    # NN prediction
    fpost_pred = model.predict(fpre_norm, batch_size=batch_size, verbose=cast(str, 0))

    # Per-snapshot metrics
    eps = 1e-15
    rel_err = np.abs((fpost_norm - fpost_pred) / (fpost_norm + eps))

    return {
        "rmsre": float(np.sqrt(np.mean(rel_err**2))),
        "mae": float(np.mean(np.abs(fpost_norm - fpost_pred))),
        "max_err": float(np.max(np.abs(fpost_norm - fpost_pred))),
        "per_dir_mean_err": rel_err.mean(axis=0),  # (Q,)
    }


def run_evaluation(
    model: Model, pairs: list[tuple[int, str, str]], batch_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Evaluate all snapshot pairs and return arrays of per-snapshot metrics.

    Returns:
        steps, rmsre_scores, mae_scores, max_err, per_dir_errors
    """
    steps, rmsre_scores, mae_scores, max_err, per_dir_errors = [], [], [], [], []

    for step, fpre_path, fpost_path in tqdm(pairs):
        result = evaluate_snapshot(model, fpre_path, fpost_path, batch_size)
        steps.append(step)
        rmsre_scores.append(result["rmsre"])
        mae_scores.append(result["mae"])
        max_err.append(result["max_err"])
        per_dir_errors.append(result["per_dir_mean_err"])
        log.info(
            "Step %6d | RMSRE=%.4e  MAE=%.4e  MaxErr=%.4e", step, result["rmsre"], result["mae"], result["max_err"]
        )

    return (
        np.array(steps),
        np.array(rmsre_scores),
        np.array(mae_scores),
        np.array(max_err),
        per_dir_errors,
    )


# ──────────────────────────────────────────────
# Saving and plotting
# ──────────────────────────────────────────────


def save_metrics_csv(steps, rmsre_scores, mae_scores, max_err, out_dir: str) -> None:
    """Save per-snapshot evaluation metrics to CSV."""
    csv_path = os.path.join(out_dir, "eval_metrics.csv")
    np.savetxt(
        csv_path,
        np.column_stack([steps, rmsre_scores, mae_scores, max_err]),
        delimiter=",",
        header="step,rmsre,mae,max_abs_error",
        comments="",
    )
    log.info("Metrics saved to: %s", csv_path)


def plot_metrics(steps, rmsre_scores, mae_scores, max_err, out_dir: str) -> None:
    """Plot RMSRE, MAE, and max absolute error over simulation time."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].semilogy(steps, rmsre_scores, "o-", color="steelblue", lw=1.5, ms=4)
    axes[0].set_ylabel("RMSRE")
    axes[0].set_title("Neural Network Evaluation on Kármán Vortex Data")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(steps, mae_scores, "o-", color="darkorange", lw=1.5, ms=4)
    axes[1].set_ylabel("MAE")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(steps, max_err, "o-", color="crimson", lw=1.5, ms=4)
    axes[2].set_ylabel("Max Abs Error")
    axes[2].set_xlabel("Simulation Step")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "eval_metrics.png")
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved to: %s", plot_path)


def plot_per_direction_errors(
    per_dir_errors: list[np.ndarray],
    rmsre_scores: np.ndarray,
    steps: np.ndarray,
    best_idx: int,
    worst_idx: int,
    out_dir: str,
) -> None:
    """Plot mean relative error per velocity direction for the best and worst snapshots."""
    Q = len(per_dir_errors[0])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, idx, label in zip(axes, [best_idx, worst_idx], ["Best snapshot", "Worst snapshot"]):
        ax.bar(range(Q), per_dir_errors[idx], color="steelblue", edgecolor="k", linewidth=0.5)
        ax.set_xticks(range(Q))
        ax.set_xticklabels([f"f{i}" for i in range(Q)])
        ax.set_ylabel("Mean relative error")
        ax.set_title(f"{label} (step {steps[idx]})\nRMSRE={rmsre_scores[idx]:.4e}")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    dist_path = os.path.join(out_dir, "per_direction_error.png")
    fig.savefig(dist_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Per-direction plot: %s", dist_path)


def print_summary(steps, rmsre_scores, mae_scores, max_err) -> None:
    """Log aggregate evaluation metrics."""
    log.info("── Summary ───────────────────────────────────────")
    log.info("  Snapshots evaluated : %d", len(steps))
    log.info("  RMSRE  — mean: %.4e  min: %.4e  max: %.4e", rmsre_scores.mean(), rmsre_scores.min(), rmsre_scores.max())
    log.info("  MAE    — mean: %.4e  min: %.4e  max: %.4e", mae_scores.mean(), mae_scores.min(), mae_scores.max())
    log.info("  MaxErr — mean: %.4e  min: %.4e  max: %.4e", max_err.mean(), max_err.min(), max_err.max())
    log.info("──────────────────────────────────────────────────")


# ──────────────────────────────────────────────
# Animation
# ──────────────────────────────────────────────


def make_animation(model: Model, args) -> None:
    """
    Run a full LBM simulation using the NN as the collision operator.
    Saves a GIF frame every --update-steps steps.
    Geometry matches lbm_karman-ng.py defaults.
    """

    log.info("Running NN-driven simulation for animation ...")

    # Simulation parameters (match lbm_karman-ng.py defaults)
    res = 250
    Nx = int(round(2.2 * res))
    Ny = int(round(0.41 * res))
    cx_cyl = int(round(0.2 * res))
    cy_cyl = int(round(0.2 * res))
    r_cyl = int(round(0.05 * res))
    U_inlet = 0.12
    Re = 150.0
    D = 2 * r_cyl
    nu = U_inlet * D / Re
    tau = 3.0 * nu + 0.5

    log.info("Grid: %d x %d,  tau=%.4f,  nu=%.6f", Nx, Ny, tau, nu)

    # Obstacle mask
    x = np.arange(Nx)
    y = np.arange(Ny)
    X_grid, Y_grid = np.meshgrid(x, y, indexing="ij")
    obstacle = (X_grid - cx_cyl) ** 2 + (Y_grid - cy_cyl) ** 2 <= r_cyl**2

    # Initial conditions
    rho = np.ones((Nx, Ny))
    ux = np.full((Nx, Ny), U_inlet)
    uy = 0.001 * U_inlet * np.sin(2.0 * np.pi * Y_grid / Ny)
    for arr in (ux, uy):
        arr[obstacle] = 0.0
        arr[:, 0] = 0.0
        arr[:, -1] = 0.0

    f = equilibrium(rho, ux, uy)

    # Direction indices for bounce-back (opposite directions)
    opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    # 1-cell shell around the obstacle (for positivity breakdown)
    obstacle_adj = np.zeros_like(obstacle)
    for shift, ax in [(1, 0), (-1, 0), (1, 1), (-1, 1)]:
        obstacle_adj |= np.roll(obstacle, shift, axis=ax)
    obstacle_adj &= ~obstacle
    interior = ~obstacle & ~obstacle_adj

    positivity_records: list[dict] = []

    frames_ux, frames_uy, frame_steps = [], [], []
    U_max = U_inlet * 2.0
    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")

    for step in tqdm(range(1, args.anim_steps + 1)):
        # Macroscopic quantities
        ux, uy, rho = f_to_velocity(f.reshape(-1, 9), Nx, Ny)

        # NN collision with density normalization
        fpre_norm, norm = normalize(f.reshape(-1, 9))
        f_out = (model.predict(fpre_norm, batch_size=args.batch_size, verbose=cast(str, 0)) * norm).reshape(Nx, Ny, 9)

        if args.track_positivity:
            neg = f_out < 0
            neg_vals = f_out[neg]
            positivity_records.append({
                "step": step,
                "n_neg_total": int(neg.sum()),
                "n_neg_adj": int(neg[obstacle_adj].sum()),
                "n_neg_interior": int(neg[interior].sum()),
                "min_f": float(neg_vals.min()) if neg_vals.size else 0.0,
                "mean_neg_f": float(neg_vals.mean()) if neg_vals.size else 0.0,
            })

        # Bounce-back on obstacle
        for i in range(9):
            f_out[obstacle, i] = f[obstacle, opp[i]]

        # Streaming
        for i in range(9):
            f[:, :, i] = np.roll(f_out[:, :, i], shift=c[i, 0], axis=0)
            f[:, :, i] = np.roll(f[:, :, i], shift=c[i, 1], axis=1)

        # Wall bounce-back
        f[:, 0, 2] = f_out[:, 0, 4]
        f[:, 0, 5] = f_out[:, 0, 7]
        f[:, 0, 6] = f_out[:, 0, 8]
        f[:, -1, 4] = f_out[:, -1, 2]
        f[:, -1, 7] = f_out[:, -1, 5]
        f[:, -1, 8] = f_out[:, -1, 6]

        # Outlet BC (Zou-He pressure)
        rho_out = 1.0
        iy = slice(1, -1)
        ux_out = np.clip(
            -1.0
            + (f[-1, iy, 0] + f[-1, iy, 2] + f[-1, iy, 4] + 2.0 * (f[-1, iy, 1] + f[-1, iy, 5] + f[-1, iy, 8]))
            / rho_out,
            0.0,
            0.5,
        )
        f[-1, iy, 3] = f[-1, iy, 1] - (2.0 / 3.0) * rho_out * ux_out
        f[-1, iy, 7] = f[-1, iy, 5] + 0.5 * (f[-1, iy, 2] - f[-1, iy, 4]) - (1.0 / 6.0) * rho_out * ux_out
        f[-1, iy, 6] = f[-1, iy, 8] - 0.5 * (f[-1, iy, 2] - f[-1, iy, 4]) - (1.0 / 6.0) * rho_out * ux_out
        for yc in [0, Ny - 1]:
            f[-1, yc, 3] = f[-2, yc, 3]
            f[-1, yc, 6] = f[-2, yc, 6]
            f[-1, yc, 7] = f[-2, yc, 7]

        # Inlet BC (Zou-He velocity)
        rho_in = (f[0, :, 0] + f[0, :, 2] + f[0, :, 4] + 2.0 * (f[0, :, 3] + f[0, :, 6] + f[0, :, 7])) / (1.0 - U_inlet)
        f[0, :, 1] = f[0, :, 3] + (2.0 / 3.0) * rho_in * U_inlet
        f[0, :, 5] = f[0, :, 7] - 0.5 * (f[0, :, 2] - f[0, :, 4]) + (1.0 / 6.0) * rho_in * U_inlet
        f[0, :, 8] = f[0, :, 6] + 0.5 * (f[0, :, 2] - f[0, :, 4]) + (1.0 / 6.0) * rho_in * U_inlet

        # Save frames for GIF every N steps
        if step % args.update_steps == 0:
            frames_ux.append(ux.copy())
            frames_uy.append(uy.copy())
            frame_steps.append(step)
            log.info("Frame appended at step %d/%d", step, args.anim_steps)

            if args.preview_path:
                fig_p, ax_p = plt.subplots(figsize=(10, 4), dpi=100)
                render_velocity_frame(ax_p, ux, uy, obstacle, f"{step}/{args.anim_steps}", U_max, Nx, Ny, X, Y)
                fig_p.tight_layout()
                fig_p.savefig(args.preview_path)
                plt.close(fig_p)

    if args.track_positivity and positivity_records:
        csv_path = os.path.join(args.out_dir, "positivity_stats.csv")
        keys = list(positivity_records[0].keys())
        rows = np.array([[r[k] for k in keys] for r in positivity_records])
        np.savetxt(csv_path, rows, delimiter=",", header=",".join(keys), comments="")
        total_neg = int(rows[:, 1].sum())
        first_nonzero = next((r["step"] for r in positivity_records if r["n_neg_total"] > 0), None)
        log.info("Positivity stats saved to: %s", csv_path)
        log.info("  Total negative-f events : %d", total_neg)
        log.info("  First step with neg-f   : %s", first_nonzero if first_nonzero else "none")

    # Build and save GIF
    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

    def update(i: int) -> list[Artist]:
        render_velocity_frame(ax, frames_ux[i], frames_uy[i], obstacle, frame_steps[i], U_max, Nx, Ny, X, Y)
        return []

    anim = FuncAnimation(fig, update, frames=len(frames_ux), interval=1000 // args.gif_fps)
    gif_path = os.path.join(args.out_dir, "nn_velocity_field.gif")
    anim.save(gif_path, writer=PillowWriter(fps=args.gif_fps))
    plt.close(fig)
    log.info("GIF saved to: %s", gif_path)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Load model
    K.set_floatx("float64")
    model = load_model(args.model_path)

    if not args.skip_evaluate:
        pairs = discover_snapshot_pairs(args.data_dir)
        if not pairs:
            if not args.animate:
                raise FileNotFoundError(
                    f"No fpre_*.npy files found in '{args.data_dir}'.\n"
                    "Run lbm_karman-ng.py with --save-every > 0 first."
                )
            log.info("No fpre/fpost pairs found in '%s'; skipping evaluation.", args.data_dir)
        else:
            log.info("Found %d snapshot pair(s) in '%s'", len(pairs), args.data_dir)
            steps, rmsre_scores, mae_scores, max_err, per_dir_errors = run_evaluation(model, pairs, args.batch_size)
            save_metrics_csv(steps, rmsre_scores, mae_scores, max_err, args.out_dir)
            if not args.skip_plot:
                plot_metrics(steps, rmsre_scores, mae_scores, max_err, args.out_dir)
            if not args.skip_per_direction:
                best_idx = int(np.argmin(rmsre_scores))
                worst_idx = int(np.argmax(rmsre_scores))
                plot_per_direction_errors(per_dir_errors, rmsre_scores, steps, best_idx, worst_idx, args.out_dir)
            print_summary(steps, rmsre_scores, mae_scores, max_err)

    if args.animate:
        make_animation(model, args)


if __name__ == "__main__":
    main()
