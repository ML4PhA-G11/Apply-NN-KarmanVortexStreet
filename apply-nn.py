"""
evaluate_nn.py
==============
Evaluates the trained neural network collision operator against
f_pre / f_post pairs saved from the Kármán vortex street simulation.

Usage:
    python evaluate_nn.py --model-path artifacts-run-all-tensorflow/example_network.keras \
                          --data-dir output \
                          --out-dir eval_results

The script expects pairs of files named:
    fpre_XXXXXX.npy   (pre-collision distribution)
    fpost_XXXXXX.npy  (post-collision distribution, BGK ground truth)
"""

import os
import glob
import argparse
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import tensorflow as tf
import keras
from keras import backend as K
from numba import njit
from utils import D4Symmetry, AlgReconstruction, D4AntiSymmetry

c = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]])

w = np.array([4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36])


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


def rmsre(y_true, y_pred):
    """Root Mean Squared Relative Error — same loss used during training."""
    return keras.backend.sqrt(
        keras.backend.mean(keras.backend.square((y_true - y_pred) / (y_true + keras.backend.epsilon())))
    )


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
    p.add_argument("--update-steps", type=int, default=25, help="Update the GIF every N simulation steps")
    p.add_argument(
        "--anim-steps", type=int, default=5000, help="Total number of simulation steps to run for the animation"
    )
    return p.parse_args()


# Animation
def make_animation(model, args):
    """
    Runs a full LBM simulation using the NN as the collision operator.
    Saves a GIF frame every --update-steps steps.
    Geometry matches lbm_karman-ng.py defaults.
    """
    print("\nRunning NN-driven simulation for animation ...")

    # -- Simulation parameters (match lbm_karman-ng.py defaults) --
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

    print(f"  Grid: {Nx} x {Ny},  tau={tau:.4f},  nu={nu:.6f}")

    # -- Obstacle mask --
    x = np.arange(Nx)
    y = np.arange(Ny)
    X_grid, Y_grid = np.meshgrid(x, y, indexing="ij")
    obstacle = (X_grid - cx_cyl) ** 2 + (Y_grid - cy_cyl) ** 2 <= r_cyl**2

    # -- Initial conditions --
    rho = np.ones((Nx, Ny))
    ux = np.full((Nx, Ny), U_inlet)
    uy = np.zeros((Nx, Ny))
    uy += 0.001 * U_inlet * np.sin(2.0 * np.pi * Y_grid / Ny)
    ux[obstacle] = 0.0
    uy[obstacle] = 0.0
    ux[:, 0] = 0.0
    uy[:, 0] = 0.0
    ux[:, -1] = 0.0
    uy[:, -1] = 0.0

    f = equilibrium(rho, ux, uy)

    opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

    frames_ux = []
    frames_uy = []
    frame_steps = []

    n_steps = args.anim_steps
    update_every = args.update_steps

    for step in range(1, n_steps + 1):

        # -- Macroscopic quantities --
        rho = np.sum(f, axis=2)
        ux = np.sum(f * c[:, 0], axis=2) / rho
        uy = np.sum(f * c[:, 1], axis=2) / rho

        # -- NN collision --
        fpre = f.reshape(-1, 9)
        norm = np.sum(fpre, axis=1, keepdims=True)
        fpre_norm = fpre / norm
        f_out_norm = model.predict(fpre_norm, batch_size=args.batch_size, verbose=0)
        f_out = (f_out_norm * norm).reshape(Nx, Ny, 9)

        # -- Bounce-back on obstacle --
        for i in range(9):
            f_out[obstacle, i] = f[obstacle, opp[i]]

        # -- Streaming --
        for i in range(9):
            f[:, :, i] = np.roll(f_out[:, :, i], shift=c[i, 0], axis=0)
            f[:, :, i] = np.roll(f[:, :, i], shift=c[i, 1], axis=1)

        # -- Wall bounce-back --
        f[:, 0, 2] = f_out[:, 0, 4]
        f[:, 0, 5] = f_out[:, 0, 7]
        f[:, 0, 6] = f_out[:, 0, 8]
        f[:, -1, 4] = f_out[:, -1, 2]
        f[:, -1, 7] = f_out[:, -1, 5]
        f[:, -1, 8] = f_out[:, -1, 6]

        # -- Outlet BC (Zou-He pressure) --
        rho_out = 1.0
        iy = slice(1, -1)
        ux_out = (
            -1.0
            + (f[-1, iy, 0] + f[-1, iy, 2] + f[-1, iy, 4] + 2.0 * (f[-1, iy, 1] + f[-1, iy, 5] + f[-1, iy, 8]))
            / rho_out
        )
        ux_out = np.clip(ux_out, 0.0, 0.5)
        f[-1, iy, 3] = f[-1, iy, 1] - (2.0 / 3.0) * rho_out * ux_out
        f[-1, iy, 7] = f[-1, iy, 5] + 0.5 * (f[-1, iy, 2] - f[-1, iy, 4]) - (1.0 / 6.0) * rho_out * ux_out
        f[-1, iy, 6] = f[-1, iy, 8] - 0.5 * (f[-1, iy, 2] - f[-1, iy, 4]) - (1.0 / 6.0) * rho_out * ux_out
        for yc in [0, Ny - 1]:
            f[-1, yc, 3] = f[-2, yc, 3]
            f[-1, yc, 6] = f[-2, yc, 6]
            f[-1, yc, 7] = f[-2, yc, 7]

        # -- Inlet BC (Zou-He velocity) --
        rho_in = (f[0, :, 0] + f[0, :, 2] + f[0, :, 4] + 2.0 * (f[0, :, 3] + f[0, :, 6] + f[0, :, 7])) / (1.0 - U_inlet)
        f[0, :, 1] = f[0, :, 3] + (2.0 / 3.0) * rho_in * U_inlet
        f[0, :, 5] = f[0, :, 7] - 0.5 * (f[0, :, 2] - f[0, :, 4]) + (1.0 / 6.0) * rho_in * U_inlet
        f[0, :, 8] = f[0, :, 6] + 0.5 * (f[0, :, 2] - f[0, :, 4]) + (1.0 / 6.0) * rho_in * U_inlet

        # -- Collect frame --
        if step % update_every == 0:
            frames_ux.append(ux.copy())
            frames_uy.append(uy.copy())
            frame_steps.append(step)
            print(f"  Frame saved at step {step}/{n_steps}")

    # -- Build and save GIF --
    U_max = U_inlet * 2.0
    X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing="ij")

    fig, ax = plt.subplots(figsize=(10, 4), dpi=100)

    def update(i):
        ax.cla()
        speed = np.sqrt(frames_ux[i] ** 2 + frames_uy[i] ** 2)
        speed[obstacle] = np.nan
        ax.imshow(speed.T, origin="lower", cmap="jet", vmin=0, vmax=U_max, aspect="auto", extent=[0, Nx, 0, Ny])
        ux_i = frames_ux[i].copy()
        ux_i[obstacle] = 0.0
        uy_i = frames_uy[i].copy()
        uy_i[obstacle] = 0.0
        ax.streamplot(X.T, Y.T, ux_i.T, uy_i.T, density=0.5, color="w", linewidth=0.6)
        ax.set_title(f"NN predicted velocity — step {frame_steps[i]}", fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    anim = FuncAnimation(fig, update, frames=len(frames_ux), interval=1000 // args.gif_fps)
    gif_path = os.path.join(args.out_dir, "nn_velocity_field.gif")
    anim.save(gif_path, writer=PillowWriter(fps=args.gif_fps))
    plt.close(fig)
    print(f"  GIF saved to: {gif_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 1. Load model ──────────────────────────────────────────────────────
    K.set_floatx("float64")
    print(f"Loading model from: {args.model_path}")
    model = keras.models.load_model(
        args.model_path,
        custom_objects={
            "rmsre": rmsre,
            "D4Symmetry": D4Symmetry,
            "AlgReconstruction": AlgReconstruction,
            "D4AntiSymmetry": D4AntiSymmetry,
        },
    )
    model.summary()

    # ── 2. Discover data files ─────────────────────────────────────────────
    fpre_files = sorted(glob.glob(os.path.join(args.data_dir, "fpre_*.npy")))

    if len(fpre_files) == 0:
        # The animation runs a self-contained NN-driven LBM simulation and does
        # not need the recorded fpre/fpost snapshots. So when there is no eval
        # data, only fail if the user actually asked for evaluation.
        if args.animate:
            print(
                f"\nNo fpre_*.npy files found in '{args.data_dir}'; " "skipping evaluation and running animation only."
            )
            make_animation(model, args)
            return
        raise FileNotFoundError(
            f"No fpre_*.npy files found in '{args.data_dir}'.\n" "Run lbm_karman-ng.py with --save-every > 0 first."
        )

    print(f"\nFound {len(fpre_files)} snapshot pair(s) in '{args.data_dir}'")

    # ── 3. Evaluate each snapshot ──────────────────────────────────────────
    steps = []
    rmsre_scores = []
    mae_scores = []
    max_err = []

    for fpre_path in fpre_files:
        # Derive matching fpost path
        fname = os.path.basename(fpre_path)
        step_str = fname.replace("fpre_", "").replace(".npy", "")
        fpost_path = os.path.join(args.data_dir, f"fpost_{step_str}.npy")

        if not os.path.exists(fpost_path):
            print(f"  [SKIP] No matching fpost for {fname}")
            continue

        step = int(step_str)
        steps.append(step)

        # Load and flatten spatial dimensions → (N_cells, 9)
        fpre_raw = np.load(fpre_path)  # shape: (Nx, Ny, 9)
        fpost_raw = np.load(fpost_path)  # shape: (Nx, Ny, 9)

        Nx, Ny, Q = fpre_raw.shape
        fpre_flat = fpre_raw.reshape(-1, Q)  # (Nx*Ny, 9)
        fpost_flat = fpost_raw.reshape(-1, Q)

        # Normalize — same as training pipeline
        fpre_norm, norm = normalize(fpre_flat)
        fpost_norm, _ = normalize(fpost_flat)

        # NN prediction
        fpost_pred_norm = model.predict(fpre_norm, batch_size=args.batch_size, verbose=0)

        # ── Per-snapshot metrics ───────────────────────────────────────────
        eps = 1e-15

        # RMSRE
        rel_sq = ((fpost_norm - fpost_pred_norm) / (fpost_norm + eps)) ** 2
        rmsre_val = np.sqrt(np.mean(rel_sq))

        # MAE
        mae_val = np.mean(np.abs(fpost_norm - fpost_pred_norm))

        # Max absolute error
        max_val = np.max(np.abs(fpost_norm - fpost_pred_norm))

        rmsre_scores.append(rmsre_val)
        mae_scores.append(mae_val)
        max_err.append(max_val)

        print(f"  Step {step:>6d} | RMSRE={rmsre_val:.4e}  MAE={mae_val:.4e}  MaxErr={max_val:.4e}")

    steps = np.array(steps)
    rmsre_scores = np.array(rmsre_scores)
    mae_scores = np.array(mae_scores)
    max_err = np.array(max_err)

    # ── 4. Save metrics to CSV ─────────────────────────────────────────────
    csv_path = os.path.join(args.out_dir, "eval_metrics.csv")
    header = "step,rmsre,mae,max_abs_error"
    data = np.column_stack([steps, rmsre_scores, mae_scores, max_err])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    print(f"\nMetrics saved to: {csv_path}")

    # ── 5. Plot metrics over simulation time ───────────────────────────────
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
    plot_path = os.path.join(args.out_dir, "eval_metrics.png")
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to:    {plot_path}")

    # ── 6. Distribution plot — best vs worst snapshot ─────────────────────
    # Reload best and worst snapshots for a per-direction error breakdown
    best_idx = np.argmin(rmsre_scores)
    worst_idx = np.argmax(rmsre_scores)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, idx, label in zip(axes, [best_idx, worst_idx], ["Best snapshot", "Worst snapshot"]):
        step_str = f"{steps[idx]:06d}"
        fpre_raw = np.load(os.path.join(args.data_dir, f"fpre_{step_str}.npy"))
        fpost_raw = np.load(os.path.join(args.data_dir, f"fpost_{step_str}.npy"))

        Nx, Ny, Q = fpre_raw.shape
        fpre_norm, _ = normalize(fpre_raw.reshape(-1, Q))
        fpost_norm, _ = normalize(fpost_raw.reshape(-1, Q))

        fpost_pred = model.predict(fpre_norm, batch_size=args.batch_size, verbose=0)

        eps = 1e-15
        rel_err = np.abs((fpost_norm - fpost_pred) / (fpost_norm + eps))

        # Mean relative error per velocity direction
        mean_per_dir = rel_err.mean(axis=0)
        ax.bar(range(Q), mean_per_dir, color="steelblue", edgecolor="k", linewidth=0.5)
        ax.set_xticks(range(Q))
        ax.set_xticklabels([f"f{i}" for i in range(Q)])
        ax.set_ylabel("Mean relative error")
        ax.set_title(f"{label} (step {steps[idx]})\nRMSRE={rmsre_scores[idx]:.4e}")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    dist_path = os.path.join(args.out_dir, "per_direction_error.png")
    fig.savefig(dist_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Per-direction plot: {dist_path}")

    # ── 7. Summary ─────────────────────────────────────────────────────────
    print("\n── Summary ───────────────────────────────────────")
    print(f"  Snapshots evaluated : {len(steps)}")
    print(
        f"  RMSRE  — mean: {rmsre_scores.mean():.4e}  " f"min: {rmsre_scores.min():.4e}  max: {rmsre_scores.max():.4e}"
    )
    print(f"  MAE    — mean: {mae_scores.mean():.4e}  " f"min: {mae_scores.min():.4e}  max: {mae_scores.max():.4e}")
    print(f"  MaxErr — mean: {max_err.mean():.4e}  " f"min: {max_err.min():.4e}  max: {max_err.max():.4e}")
    print("──────────────────────────────────────────────────\n")

    if args.animate:
        make_animation(model, args)


if __name__ == "__main__":
    main()
