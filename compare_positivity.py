"""Compare positivity_stats.csv from multiple model runs.

Usage examples:
  python compare_positivity.py                                    # auto-scan artifacts/
  python compare_positivity.py artifacts/model_a artifacts/model_b
  python compare_positivity.py --artifacts-dir my_runs --output out.png
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

STATS_RELPATH = "eval_results/positivity_stats.csv"


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "models",
        nargs="*",
        help=(
            "Artifact directories to compare "
            "(e.g. artifacts/lenn_18_18_18_softmax_1000). "
            f"If omitted, all sub-directories of --artifacts-dir that contain "
            f"'{STATS_RELPATH}' are used."
        ),
    )
    p.add_argument(
        "--artifacts-dir",
        default="artifacts",
        help="Base directory to scan when no model directories are given.",
    )
    p.add_argument(
        "--output",
        default="positivity_comparison.png",
        help="Path for the output PNG file.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress INFO logging.",
    )
    return p.parse_args()


def _setup_logging(verbose: bool = True) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
        force=True,
    )


# ── Data loading ──────────────────────────────────────────────────────────────

def _resolve_paths(models: list[str], artifacts_dir: str) -> dict[str, Path]:
    """Return {label: csv_path} for each model to compare."""
    if models:
        dirs = [Path(m) for m in models]
    else:
        base = Path(artifacts_dir)
        if not base.is_dir():
            raise FileNotFoundError(f"Artifacts directory not found: '{base}'")
        dirs = sorted(d for d in base.iterdir() if d.is_dir())
        logger.info("Scanning '%s' for model runs …", base)

    paths: dict[str, Path] = {}
    for d in dirs:
        csv = d / STATS_RELPATH
        if csv.is_file():
            paths[d.name] = csv
        else:
            logger.warning("No %s in '%s' — skipping.", STATS_RELPATH, d)

    if not paths:
        raise FileNotFoundError(
            "No positivity_stats.csv files found. "
            "Run the NN evaluation with positivity logging enabled first."
        )
    return paths


def _load(path: Path) -> np.ndarray:
    return np.genfromtxt(path, delimiter=",", names=True)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _plot(data: dict[str, np.ndarray], output: Path) -> None:
    """Plot positivity violation statistics for all models and save to disk."""
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for i, (label, d) in enumerate(data.items()):
        color = colors[i % len(colors)]
        axes[0].plot(d["step"], d["n_neg_total"], label=label, color=color, lw=1.2)
        axes[1].plot(d["step"], d["n_neg_adj"], label=label, color=color, lw=1.2, linestyle="--")
        axes[1].plot(d["step"], d["n_neg_interior"], label=f"{label} (interior)", color=color, lw=1.2, linestyle=":")
        axes[2].plot(d["step"], np.abs(d["min_f"]), label=label, color=color, lw=1.2)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("# negative f (total)")
    axes[0].set_title("Positivity violation comparison")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, which="both")

    axes[1].set_yscale("log")
    axes[1].set_ylabel("# negative f")
    axes[1].set_title("Adjacent to obstacle (--) vs interior (:)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which="both")

    axes[2].set_yscale("log")
    axes[2].set_ylabel("|min f|")
    axes[2].set_title("Most negative distribution value per step (absolute)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, which="both")
    axes[2].set_xlabel("Simulation step")

    fig.tight_layout()
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output)


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(data: dict[str, np.ndarray]) -> None:
    """Log first violation step and total negative-f events per model."""
    logger.info("── Summary ──────────────────────────────────────────────────────")
    for label, d in data.items():
        mask = d["n_neg_total"] > 0
        first = int(d["step"][mask][0]) if mask.any() else None
        total = int(d["n_neg_total"].sum())
        logger.info("  %-52s  first neg-f: step %-6s  total: %d", label, first, total)
    logger.info("─────────────────────────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    _setup_logging(verbose=not args.quiet)

    paths = _resolve_paths(args.models, args.artifacts_dir)
    logger.info("Comparing %d model run(s):", len(paths))
    for label, path in paths.items():
        logger.info("  %s  →  %s", label, path)

    data = {label: _load(path) for label, path in paths.items()}
    _plot(data, Path(args.output))
    _print_summary(data)


if __name__ == "__main__":
    main()
