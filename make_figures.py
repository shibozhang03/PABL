"""Create the manuscript's simulation panels and application agreement figure."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pabl.plotting import plot_continuous, plot_survival, plot_agreement


def simulation_figures(root, output):
    count = 0
    for family in ("continuous", "survival"):
        for linearity in (
            ["linear", "nonlinear"] if family == "continuous" else ["linear"]
        ):
            for u in (0, 1):
                for n in (2000, 5000):
                    name = f"{family}_{linearity}_U{u}_{n}"
                    files = [root / name / f"rep_{rep:03d}.csv" for rep in range(100)]
                    missing = [str(path) for path in files if not path.is_file()]
                    if missing:
                        raise FileNotFoundError(
                            "Incomplete simulation setting: " + name
                        )
                    frame = pd.concat(
                        [pd.read_csv(path) for path in files], ignore_index=True
                    )
                    keys = ["rep", "method_full"] + (
                        ["t0"] if family == "survival" else []
                    )
                    expected = 2700 if family == "survival" else 900
                    if (
                        len(frame) != expected
                        or frame.duplicated(keys).any()
                        or set(frame.rep) != set(range(100))
                        or (frame.method_full.nunique() != 9)
                        or (not (frame.n_test == 500000).all())
                        or (not (frame.n_train == n).all())
                        or (not (frame.status == "ok").all())
                        or (not np.isfinite(frame.true_val).all())
                    ):
                        raise ValueError("Invalid or incomplete results: " + name)
                    if family == "survival" and set(frame.t0) != {0.75, 1.0, 1.25}:
                        raise ValueError("Unexpected survival horizons: " + name)
                    destination = output / name
                    destination.mkdir(parents=True, exist_ok=True)
                    combined = destination / "values.csv"
                    frame.sort_values(keys).to_csv(combined, index=False)
                    plot = plot_continuous if family == "continuous" else plot_survival
                    count += len(plot(str(combined), str(destination)))
    return count


def application_figure(root, output):
    names = ["PABL-h(X)", "DR-PABL(X)", "PABL-q(X)"]
    titles = ["DAOH28", "90-day survival: complete", "90-day survival: 28.24% censored"]
    panels = []
    for outcome, title in zip(("daoh28", "survival90", "censored"), titles):
        values = json.loads((root / (outcome + "_agreement.json")).read_text())
        matrix = np.full((3, 3), 100.0)
        for i in range(3):
            for j in range(i + 1, 3):
                matrix[i, j] = matrix[j, i] = values[names[i] + " | " + names[j]]
        panels.append((title, matrix))
    output.mkdir(parents=True, exist_ok=True)
    plot_agreement(panels, output / "blood_culture_recommendation_agreement.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part", choices=["all", "simulation", "real-data"], default="all"
    )
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    if args.part in ("all", "simulation"):
        count = simulation_figures(args.results / "simulation", args.output)
        print(f"Saved {count} simulation panels to {args.output}")
    if args.part in ("all", "real-data"):
        application_figure(args.results / "real_data", args.output)
        print(
            "Saved " + str(args.output / "blood_culture_recommendation_agreement.pdf")
        )


if __name__ == "__main__":
    main()
