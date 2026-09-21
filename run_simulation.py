"""Run the simulation settings reported in the manuscript and supplement."""

import os

for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(name, "1")
import argparse
from pathlib import Path
import pandas as pd
from pabl.simulation_continuous import run_one_rep
from pabl.simulation_survival import run_one_rep_multi_t, run_goldberg_oracle_xu_multi_t

BASE_SEED = 20260511
HORIZONS = [0.75, 1.0, 1.25]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outcome", choices=["all", "continuous", "survival"], default="all"
    )
    parser.add_argument(
        "--n", type=int, choices=[2000, 5000], nargs="+", default=[2000, 5000]
    )
    parser.add_argument("--rep-start", type=int, default=0)
    parser.add_argument("--rep-stop", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/simulation"))
    args = parser.parse_args()
    if not 0 <= args.rep_start < args.rep_stop <= 100:
        parser.error("Replications must form a nonempty interval within [0, 100).")
    args.output.mkdir(parents=True, exist_ok=True)
    families = ["continuous", "survival"] if args.outcome == "all" else [args.outcome]
    for family in families:
        for linearity in (
            ["linear", "nonlinear"] if family == "continuous" else ["linear"]
        ):
            for u_inter in (False, True):
                for n in args.n:
                    name = f"{family}_{linearity}_U{int(u_inter)}_{n}"
                    destination = args.output / name
                    destination.mkdir(exist_ok=True)
                    for rep in range(args.rep_start, args.rep_stop):
                        path = destination / f"rep_{rep:03d}.csv"
                        if path.exists():
                            continue
                        kwargs = dict(
                            rep=rep,
                            train_seed=BASE_SEED + 10000 * rep + 42,
                            test_seed=BASE_SEED + 10000 * rep + 999,
                            n_train=n,
                            n_test=500000,
                            linearity=linearity,
                            U_inter=u_inter,
                            device=args.device,
                        )
                        if family == "continuous":
                            rows = run_one_rep(**kwargs)
                        else:
                            rows = run_one_rep_multi_t(**kwargs, t_grid=HORIZONS)
                            rows += run_goldberg_oracle_xu_multi_t(
                                **kwargs, t_grid=HORIZONS, survival_dgp="weibull"
                            )
                        frame = pd.DataFrame(rows)
                        frame["status"] = "ok"
                        temporary = path.with_suffix(".tmp")
                        frame.to_csv(temporary, index=False)
                        temporary.replace(path)
                    print("Saved " + str(destination))


if __name__ == "__main__":
    main()
