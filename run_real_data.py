"""Reproduce the application tables and recommendation-agreement figure."""

import argparse
import json
from pathlib import Path
from real_data.analysis import prepare_cohort, analyze
from real_data.survival_nuisance import _load_survival_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        default="mimiciv31",
        help="Authorized local PostgreSQL MIMIC-IV 3.1 database",
    )
    parser.add_argument(
        "--outcome", choices=["all", "daoh28", "survival90", "censored"], default="all"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("results/real_data"))
    args = parser.parse_args()
    sql = Path(__file__).resolve().parent / "real_data/cohort.sql"
    frame = _load_survival_features(args.database, sql)
    (selected, treatment, folds) = prepare_cohort(frame)
    args.output.mkdir(parents=True, exist_ok=True)
    for outcome in (
        ["daoh28", "survival90", "censored"]
        if args.outcome == "all"
        else [args.outcome]
    ):
        (table, agreement) = analyze(selected, treatment, folds, outcome, args.device)
        table.to_csv(args.output / (outcome + ".csv"), index=False)
        (args.output / (outcome + "_agreement.json")).write_text(
            json.dumps(agreement, indent=2) + "\n"
        )
        print("Saved " + str(args.output / (outcome + ".csv")))


if __name__ == "__main__":
    main()
