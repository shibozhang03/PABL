from pathlib import Path
from typing import List
import pandas as pd


def plot_continuous(per_run_csv: str, output_dir: str) -> List[str]:
    """Create Qi-style continuous boxplots with fixed method order and colors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    method_specs = [
        ("PABL-h(Z,X)", ["Proximal(Z, X)"], "#1f77b4"),
        ("PABL-q(W,X)", ["Proximal(X, W)"], "#c77dbb"),
        ("DR-PABL(X)", ["DR Proximal(X)"], "#9b9b9b"),
        ("PABL-h(X)", ["Proximal-h(X)"], "#1b9e77"),
        ("PABL-q(X)", ["Proximal-q(X)"], "#56b1df"),
        ("Group-AD(X)", ["Qi AD(X)", "Qi AD (X)", "Qi-AD(X)", "Naive(X)"], "#f2a6d7"),
        ("Angle-IPW(X)", ["Xue(X)"], "#d9d242"),
        ("l1-PLS(X)", ["l1-PLS(X)", "L1-PLS(X)", "l1_pls_baseline(X)"], "#f28e2b"),
        ("NUC-AD", ["Oracle(X, U)", "Oracle(X,U)", "Oracle(X)"], "#f7f5ef"),
    ]
    df = pd.read_csv(per_run_csv)
    df_ok = (
        df[df.get("status", "ok") == "ok"].copy()
        if "status" in df.columns
        else df.copy()
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for metric in ["true_val"]:
        if metric not in df_ok.columns:
            continue
        sub = df_ok.dropna(subset=[metric])
        if sub.empty:
            continue
        selected = []
        for label, candidates, color in method_specs:
            present = [m for m in candidates if m in set(sub["method_full"])]
            if not present:
                continue
            method = present[0]
            vals = sub.loc[sub["method_full"] == method, metric].dropna().values
            if len(vals) == 0:
                continue
            selected.append((label, method, color, vals))
        if not selected:
            continue
        colors = [x[2] for x in selected]
        data = [x[3] for x in selected]
        legend_rows = (len(selected) + 2) // 3
        fig_height = 2.43 + 0.24 * legend_rows
        (fig, ax) = plt.subplots(figsize=(5.0, fig_height))
        bp = ax.boxplot(
            data,
            labels=[""] * len(selected),
            showmeans=True,
            meanprops={
                "marker": "D",
                "markersize": 4.7,
                "markerfacecolor": "white",
                "markeredgecolor": "#222222",
                "markeredgewidth": 1.0,
                "zorder": 4,
            },
            patch_artist=True,
            widths=0.65,
            medianprops={"color": "#333333", "linewidth": 1.6},
            whiskerprops={"color": "#333333", "linewidth": 1.2},
            capprops={"color": "#333333", "linewidth": 1.2},
            boxprops={"edgecolor": "#333333", "linewidth": 1.2},
            flierprops={
                "marker": "o",
                "markersize": 3.0,
                "markerfacecolor": "#333333",
                "markeredgecolor": "#333333",
                "alpha": 0.75,
            },
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.95)
        ax.legend(
            handles=[
                Line2D(
                    [],
                    [],
                    linestyle="none",
                    marker="D",
                    markersize=4.7,
                    markerfacecolor="white",
                    markeredgecolor="#222222",
                    label="Mean",
                ),
                Line2D([], [], color="#333333", linewidth=1.6, label="Median"),
            ],
            loc="best",
            ncol=2,
            frameon=False,
            fontsize=8.5,
            handlelength=1.2,
            handletextpad=0.4,
            columnspacing=1.0,
        )
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax.tick_params(axis="y", which="both", left=True, labelleft=True, labelsize=12)
        ax.grid(axis="y", alpha=0.35, linewidth=0.8)
        ax.set_axisbelow(True)
        handles = [
            Patch(facecolor=color, edgecolor="#333333", label=label)
            for (label, _, color, _) in selected
        ]
        legend_handles = [
            handles[row * 3 + column]
            for column in range(3)
            for row in range(legend_rows)
            if row * 3 + column < len(handles)
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=3,
            frameon=False,
            fontsize=12,
            handlelength=1.1,
            handletextpad=0.4,
            columnspacing=0.8,
            labelspacing=0.3,
            borderaxespad=0,
        )
        fig.subplots_adjust(
            left=0.15,
            right=0.99,
            bottom=(0.24 * legend_rows + 0.15) / fig_height,
            top=0.95,
        )
        path = out_dir / f"boxplot_{metric}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_survival(per_run_csv: str, output_dir: str) -> List[str]:
    """Create fixed-order survival boxplots, saving one image per t0."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    method_specs = [
        ("PABL-h(Z,X)", ["Proximal(Z, X)"], "#1f77b4"),
        ("PABL-q(W,X)", ["Proximal(X, W)", "Proximal (X, W)"], "#c77dbb"),
        ("DR-PABL(X)", ["DR Proximal(X)", "Doubly Robust Proximal(X)"], "#9b9b9b"),
        ("PABL-h(X)", ["Proximal-h(X)"], "#1b9e77"),
        ("PABL-q(X)", ["Proximal-q(X)", "Proximal (X)"], "#56b1df"),
        ("Surv-AD(X)", ["Qi AD (X)", "Qi-AD(X)"], "#f2a6d7"),
        ("Angle-KM(X)", ["Xue Angle (X)", "Xue(X)"], "#d9d242"),
        ("Cens-Q(X)", ["Goldberg Q (X)", "Goldberg(X)"], "#f28e2b"),
        (
            "NUC-Cens-Q",
            [
                "Goldberg Q (X,U)",
                "Goldberg Q (X, U)",
                "Goldberg(X,U)",
                "Goldberg(X, U)",
            ],
            "#f7f5ef",
        ),
    ]
    display_by_method = {
        method_name: display_label
        for (display_label, method_names, _) in method_specs
        for method_name in method_names
    }
    color_by_display = {
        display_label: color for (display_label, _, color) in method_specs
    }
    ordered_displays = [display_label for (display_label, _, _) in method_specs]

    def t_suffix(t0):
        if t0 is None:
            return ""
        return ("_t" + f"{float(t0):g}").replace("-", "m").replace(".", "p")

    df = pd.read_csv(per_run_csv)
    df_ok = (
        df[df.get("status", "ok") == "ok"].copy()
        if "status" in df.columns
        else df.copy()
    )
    if "method_full" not in df_ok.columns:
        raise ValueError("per_run_csv must contain a 'method_full' column")
    df_ok["method_display"] = df_ok["method_full"].map(display_by_method)
    df_ok = df_ok.dropna(subset=["method_display"])
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    if "t0" in df_ok.columns and (not df_ok["t0"].dropna().empty):
        t_values = sorted(df_ok["t0"].dropna().unique())
    else:
        t_values = [None]
    for metric in ["true_val"]:
        if metric not in df_ok.columns:
            continue
        sub_all = df_ok.dropna(subset=[metric])
        if sub_all.empty:
            continue
        for t0 in t_values:
            sub = sub_all if t0 is None else sub_all[sub_all["t0"] == t0]
            if sub.empty:
                continue
            present_order = [
                label
                for label in ordered_displays
                if (sub["method_display"] == label).any()
            ]
            if not present_order:
                continue
            data = [
                sub.loc[sub["method_display"] == label, metric].values
                for label in present_order
            ]
            legend_rows = (len(present_order) + 2) // 3
            fig_height = 2.43 + 0.24 * legend_rows
            (fig, ax) = plt.subplots(figsize=(5.0, fig_height))
            box = ax.boxplot(
                data,
                labels=[""] * len(present_order),
                showmeans=True,
                meanprops={
                    "marker": "D",
                    "markersize": 4.7,
                    "markerfacecolor": "white",
                    "markeredgecolor": "#222222",
                    "markeredgewidth": 1.0,
                    "zorder": 4,
                },
                patch_artist=True,
                widths=0.65,
                medianprops={"color": "#333333", "linewidth": 1.5},
                whiskerprops={"color": "#333333", "linewidth": 1.1},
                capprops={"color": "#333333", "linewidth": 1.1},
                boxprops={"edgecolor": "#333333", "linewidth": 1.1},
                flierprops={
                    "marker": "o",
                    "markersize": 3.0,
                    "markerfacecolor": "#333333",
                    "markeredgecolor": "#333333",
                    "alpha": 0.75,
                },
            )
            for patch, label in zip(box["boxes"], present_order):
                patch.set_facecolor(color_by_display[label])
                patch.set_alpha(0.95)
            for artist in box.get("means", []):
                artist.set_color("#333333")
                artist.set_linewidth(1.0)
            ax.legend(
                handles=[
                    Line2D(
                        [],
                        [],
                        linestyle="none",
                        marker="D",
                        markersize=4.7,
                        markerfacecolor="white",
                        markeredgecolor="#222222",
                        label="Mean",
                    ),
                    Line2D([], [], color="#333333", linewidth=1.5, label="Median"),
                ],
                loc="best",
                ncol=2,
                frameon=False,
                fontsize=8.5,
                handlelength=1.2,
                handletextpad=0.4,
                columnspacing=1.0,
            )
            ax.set_title("" if t0 is None else f"t={float(t0):g}", fontsize=12, pad=4)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            ax.tick_params(
                axis="y", which="both", left=True, labelleft=True, labelsize=12
            )
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            handles = [
                Patch(
                    facecolor=color_by_display[label], edgecolor="#333333", label=label
                )
                for label in ordered_displays
                if label in present_order
            ]
            legend_handles = [
                handles[row * 3 + column]
                for column in range(3)
                for row in range(legend_rows)
                if row * 3 + column < len(handles)
            ]
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, 0.015),
                ncol=3,
                frameon=False,
                fontsize=12,
                handlelength=1.1,
                handletextpad=0.4,
                columnspacing=0.8,
                labelspacing=0.3,
                borderaxespad=0,
            )
            fig.subplots_adjust(
                left=0.15,
                right=0.99,
                bottom=(0.24 * legend_rows + 0.15) / fig_height,
                top=0.89,
            )
            path = out_dir / f"boxplot_{metric}{t_suffix(t0)}.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            paths.append(str(path))
    return paths


from reportlab.lib.colors import Color, black, white
from reportlab.pdfgen.canvas import Canvas

LABELS = ["PABL-h", "DR-PABL", "PABL-q"]


def blue(value: float) -> Color:
    """A compact light-to-dark blue scale for values from 0 to 100."""
    fraction = max(0.0, min(1.0, value / 100.0))
    light = (0.94, 0.97, 1.0)
    dark = (0.03, 0.2, 0.43)
    return Color(
        *(light[index] + fraction * (dark[index] - light[index]) for index in range(3))
    )


def plot_agreement(panels, output) -> None:
    (width, height) = (520.0, 180.0)
    canvas = Canvas(str(output), pagesize=(width, height))
    canvas.setTitle("Blood-culture proximal policy recommendation agreement")
    panel_width = 168.0
    cell = 31.0
    matrix_y = 49.0
    for panel_index, (title, matrix) in enumerate(panels):
        panel_x = 4.0 + panel_index * panel_width
        matrix_x = panel_x + 59.0
        canvas.setFillColor(black)
        canvas.setFont("Helvetica-Bold", 8.0)
        canvas.drawCentredString(panel_x + 84.0, 165.0, title)
        canvas.setFont("Helvetica", 6.5)
        for column, label in enumerate(LABELS):
            canvas.drawCentredString(matrix_x + (column + 0.5) * cell, 148.0, label)
        for row, label in enumerate(LABELS):
            canvas.drawRightString(
                matrix_x - 5.0, matrix_y + (2.5 - row) * cell - 2.0, label
            )
        for row in range(3):
            for column in range(3):
                value = matrix[row][column]
                x = matrix_x + column * cell
                y = matrix_y + (2 - row) * cell
                canvas.setFillColor(blue(value))
                canvas.rect(x, y, cell, cell, stroke=0, fill=1)
                canvas.setFillColor(white if value >= 70 else black)
                canvas.setFont("Helvetica-Bold", 7.0)
                canvas.drawCentredString(
                    x + cell / 2, y + cell / 2 - 2.4, f"{value:.1f}"
                )
    (bar_x, bar_y, bar_width, bar_height) = (188.0, 14.0, 144.0, 7.0)
    steps = 48
    for step in range(steps):
        value = 100.0 * step / (steps - 1)
        canvas.setFillColor(blue(value))
        canvas.rect(
            bar_x + step * bar_width / steps,
            bar_y,
            bar_width / steps + 0.2,
            bar_height,
            stroke=0,
            fill=1,
        )
    canvas.setFillColor(black)
    canvas.setFont("Helvetica", 6.5)
    canvas.drawRightString(bar_x - 4.0, bar_y + 0.5, "0")
    canvas.drawString(bar_x + bar_width + 4.0, bar_y + 0.5, "100")
    canvas.setFont("Helvetica", 7.0)
    canvas.drawCentredString(bar_x + bar_width / 2, 29.0, "Pairwise agreement (%)")
    canvas.save()
