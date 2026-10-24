"""Data for two figures across type distributions (draft figures; later the supplement):

  (1) per market: share of the first-best gain of every τ by distribution (top row) and the
      τ of the four forms over each type distribution (bottom row);
  (2) per distribution: the VCG-PA-optimized τ of every market with k < n.

    python scripts/family_figures_data.py OUT_DIR RUN [RUN ...]

Each value is taken from the first run that has it (so a run with the synthetic markets can
come before a run with more families). Type-plane panels are zoomed to the type distribution
(externality quantiles 0.5% and 99.5%, valuation quantiles 0.5% and 99%; 98% for heavy-tail)
on a linear valuation axis. Comparison panels: heavy-tail-log (logarithmic, valuation quantiles
0.5% and 99.5%), heavy-tail-q80 and heavy-tail-q75 (linear up to the 80% and 75% valuation
quantiles), independent-zoom (externality quantiles 2.5% and 97.5%). Writes to OUT_DIR:
  family_share_n<n>k<k>.dat        x (distribution index), family, tau0 cutoff affine relaxtau
                                   vcgpa ivcgpa ivcgaffine bound (share of the first-best gain,
                                   %, nan where a run lacks the design)
  family_forms_n<n>k<k>_<panel>.dat    e tau0 cutoff affine nodes (VCG-PA τ of each form)
  family_tau_<panel>.dat           e and one column n<n>k<k> per market with k < n
  family_tau_<panel>.tex           pgfplots commands drawing those τ (color n/k, style n)
  bg_<panel>.png, family_limits.tex    type background and axis limits of each panel
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from PA_auctions import figures  # noqa: E402
from web27_paper_data import label, load_data, load_types  # noqa: E402

FAMILY_ORDER = ["independent", "harmful-middle", "heavy-tail", "harm-rises", "two-modes"]
MACROS = {"independent": "Independent", "harmful-middle": "HarmfulMiddle",
          "heavy-tail": "HeavyTail", "harm-rises": "HarmRises", "two-modes": "TwoModes",
          "heavy-tail-log": "HeavyTailLog", "heavy-tail-q80": "HeavyTailQEighty",
          "heavy-tail-q75": "HeavyTailQSeventyFive", "independent-zoom": "IndependentZoom"}
E_LEVELS, V_LEVELS, V_TOP_HEAVY, V_LEVELS_LOG = (0.005, 0.995), (0.005, 0.99), 0.98, (0.005, 0.995)
SHARE_KEYS = {"tau0": ("zero-welfare", "share_vcg_pa"), "cutoff": ("vcg_pa_cutoff", "share_vcg_pa"),
              "affine": ("vcg_pa_poly1", "share_vcg_pa"), "relaxtau": ("relaxation", "share_vcg_pa"),
              "vcgpa": ("vcg_pa_nodes", "share_vcg_pa"), "ivcgpa": ("ivcg_pa_nodes", "share_ivcg_pa"),
              "ivcgaffine": ("ivcg_pa_poly1", "share_ivcg_pa")}
FORM_KEYS = {"tau0": "zero-welfare", "cutoff": "vcg_pa_cutoff", "affine": "vcg_pa_poly1",
             "nodes": "vcg_pa_nodes"}
FORM_MARKETS = [(2, 1), (8, 1), (16, 4)]
# Figure (2): color by bidders per good n/k (viridis, figstyle sevenA–G), line style by n.
RATIO_COLORS = {8 / 7: "sevenA", 4 / 3: "sevenB", 2: "sevenC", 4: "sevenD", 8: "sevenE",
                16: "sevenF", 32: "sevenG"}
BIDDER_STYLES = {2: "dotted", 4: "dash dot dot", 8: "dashdotted", 16: "dashed", 32: "solid"}


def shown(data, view):
    """τ(e) where it is inside the view, nan above it (excluded externalities included)."""
    tau = np.asarray(data["tau_grid"], dtype=float)
    low, high = (float(x) for x in view["v_limits"])
    return np.where(np.isfinite(tau) & (tau <= high), np.maximum(tau, low), np.nan)


def _quantiles(axis, weights, levels):
    cumulative = np.cumsum(weights) / weights.sum()
    return [float(np.interp(level, cumulative, axis)) for level in levels]


def zoomed_view(types, scale="linear", e_levels=E_LEVELS, v_top=None):
    """The view of a type-plane panel, zoomed to quantiles of the marginal distributions of e and
    v (from the background density, valuations × externalities), with a 3% margin; v_top
    overrides the upper valuation quantile of a linear view."""
    view = figures.type_view(types, scale)
    density = np.asarray(types["background_density"], dtype=float)
    e_low, e_high = _quantiles(np.asarray(types["background_e"]), density.sum(axis=0), e_levels)
    heavy = figures.default_view(types)["scale"] == "log"
    top = v_top or (V_TOP_HEAVY if heavy else V_LEVELS[1])
    levels = V_LEVELS_LOG if scale == "log" else (V_LEVELS[0], top)
    v_low, v_high = _quantiles(np.asarray(types["background_v"]), density.sum(axis=1), levels)
    e_margin = 0.03 * (e_high - e_low)
    view["e_limits"] = np.array([e_low - e_margin, e_high + e_margin])
    if scale == "log":
        view["v_limits"] = np.array([max(v_low, 1e-3), v_high * 1.05])
    else:
        v_margin = 0.03 * (v_high - v_low)
        view["v_limits"] = np.array([v_low - v_margin, v_high + v_margin])
    return view


def background(view, path):
    fig = figures.plt.figure(figsize=(3.0, 2.0), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    figures.draw_types(ax, view)
    ax.set_axis_off()
    fig.savefig(path, dpi=200, transparent=True)
    figures.plt.close(fig)
    return view


class Runs:
    """Summaries and gap decompositions of several runs; lookups take the first run with a row."""

    def __init__(self, folders):
        self.folders = [Path(f) for f in folders]
        self.summaries, self.gaps = [], []
        for folder in self.folders:
            summary = pd.read_csv(folder / "summary.csv")
            summary["key"] = summary.apply(label, axis=1)
            self.summaries.append(summary)
            self.gaps.append(pd.read_csv(folder / "gap-decomposition.csv"))

    def families(self):
        present = set().union(*(set(s.distribution) for s in self.summaries))
        return [f for f in FAMILY_ORDER if f in present]

    def rows(self, family, n, k, key):
        for folder, summary in zip(self.folders, self.summaries):
            rows = summary[(summary.distribution == family) & (summary.num_bidders == n)
                           & (summary.num_goods == k) & (summary.key == key)]
            if len(rows):
                return folder, rows
        return None, None

    def share(self, family, n, k, key, column):
        _, rows = self.rows(family, n, k, key)
        return float(rows[column].max()) if rows is not None else np.nan

    def bound(self, family, n, k):
        """Relaxation bound as in the paper's table: VCG-PA optimized share + relaxation gap."""
        for summary, gaps in zip(self.summaries, self.gaps):
            nodes = gaps[(gaps.distribution == family) & (gaps.num_bidders == n)
                         & (gaps.num_goods == k) & (gaps.threshold_class == "nodes")]
            rows = summary[(summary.distribution == family) & (summary.num_bidders == n)
                           & (summary.num_goods == k) & (summary.key == "vcg_pa_nodes")]
            if len(nodes) and len(rows) and np.isfinite(nodes.relaxation_minus_opt_vcg_share.iloc[0]):
                return float(rows.share_vcg_pa.max() + nodes.relaxation_minus_opt_vcg_share.iloc[0])
        return np.nan

    def leaf(self, family, n, k, key):
        folder, rows = self.rows(family, n, k, key)
        if rows is None:
            return None
        column = "share_ivcg_pa" if key.startswith("ivcg") else "share_vcg_pa"
        return folder / rows.loc[rows[column].idxmax()].path  # best of several optimizers

    def markets(self, family):
        found = set()
        for summary in self.summaries:
            part = summary[(summary.distribution == family) & (summary.key == "vcg_pa_nodes")]
            found |= set(zip(part.num_bidders, part.num_goods))
        return sorted(found)

    def types(self, family):
        for folder in self.folders:
            if (folder / family / "z1" / "types.npz").exists():
                return load_types(folder, family)
        raise FileNotFoundError(family)


def percent(value):
    return "nan" if not np.isfinite(value) else f"{100 * value:.2f}"


def main(out, folders):
    out.mkdir(parents=True, exist_ok=True)
    runs = Runs(folders)
    families = runs.families()[:4]
    # panels: (name, family, view); heavy-tail also in a logarithmic version
    panels = [(family, family, zoomed_view(runs.types(family))) for family in families]
    if "heavy-tail" in families:
        heavy = runs.types("heavy-tail")
        panels += [("heavy-tail-log", "heavy-tail", zoomed_view(heavy, "log")),
                   ("heavy-tail-q80", "heavy-tail", zoomed_view(heavy, v_top=0.80)),
                   ("heavy-tail-q75", "heavy-tail", zoomed_view(heavy, v_top=0.75))]
    if "independent" in families:  # externality quantiles 2.5% and 97.5%
        panels.append(("independent-zoom", "independent",
                       zoomed_view(runs.types("independent"), e_levels=(0.025, 0.975))))
    limits = []
    for name, _, view in panels:
        background(view, out / f"bg_{name}.png")
        e_low, e_high = (float(x) for x in view["e_limits"])
        v_low, v_high = (float(x) for x in view["v_limits"])
        macro = MACROS[name]
        limits.append(rf"\def\emin{macro}{{{e_low:.4f}}}\def\emax{macro}{{{e_high:.4f}}}"
                      rf"\def\vmin{macro}{{{v_low:.5g}}}\def\vmax{macro}{{{v_high:.4f}}}")
    (out / "family_limits.tex").write_text("\n".join(limits) + "\n")

    markets = sorted(set().union(*(set(runs.markets(f)) for f in families)))
    for n, k in markets:
        with open(out / f"family_share_n{n}k{k}.dat", "w") as file:
            file.write("x family " + " ".join(list(SHARE_KEYS) + ["bound"]) + "\n")
            for index, family in enumerate(families):
                values = [runs.share(family, n, k, *SHARE_KEYS[c]) for c in SHARE_KEYS]
                values.append(runs.bound(family, n, k))
                file.write(f"{index} {family} " + " ".join(percent(v) for v in values) + "\n")

    for n, k in FORM_MARKETS:
        for name, family, view in panels:
            columns, e_grid = [], None
            for key in FORM_KEYS.values():
                leaf = runs.leaf(family, n, k, key)
                if leaf is None:
                    columns.append(None)
                    continue
                data = load_data(leaf)
                e_grid = data["e_grid"]
                columns.append(shown(data, view))
            columns = [c if c is not None else np.full(len(e_grid), np.nan) for c in columns]
            np.savetxt(out / f"family_forms_n{n}k{k}_{name}.dat", np.column_stack([e_grid] + columns),
                       header="e " + " ".join(FORM_KEYS), comments="", fmt="%.5g")

    for panel, family, view in panels:
        names, columns, e_grid = [], [], None
        for n, k in runs.markets(family):
            if k >= n:
                continue
            data = load_data(runs.leaf(family, n, k, "vcg_pa_nodes"))
            e_grid = data["e_grid"]
            columns.append(shown(data, view))
            names.append(f"n{n}k{k}")
        np.savetxt(out / f"family_tau_{panel}.dat", np.column_stack([e_grid] + columns),
                   header="e " + " ".join(names), comments="", fmt="%.5g")
        commands = []
        for name in sorted(names, key=lambda m: -int(m[1:].split("k")[0]) / int(m.split("k")[1])):
            n, k = (int(x) for x in name[1:].split("k"))
            color = RATIO_COLORS[min(RATIO_COLORS, key=lambda r: abs(r - n / k))]
            commands.append(rf"\addplot[{color},{BIDDER_STYLES[n]},line width=1.2pt,forget plot,"
                            rf"unbounded coords=jump] table[x=e,y={name}]{{\familydata/family_tau_{panel}.dat}};")
        (out / f"family_tau_{panel}.tex").write_text("\n".join(commands) + "\n")
    print("families", families, "markets", len(markets))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
