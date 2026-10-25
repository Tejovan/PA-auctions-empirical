r"""Data for two figures across type distributions (draft figures; later the supplement):

  (1) per market: share of the first-best gain of every τ by distribution (top row) and the
      τ of the four forms over each type distribution (bottom row);
  (2) per distribution: the VCG-PA-optimized τ of every market with k < n.

    python scripts/family_figures_data.py OUT_DIR RUN [RUN ...]

Each value is taken from the first run that has it. Type-plane panels are zoomed to the type distribution
on a linear valuation axis (web27_paper_data.zoomed_view: externality quantiles 0.5% and 99.5%,
valuation quantiles 0.5% and 99%; VIEW_OPTIONS for independent and heavy-tail). Writes to OUT_DIR:
  family_share_n<n>k<k>.dat        x (distribution index), family, tau0 cutoff affine relaxtau
                                   vcgpa ivcgpa ivcgaffine exante exanteivcg gammacut bound
                                   guarantee gammabound (share of the first-best gain, %, nan where
                                   a run lacks the design; exante, exanteivcg = VCG-PA, IVCG-PA with
                                   the ex ante welfare cutoffs; gammacut = VCG-PA with the cutoff of
                                   the VCG-PA guarantee (run of configs/vcg_guarantee_run.toml);
                                   guarantee, gammabound = the shares of max{β_{n,k}, 1 − 1/e}·FB
                                   and γ_{n,k}·FB); for the columns of LOW_HEIGHTS also <c>low
                                   (the value where it is below SHARE_FLOOR, else nan) and <c>lowy
                                   (LOW_HEIGHTS[c] there, else nan): where to draw marks below the
                                   plotted range
  family_fb_n<n>k<k>.dat           x family, the same designs and bounds in % of the first best, and
                                   vcg (VCG without audits)
  family_forms_n<n>k<k>_<family>.dat    e tau0 cutoff affine nodes relaxtau exante (VCG-PA τ of each
                                       form, the relaxation τ and the ex ante welfare cutoffs)
  family_tau_<family>.dat           e and one column n<n>k<k> per market with k < n
  family_tau_<family>.tex           pgfplots commands drawing those τ (color n/k, style n; solid
                                   lines first, dotted last, so dashed and dotted stay visible)
  family_taufew_<family>.dat/.tex   the same for the markets with n/k in FEW_RATIOS, for the
                                    zoomed limits
  family_tau_limits.tex             zoomed axis limits for those (\eminZoom<Family> ...): the
                                    range of the curves, v up to ZOOM_TOP (ZOOM_TOP_FAMILY) of the view; and the
                                    limits of the share figure's panels (\eminShare<Family> ...)
  bg_<family>.png, family_limits.tex    type background and axis limits of each panel
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from PA_auctions import figures  # noqa: E402
from PA_auctions.threshold_forms import ex_ante_guarantee, vcg_guarantee  # noqa: E402
from web27_paper_data import (VIEW_OPTIONS, label, limit_macros, load_data, load_types, padded,  # noqa: E402
                              at_support_top, zoomed_view)

FAMILY_ORDER = ["independent", "harmful-middle", "heavy-tail", "harm-rises", "two-modes"]
MACROS = {"independent": "Independent", "harmful-middle": "HarmfulMiddle",
          "heavy-tail": "HeavyTail", "harm-rises": "HarmRises", "two-modes": "TwoModes",
}
SHARE_KEYS = {"tau0": ("zero-welfare", "share_vcg_pa"), "cutoff": ("vcg_pa_cutoff", "share_vcg_pa"),
              "affine": ("vcg_pa_poly1", "share_vcg_pa"), "relaxtau": ("relaxation", "share_vcg_pa"),
              "vcgpa": ("vcg_pa_nodes", "share_vcg_pa"), "ivcgpa": ("ivcg_pa_nodes", "share_ivcg_pa"),
              "ivcgaffine": ("ivcg_pa_poly1", "share_ivcg_pa"), "exante": ("ex-ante", "share_vcg_pa"),
              "exanteivcg": ("ex-ante", "share_ivcg_pa"), "gammacut": ("vcg-guarantee", "share_vcg_pa")}
SHARE_FLOOR = -8.0  # %: lower values are drawn below the axis range and labeled with their value
LOW_HEIGHTS = {"exante": -8.0, "exanteivcg": -8.0, "gammacut": -8.0, "guarantee": -21.0,
               "gammabound": -21.0}  # % heights of those marks
FORM_KEYS = {"tau0": "zero-welfare", "cutoff": "vcg_pa_cutoff", "affine": "vcg_pa_poly1",
             "nodes": "vcg_pa_nodes", "relaxtau": "relaxation", "exante": "ex-ante"}
FORM_MARKETS = [(2, 1), (8, 1), (16, 4)]
# Figure (2): color by bidders per good n/k (viridis, figstyle sevenA–G), line style by n.
RATIO_COLORS = {8 / 7: "sevenA", 4 / 3: "sevenB", 2: "sevenC", 4: "sevenD", 8: "sevenE",
                16: "sevenF", 32: "sevenG"}
BIDDER_STYLES = {2: "dotted", 4: "dash dot dot", 8: "dashdotted", 16: "dashed", 32: "solid"}
FEW_RATIOS = (1.2, 8)  # the markets of family_taufew: 1.2 < n/k ≤ 8 (n/k = 8/7 left out)
ZOOM_TOP = 0.85
ZOOM_TOP_FAMILY = {"independent": 1.0}  # independent: valuations up to the top of the view
# Type-plane panels of the share figure: externality range narrowed by SHARE_E[0] of its width on the
# left and widened by SHARE_E[1] on the right; valuation range widened by SHARE_V at the top.
SHARE_E = (0.12, 0.02)
SHARE_V = 0.12
SHARE_V_FAMILY = {"independent": 0.0}  # valuations uniform on [0, 1]: no types above the view


def share_limits(view, family):
    e_low, e_high = (float(x) for x in view["e_limits"])
    v_low, v_high = (float(x) for x in view["v_limits"])
    width, height = e_high - e_low, v_high - v_low
    return (e_low + SHARE_E[0] * width, e_high + SHARE_E[1] * width, v_low,
            v_high + SHARE_V_FAMILY.get(family, SHARE_V) * height)


def zoom_limits(view, e_grid, columns, zoom_top=ZOOM_TOP):
    """Axis limits around τ curves: valuations up to ZOOM_TOP of the view, externalities from
    the first point below that top to the last point with τ > 0 (2% of the view), with a 4% margin."""
    v_low, v_high = (float(x) for x in view["v_limits"])
    e_low, e_high = (float(x) for x in view["e_limits"])
    top = v_low + zoom_top * (v_high - v_low)
    taus = np.column_stack(columns)
    below_top = np.any(taus <= top, axis=1)
    above_bottom = np.any(taus > max(v_low, 0.0) + 0.02 * (v_high - v_low), axis=1)
    left = float(e_grid[below_top].min()) if below_top.any() else e_low
    right = float(e_grid[above_bottom].max()) if above_bottom.any() else e_high
    margin = 0.04 * (right - left)
    return max(e_low, left - margin), min(e_high, right + margin), v_low, top


def shown(data, view):
    """τ(e) for a panel of the view: an excluded externality at the top valuation v̄
    (at_support_top); values above the panel are clipped by the axis."""
    return at_support_top(data["tau_grid"], float(view["v_limits"][0]), float(view["maximum_valuation"]))


def raw(data):
    """τ(e), nan where the externality is excluded."""
    tau = np.asarray(data["tau_grid"], dtype=float)
    return np.where(np.isfinite(tau), tau, np.nan)


def background(view, path):
    fig = figures.plt.figure(figsize=(3.0, 2.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    figures.draw_types(ax, padded(view))
    ax.set_axis_off()
    fig.savefig(path, dpi=300, transparent=True)
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

    def guarantee(self, family, n, k, factor=ex_ante_guarantee):
        """Share of the first-best gain of factor(n, k)·FB: max{β_{n,k}, 1 − 1/e} (ex ante
        guarantee, IVCG-PA) or γ_{n,k} (VCG-PA guarantee)."""
        if k >= n:
            return np.nan
        for summary in self.summaries:
            rows = summary[(summary.distribution == family) & (summary.num_bidders == n)
                           & (summary.num_goods == k)]
            if len(rows):
                first_best, vcg = rows.welfare_first_best.iloc[0], rows.welfare_vcg.iloc[0]
                return float((factor(n, k) * first_best - vcg) / (first_best - vcg))
        return np.nan

    def vcg_fraction(self, family, n, k):
        """Welfare of VCG without audits as a fraction of the first best."""
        for summary in self.summaries:
            rows = summary[(summary.distribution == family) & (summary.num_bidders == n)
                           & (summary.num_goods == k)]
            if len(rows):
                return float(rows.welfare_vcg.iloc[0] / rows.welfare_first_best.iloc[0])
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


BOUNDS = ["bound", "guarantee", "gammabound"]


def bounds(runs, family, n, k):
    """Relaxation bound, ex ante guarantee and VCG-PA guarantee γ_{n,k}, as shares."""
    return [runs.bound(family, n, k), runs.guarantee(family, n, k),
            runs.guarantee(family, n, k, lambda n, k: vcg_guarantee(n, k)[0])]


def percent(value):
    return "nan" if not np.isfinite(value) else f"{100 * value:.2f}"


def main(out, folders):
    out.mkdir(parents=True, exist_ok=True)
    runs = Runs(folders)
    families = runs.families()[:4]
    panels = [(family, family, zoomed_view(runs.types(family), **VIEW_OPTIONS.get(family, {})))
              for family in families]
    limits = []
    for name, _, view in panels:
        background(view, out / f"bg_{name}.png")
        limits.append(limit_macros(MACROS[name], view))
    (out / "family_limits.tex").write_text("\n".join(limits) + "\n")

    markets = sorted(set().union(*(set(runs.markets(f)) for f in families)))
    for n, k in markets:
        with open(out / f"family_share_n{n}k{k}.dat", "w") as file:
            extra = [f"{c}{suffix}" for c in LOW_HEIGHTS for suffix in ("low", "lowy")]
            file.write("x family " + " ".join(list(SHARE_KEYS) + BOUNDS + extra) + "\n")
            for index, family in enumerate(families):
                values = [runs.share(family, n, k, *SHARE_KEYS[c]) for c in SHARE_KEYS]
                values += bounds(runs, family, n, k)
                named = dict(zip(list(SHARE_KEYS) + BOUNDS, values))
                for column in LOW_HEIGHTS:
                    low = np.isfinite(named[column]) and 100 * named[column] < SHARE_FLOOR
                    values += [named[column] if low else np.nan, LOW_HEIGHTS[column] / 100 if low else np.nan]
                file.write(f"{index} {family} " + " ".join(percent(v) for v in values) + "\n")
        # The same designs in % of the first best: W/FB = W0/FB + share·(1 − W0/FB), W0 = VCG without audits.
        with open(out / f"family_fb_n{n}k{k}.dat", "w") as file:
            file.write("x family " + " ".join(list(SHARE_KEYS) + BOUNDS + ["vcg"]) + "\n")
            for index, family in enumerate(families):
                base = runs.vcg_fraction(family, n, k)
                values = [runs.share(family, n, k, *SHARE_KEYS[c]) for c in SHARE_KEYS]
                values += bounds(runs, family, n, k)
                values = [base + v * (1 - base) for v in values] + [base]
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

    zooms = []
    for panel, family, view in panels:
        names, columns, raws, e_grid = [], [], [], None
        for n, k in runs.markets(family):
            if k >= n:
                continue
            data = load_data(runs.leaf(family, n, k, "vcg_pa_nodes"))
            e_grid = data["e_grid"]
            columns.append(shown(data, view))
            raws.append(raw(data))
            names.append(f"n{n}k{k}")
        np.savetxt(out / f"family_tau_{panel}.dat", np.column_stack([e_grid] + columns),
                   header="e " + " ".join(names), comments="", fmt="%.5g")
        markets = [tuple(int(x) for x in name[1:].split("k")) for name in names]
        # Solid (n = 32) first, dotted (n = 2) last; within one n, larger n/k first.
        order = sorted(markets, key=lambda m: (-m[0], -m[0] / m[1]))
        for suffix, chosen in (("tau", order), ("taufew", [m for m in order if FEW_RATIOS[0] < m[0] / m[1] <= FEW_RATIOS[1]])):
            commands = []
            for n, k in chosen:
                color = RATIO_COLORS[min(RATIO_COLORS, key=lambda r: abs(r - n / k))]
                commands.append(rf"\addplot[{color},{BIDDER_STYLES[n]},line width=1.2pt,forget plot,"
                                rf"unbounded coords=jump] table[x=e,y=n{n}k{k}]{{\familydata/family_{suffix}_{panel}.dat}};")
            (out / f"family_{suffix}_{panel}.tex").write_text("\n".join(commands) + "\n")
        few = [i for i, m in enumerate(markets) if FEW_RATIOS[0] < m[0] / m[1] <= FEW_RATIOS[1]]
        e_min, e_max, v_min, v_max = zoom_limits(view, np.asarray(e_grid, dtype=float), [raws[i] for i in few],
                                                 ZOOM_TOP_FAMILY.get(family, ZOOM_TOP))
        np.savetxt(out / f"family_taufew_{panel}.dat",
                   np.column_stack([e_grid] + [columns[i] for i in few]),
                   header="e " + " ".join(names[i] for i in few), comments="", fmt="%.5g")
        macro = MACROS[panel]
        zooms.append(rf"\def\eminZoom{macro}{{{e_min:.4f}}}\def\emaxZoom{macro}{{{e_max:.4f}}}"
                     rf"\def\vminZoom{macro}{{{v_min:.5g}}}\def\vmaxZoom{macro}{{{v_max:.4f}}}")
        e_min, e_max, v_min, v_max = share_limits(view, family)
        zooms.append(rf"\def\eminShare{macro}{{{e_min:.4f}}}\def\emaxShare{macro}{{{e_max:.4f}}}"
                     rf"\def\vminShare{macro}{{{v_min:.5g}}}\def\vmaxShare{macro}{{{v_max:.4f}}}")
    (out / "family_tau_limits.tex").write_text("\n".join(zooms) + "\n")
    print("families", families, "markets", len(markets))


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
