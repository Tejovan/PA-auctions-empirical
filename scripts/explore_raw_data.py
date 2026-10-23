"""Exploration of the raw Community Notes / X data behind data/samples/g_full_tweets.csv.

Reads the MariaDB database `tweets` (tables note_ratings, notes, tweets; loaded from the
untracked dump raw_data/tweets.sql.tar.gz), caches per-note and per-post aggregates in raw_data/cache/,
and writes figures and a README with summary statistics to output/raw_data_exploration/.

    mariadb tweets < <(tar -xzOf raw_data/tweets.sql.tar.gz)      # once
    .venv/bin/python scripts/explore_raw_data.py

Post selection (gabemgem/XAnalysis, v1/database_queries.ipynb, which wrote top_tweetIds.pkl
for get_tweets.ipynb): posts with a note that has more than 400 HELPFUL ratings,

    SELECT r.noteId, n.tweetId, COUNT(r.noteId) FROM note_ratings r JOIN notes n
    ON r.noteId = n.noteId WHERE r.helpfulnessLevel = 'HELPFUL'
    GROUP BY r.noteId, n.tweetId HAVING COUNT(r.noteId) > 400

(20,558 notes on 13,683 posts; 11,934 of them returned by the X API).

The post-level aggregate rating reproduces create_distribution.ipynb: rating scores
HELPFUL 1, SOMEWHAT_HELPFUL 0.5, NOT_HELPFUL 0 (ratings without a helpfulness level are
dropped), +1 per note for its author, note sign -1 (misleading) or +1 (not misleading),
summed over a post's notes; posts with zero impressions dropped; 0.1% tails of the aggregate
rating and of the action count trimmed.
"""

import io
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PA_auctions.figures import (CATEGORY_COLORS, HALO, INK, INK_SECONDARY,  # noqa: E402
                                     REFERENCE, STYLE)

CACHE = ROOT / "raw_data" / "cache"
OUT = ROOT / "output" / "raw_data_exploration"
DAY = 86_400_000  # milliseconds
LAG_DAYS = (1, 3, 7, 14, 30, 90, 365)
PULL_DATE = pd.Timestamp("2025-02-02")  # date_pulled in create_distribution.ipynb
MISLEADING = "MISINFORMED_OR_POTENTIALLY_MISLEADING"
NOT_MISLEADING = "NOT_MISLEADING"


def sql(query):
    """Run a query with the mariadb client and return the result as a DataFrame (None for a
    statement without result rows)."""
    result = subprocess.run(["mariadb", "--batch", "--quick", "-e", query, "tweets"],
                            capture_output=True, text=True, check=True)
    if not result.stdout.strip():
        return None
    return pd.read_csv(io.StringIO(result.stdout), sep="\t", na_values=["NULL"],
                       keep_default_na=False)


def cached(name, build):
    path = CACHE / f"{name}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    frame = build()
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(path)
    return frame


def note_stats():
    """Per note: rating counts by helpfulness level and by delay after the note was written."""
    old = "(r.helpfulnessLevel = '' OR r.helpfulnessLevel IS NULL)"
    lags = ",\n".join(f"SUM(r.createdAtMillis - n.createdAtMillis <= {d} * {DAY}) AS n_{d}d"
                      for d in LAG_DAYS)
    sql(f"""CREATE TABLE IF NOT EXISTS note_stats ENGINE=InnoDB AS
        SELECT r.noteId,
            COUNT(*) AS n_all,
            SUM(r.helpfulnessLevel = 'HELPFUL') AS n_h,
            SUM(r.helpfulnessLevel = 'SOMEWHAT_HELPFUL') AS n_s,
            SUM(r.helpfulnessLevel = 'NOT_HELPFUL') AS n_nh,
            SUM({old}) AS n_old,
            SUM({old} AND r.helpful = 1) AS n_old_h,
            SUM({old} AND r.notHelpful = 1) AS n_old_nh,
            SUM(r.ratedOnTweetId > 0 AND r.ratedOnTweetId <> n.tweetId) AS n_other_post,
            MIN(r.createdAtMillis) AS first_ms, MAX(r.createdAtMillis) AS last_ms,
            {lags}
        FROM note_ratings r JOIN notes n ON n.noteId = r.noteId
        GROUP BY r.noteId""")
    return sql("SELECT * FROM note_stats")


def notes():
    return sql("""SELECT noteId, tweetId, classification, createdAtMillis AS note_ms,
                  isMediaNote, noteAuthorParticipantId AS author FROM notes""")


def posts():
    return sql("""SELECT id, author_id, created_at, retweet_count, reply_count, like_count,
                  quote_count, bookmark_count, impression_count, action_count FROM tweets""")


def load():
    stats = cached("note_stats", note_stats)
    all_notes = cached("notes", notes).merge(stats, on="noteId", how="left")
    count_cols = [c for c in stats.columns if c.startswith("n_")]
    all_notes[count_cols] = all_notes[count_cols].fillna(0)
    return all_notes, cached("posts", posts)


def note_scores(frame):
    """Columns of the per-note rating score under the notebook rule and variants."""
    leveled = frame.n_h + frame.n_s + frame.n_nh
    sign = frame.classification.map({MISLEADING: -1.0, NOT_MISLEADING: 1.0})
    frame = frame.assign(sign=sign, n_leveled=leveled)
    # notebook: NaN for notes without leveled ratings (dropped by the groupby sum)
    frame["score_nb"] = np.where(leveled > 0, 1 + frame.n_h + 0.5 * frame.n_s, np.nan) * sign
    # old ratings (no helpfulness level; helpful/notHelpful flags) scored 1 / 0
    frame["score_all"] = (1 + frame.n_h + 0.5 * frame.n_s + frame.n_old_h) * sign
    frame["helpful_share"] = np.where(
        frame.n_all > 0, (frame.n_h + 0.5 * frame.n_s + frame.n_old_h) / frame.n_all, np.nan)
    return frame


def post_table(all_notes, post_frame):
    """One row per post in the tweets table: notes, ratings and aggregate ratings."""
    ours = note_scores(all_notes[all_notes.tweetId.isin(post_frame.id)].copy())
    ours = ours[ours.sign.notna()]
    grouped = ours.groupby("tweetId")
    table = pd.DataFrame({
        "beta": grouped.score_nb.sum(),          # notebook: all-NaN sums to 0
        "beta_all": grouped.score_all.sum(),
        "n_notes": grouped.size(),
        "n_misleading": grouped.sign.apply(lambda s: (s < 0).sum()),
        "n_ratings": grouped.n_all.sum(),
        "n_helpful": grouped.n_h.sum(),
        "n_leveled": grouped.n_leveled.sum(),
        "n_old": grouped.n_old.sum(),
        "max_note_ratings": grouped.n_all.max(),
        "max_note_helpful": grouped.n_h.max(),
        "beta_minus": grouped.score_nb.apply(lambda x: x[x < 0].sum()),
        "beta_plus": grouped.score_nb.apply(lambda x: x[x > 0].sum()),
        "n_other_post": grouped.n_other_post.sum(),
        "first_note_ms": grouped.note_ms.min(),
    }).reset_index().rename(columns={"tweetId": "id"})
    table = post_frame.merge(table, on="id", how="inner")
    table = table[table.impression_count > 0]
    # both cutoffs from the untrimmed posts, as in the notebook
    bounds = {c: table[c].quantile([0.001, 0.999]).to_numpy() for c in ("beta", "action_count")}
    for column, (low, high) in bounds.items():
        table = table[(table[column] >= low) & (table[column] <= high)]
    table["created_at"] = pd.to_datetime(table.created_at)
    table["months_old"] = (PULL_DATE - table.created_at).dt.days / 30
    epoch_ms = (table.created_at - pd.Timestamp(0)) // pd.Timedelta(milliseconds=1)
    table["note_delay_days"] = (table.first_note_ms - epoch_ms) / DAY
    return table, ours


# ---------------------------------------------------------------- figures

def style():
    plt.rcParams.update(STYLE)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def binned_median(ax, x, y, bins=12, **kwargs):
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    index = np.clip(np.digitize(x, edges[1:-1]), 0, bins - 1)
    centers = [np.median(x[index == b]) for b in range(bins)]
    medians = [np.median(y[index == b]) for b in range(bins)]
    ax.plot(centers, medians, color=INK, marker="o", markersize=3.5, linewidth=1.4,
            label="median per bin of equal size", **kwargs)


def scatter(ax, x, y, **kwargs):
    ax.scatter(x, y, s=3, alpha=0.25, color=INK_SECONDARY, linewidths=0, rasterized=True,
               **kwargs)


def rho(x, y):
    return spearmanr(x, y)[0]


def fig_selection(all_notes, table):
    per_post = all_notes.groupby("tweetId").agg(max_note=("n_all", "max"),
                                                max_helpful=("n_h", "max"),
                                                total=("n_all", "sum"))
    ours = per_post.index.isin(table.id)
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7))
    ax = axes[0]
    bins = np.logspace(0, 4.5, 60)
    helpful = per_post.max_helpful.clip(lower=1)
    ax.hist(helpful[~ours], bins=bins, color=CATEGORY_COLORS[3], alpha=0.35,
            label="posts with notes, not in our data")
    ax.hist(helpful[ours], bins=bins, color=CATEGORY_COLORS[0], alpha=0.8,
            label="posts in our data")
    ax.axvline(400, **REFERENCE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Helpful ratings of the post's most-helpful note")
    ax.set_ylabel("number of posts")
    ax.set_title("(a) Posts by their most-helpful note", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3))
    ax = axes[1]
    edges = np.array([200, 300, 350, 390, 400, 410, 450, 500, 700, 1000, 3000, 30000])
    near = per_post[per_post.max_helpful >= edges[0]]
    index = np.digitize(near.max_helpful, edges[1:-1], right=True)
    share = pd.Series(near.index.isin(table.id)).groupby(index).mean()
    centers = np.sqrt(edges[:-1] * edges[1:])
    ax.step(edges[:-1][share.index], share.values, where="post", color=INK, linewidth=1.4)
    ax.plot(centers[share.index], share.values, "o", color=INK, markersize=3.5)
    ax.axvline(400, **REFERENCE)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Helpful ratings of the post's most-helpful note")
    ax.set_ylabel("share in our data")
    ax.set_title("(b) Selection: a note with > 400 Helpful", loc="left")
    save(fig, "1_selection")
    return per_post, ours


def fig_reach(table):
    eta = table.impression_count.to_numpy(float)
    panels = [("n_ratings", "ratings on the post's notes"), ("n_notes", "notes on the post"),
              ("beta", "aggregate rating $|\\beta|$")]
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 2.8))
    for ax, (column, label), letter in zip(axes, panels, "abc"):
        y = table[column].abs().to_numpy(float)
        keep = y > 0
        scatter(ax, eta[keep], y[keep])
        binned_median(ax, eta[keep], y[keep])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("impressions $\\eta$")
        ax.set_title(f"({letter}) {label}\nSpearman with $\\eta$: {rho(eta, y):.2f}",
                     loc="left")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.7, -0.25))
    save(fig, "2_ratings_vs_reach")


def fig_timing(ours, all_notes):
    fig, ax = plt.subplots(figsize=(3.9, 2.8))
    days = np.array(LAG_DAYS)
    for frame, color, style_, label in (
            (all_notes, CATEGORY_COLORS[3], "--", "all notes"),
            (ours, CATEGORY_COLORS[0], "-", "notes on posts in our data")):
        total = frame.n_all.sum()
        share = [frame[f"n_{d}d"].sum() / total for d in LAG_DAYS]
        ax.plot(days, share, color=color, linestyle=style_, marker="o", markersize=3.5,
                linewidth=1.4, label=label)
    ax.axvline(14, **REFERENCE)
    ax.text(14, 0.5, " status locks\n (2 weeks)", color=INK_SECONDARY, fontsize=7)
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("days after the note was written")
    ax.set_ylabel("share of ratings so far")
    ax.set_title("Ratings arrive early", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    save(fig, "3_rating_timing")


def fig_beta_parts(table, ours):
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 2.8))
    ax = axes[0]
    counts = table.n_notes.clip(upper=8)
    groups = [table.beta.abs()[counts == c] for c in range(1, 9)]
    ax.boxplot(groups, positions=range(1, 9), widths=0.6, showfliers=False,
               medianprops={"color": INK})
    ax.set_xticks(range(1, 9), [str(c) for c in range(1, 8)] + ["8+"])
    ax.set_yscale("log")
    ax.set_xlabel("notes on the post")
    ax.set_title(f"(a) $|\\beta|$ by number of notes\nSpearman: "
                 f"{rho(table.n_notes, table.beta.abs()):.2f}", loc="left")
    ax = axes[1]
    x = table.n_ratings.to_numpy(float)
    y = table.beta.abs().to_numpy(float)
    scatter(ax, x, y)
    ax.plot([x.min(), x.max()], [x.min(), x.max()], **REFERENCE, label="$|\\beta|$ = ratings")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ratings on the post's notes")
    ax.set_title(f"(b) $|\\beta|$ vs ratings\nSpearman: {rho(x, y):.2f}", loc="left")
    ax.legend(loc="upper left")
    ax = axes[2]
    share = ours.helpful_share.dropna()
    ax.hist(share, bins=40, color=CATEGORY_COLORS[1], alpha=0.8)
    ax.set_xlabel("helpful share of a note's ratings")
    ax.set_ylabel("number of notes")
    ax.set_title(f"(c) Contention per note\nmedian {share.median():.2f}", loc="left")
    save(fig, "4_beta_composition")


def fig_signs(table, ours):
    """Which ratings grow with reach: by note sign and helpfulness level."""
    eta_all = table.impression_count.to_numpy(float)
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 7.4))
    series = [("helpful", "Helpful", "-", "o"), ("somewhat", "Somewhat helpful", "--", "s"),
              ("not_helpful", "Not helpful", ":", "^"), ("top", "Helpful, top note", "-.", "D")]
    for ax, sign, letter, title in ((axes[0, 0], -1, "a", "misleading"),
                                    (axes[0, 1], 1, "b", "not-misleading")):
        counts = ours[ours.sign == sign].groupby("tweetId").agg(
            helpful=("n_h", "sum"), somewhat=("n_s", "sum"), not_helpful=("n_nh", "sum"),
            top=("n_h", "max"))
        frame = table.set_index("id").join(counts, how="inner")  # posts with such a note
        eta = frame.impression_count.to_numpy(float)
        edges = np.quantile(eta, np.linspace(0, 1, 13))
        index = np.clip(np.digitize(eta, edges[1:-1]), 0, 11)
        for (column, label, line, marker), color in zip(series, CATEGORY_COLORS):
            y = frame[column].to_numpy(float)
            x = [np.median(eta[index == b]) for b in range(12)]
            m = [np.median(y[index == b]) for b in range(12)]
            slope = np.polyfit(np.log(eta[y > 0]), np.log(y[y > 0]), 1)[0]
            ax.plot(x, m, color=color, linestyle=line, marker=marker, markersize=3.5,
                    linewidth=1.4, label=f"{label} (slope {slope:.2f})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.7, 3000)
        ax.set_xlabel("impressions $\\eta$")
        ax.set_ylabel("median ratings per post")
        ax.set_title(f"({letter}) Ratings on {title} notes\n{len(frame)} posts with such a note",
                     loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7)
    eta = eta_all
    for ax, column, letter, label in ((axes[1, 0], "beta_minus", "c", "$-\\beta^-$ (misleading notes)"),
                                      (axes[1, 1], "beta_plus", "d", "$\\beta^+$ (not-misleading notes)")):
        y = table[column].abs().to_numpy(float)
        keep = y > 0
        scatter(ax, eta[keep], y[keep])
        binned_median(ax, eta[keep], y[keep])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("impressions $\\eta$")
        ax.set_title(f"({letter}) {label}\nSpearman with $\\eta$ {rho(eta, y):.2f}; "
                     f"> 0 in {keep.mean():.0%}", loc="left")
    fig.tight_layout()
    save(fig, "7_ratings_by_sign_and_level")


def fig_normalizations(table):
    eta = table.impression_count.to_numpy(float)
    v = (table.action_count / table.impression_count).to_numpy(float)
    beta = table.beta.to_numpy(float)
    candidates = [
        ("$\\beta/\\eta$ (rating per view)", beta / eta),
        ("$\\beta$ (rating as harm per view)", beta),
        ("$\\beta$ / ratings (mean score)", beta / table.n_ratings.clip(lower=1).to_numpy(float)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(9.3, 5.8))
    negative, positive = beta < 0, beta > 0
    for column, (label, e) in enumerate(candidates):
        symbol = label.split(" (")[0]
        for row, (x, xlabel) in enumerate(((eta, "impressions $\\eta$"),
                                           (v, "actions per view $\\alpha/\\eta$"))):
            ax = axes[row, column]
            ax.scatter(x[negative], -e[negative], s=3, alpha=0.25, color=INK_SECONDARY,
                       linewidths=0, rasterized=True, label="$\\beta < 0$ (harmful)")
            ax.scatter(x[positive], e[positive], s=4, alpha=0.6, color=CATEGORY_COLORS[2],
                       linewidths=0, rasterized=True, label="$\\beta > 0$ (beneficial)")
            if row == 0:
                binned_median(ax, x[negative], -e[negative])
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"$|${symbol}$|$")
            letter = "abcdef"[3 * row + column]
            if row == 0:
                ax.set_title(f"({letter}) {label}\nSpearman with $\\eta$: {rho(eta, e):.2f}",
                             loc="left")
            else:
                ax.set_title(f"({letter}) vs valuation per view\nSpearman: {rho(v, e):.2f}",
                             loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    labels[0] = labels[0] + f" ({negative.mean():.1%} of posts)"
    labels[1] = labels[1] + f" ({positive.mean():.1%})"
    fig.legend([matplotlib.lines.Line2D([], [], color=h.get_facecolor()[0], marker="o",
                                        linestyle="none", alpha=1) for h in handles[:2]]
               + [handles[2]], labels[:2] + [labels[2] + ", harmful posts"],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "5_externality_normalizations")


def fig_age(table):
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 2.8))
    ax = axes[0]
    ax.hist(table.created_at, bins=40, color=CATEGORY_COLORS[3], alpha=0.8)
    ax.set_xlabel("post date")
    ax.set_ylabel("number of posts")
    ax.set_title("(a) When posts were written", loc="left")
    ax.tick_params(axis="x", labelrotation=30)
    ax = axes[1]
    delay = table.note_delay_days.clip(lower=1 / 24)
    ax.hist(delay, bins=np.logspace(np.log10(1 / 24), 3.5, 50), color=CATEGORY_COLORS[0],
            alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("days from post to its first note")
    ax.set_title(f"(b) Note delay, median {delay.median():.2f} days", loc="left")
    ax = axes[2]
    months = table.months_old.to_numpy(float)
    eta = table.impression_count.to_numpy(float)
    scatter(ax, months, eta)
    binned_median(ax, months, eta)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("post age in months $\\mu$")
    ax.set_ylabel("impressions $\\eta$")
    ax.set_title(f"(c) Reach does not grow with age\nSpearman: {rho(months, eta):.2f}",
                 loc="left")
    save(fig, "6_post_age")


COLOR_BINS = {
    "n_ratings": ("ratings on the post's notes", (0, 1000, 2000, 5000, np.inf)),
    "n_helpful": ("Helpful ratings on the post's notes", (0, 700, 1200, 2500, np.inf)),
}


def bin_labels(edges):
    labels = []
    for low, high in zip(edges[:-1], edges[1:]):
        labels.append(f"< {high:,.0f}" if low == 0 else
                      f"≥ {low:,.0f}" if np.isinf(high) else f"{low:,.0f}–{high:,.0f}")
    return labels


def fig_beta_colored(table):
    """Signed beta and its two sign parts against reach, points colored by rating volume."""
    eta = table.impression_count.to_numpy(float)
    panels = (("beta", "aggregate rating $\\beta$", "linear"),
              ("beta_minus", "$-\\beta^-$ (misleading notes)", "log"),
              ("beta_plus", "$\\beta^+$ (not-misleading notes)", "log"))
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 7.0))
    order = np.random.default_rng(0).permutation(len(table))
    for row, (column, (name, edges)) in enumerate(COLOR_BINS.items()):
        counts = table[column].to_numpy(float)
        index = np.digitize(counts, edges[1:-1])
        colors = matplotlib.colormaps["viridis"](np.linspace(0.1, 0.9, len(edges) - 1))
        labels = bin_labels(edges)
        for col, (field, label, scale) in enumerate(panels):
            ax = axes[row, col]
            y = table[field].to_numpy(float)
            y = -y if field == "beta_minus" else y
            keep = y > 0 if scale == "log" else np.full(len(y), True)
            sel = order[keep[order]]
            ax.scatter(eta[sel], y[sel], s=3, c=colors[index[sel]], alpha=0.45, linewidths=0,
                       rasterized=True)
            for b, color in enumerate(colors):
                in_bin = keep & (index == b)
                if in_bin.sum() >= 60:
                    edges_eta = np.quantile(eta[in_bin], np.linspace(0, 1, 7))
                    which = np.clip(np.digitize(eta[in_bin], edges_eta[1:-1]), 0, 5)
                    xs = [np.median(eta[in_bin][which == k]) for k in range(6)]
                    ms = [np.median(y[in_bin][which == k]) for k in range(6)]
                    ax.plot(xs, ms, color=color, linewidth=1.6, marker="o", markersize=3,
                            path_effects=HALO)
            ax.set_xscale("log")
            if scale == "log":
                ax.set_yscale("log")
            else:
                ax.axhline(0, **REFERENCE)
                ax.set_ylim(np.quantile(y, 0.002), np.quantile(y, 0.998))
            ax.set_xlabel("impressions $\\eta$")
            letter = "abcdef"[3 * row + col]
            ax.set_title(f"({letter}) {label}", loc="left")
        handles = [matplotlib.lines.Line2D([], [], color=c, marker="o", linestyle="-",
                                           markersize=4, label=l) for c, l in zip(colors, labels)]
        axes[row, 2].legend(handles=handles, title=f"color: {name}", loc="upper left",
                            bbox_to_anchor=(1.02, 1), title_fontsize=8)
    fig.tight_layout()
    save(fig, "13_beta_vs_reach_by_rating_volume")


def truncated_fit(x, y, cut):
    """Maximum-likelihood fit of y = a + b x + s N(0, 1), observed only where y > cut."""
    from scipy import optimize, stats

    def nll(p):
        mu = p[0] + p[1] * x
        s = np.exp(p[2])
        return -(stats.norm.logpdf(y, mu, s) - stats.norm.logsf(cut, mu, s)).sum()

    result = optimize.minimize(nll, [np.mean(y), 0.1, 0.0], method="Nelder-Mead",
                               options={"maxiter": 20000, "xatol": 1e-7, "fatol": 1e-7})
    return result.x


def top_helpful(ours, table):
    """Per post: Helpful ratings of its top misleading and top not-misleading note."""
    notes = ours[ours.tweetId.isin(table.id)]
    top = notes.groupby(["tweetId", "sign"]).n_h.max().unstack()
    top.columns = ["misleading" if c < 0 else "not_misleading" for c in top.columns]
    top["eta"] = table.set_index("id").impression_count.reindex(top.index)
    return top


def fig_truncation(ours, table):
    """Does the > 400 Helpful selection flatten the reach slope of misleading notes?"""
    top = top_helpful(ours, table)
    cut = 400
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 6.6))
    slopes = {}
    for ax, column, letter, name in ((axes[0, 0], "not_misleading", "a", "not-misleading"),
                                     (axes[0, 1], "misleading", "b", "misleading")):
        frame = top[top[column] > 0]
        x = np.log10(frame.eta.to_numpy(float))
        y = np.log10(frame[column].to_numpy(float))
        above = y > np.log10(cut)
        ax.scatter(10 ** x[~above], 10 ** y[~above], s=3, alpha=0.3, color=INK_SECONDARY,
                   linewidths=0, rasterized=True, label=f"≤ {cut} Helpful")
        ax.scatter(10 ** x[above], 10 ** y[above], s=3, alpha=0.35, color=CATEGORY_COLORS[0],
                   linewidths=0, rasterized=True, label=f"> {cut} Helpful")
        grid = np.linspace(x.min(), x.max(), 50)
        for keep, color, line, label in ((np.full(len(x), True), INK, "--", "fit, all posts"),
                                         (above, CATEGORY_COLORS[0], "-", f"fit, > {cut} only")):
            if keep.sum() < 50 or keep.sum() == len(x) and column == "misleading":
                continue
            slope, intercept = np.polyfit(x[keep], y[keep], 1)
            slopes[(column, label)] = slope
            ax.plot(10 ** grid, 10 ** (intercept + slope * grid), color=color, linestyle=line,
                    linewidth=1.6, path_effects=HALO, label=f"{label}: slope {slope:.2f}")
        ax.axhline(cut, **REFERENCE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.8, 3e4)
        ax.set_xlabel("impressions $\\eta$")
        ax.set_ylabel("Helpful ratings of the top note")
        ax.set_title(f"({letter}) Top {name} note\n{len(frame):,} posts, {above.mean():.0%} "
                     f"above {cut}", loc="left")
        ax.legend(loc="lower right", fontsize=6.5, markerscale=3)

    # (c) impose cuts on the not-misleading notes, whose slope is known without a cut
    ax = axes[0, 2]
    frame = top[top.not_misleading > 0]
    x = np.log10(frame.eta.to_numpy(float))
    y = np.log10(frame.not_misleading.to_numpy(float))
    cuts = np.array([1, 3, 10, 30, 100, 200, 400])
    naive, corrected = [], []
    for c in cuts:
        keep = y > np.log10(c + 0.5)
        naive.append(np.polyfit(x[keep], y[keep], 1)[0])
        corrected.append(truncated_fit(x[keep], y[keep], np.log10(c + 0.5))[1])
    ax.axhline(np.polyfit(x, y, 1)[0], **REFERENCE)
    ax.text(1, np.polyfit(x, y, 1)[0] + 0.015, "no cut", color=INK_SECONDARY, fontsize=7)
    ax.plot(cuts, naive, color=CATEGORY_COLORS[0], marker="o", markersize=3.5, linewidth=1.4,
            label="ordinary fit above the cut")
    ax.plot(cuts, corrected, color=CATEGORY_COLORS[1], linestyle="--", marker="s",
            markersize=3.5, linewidth=1.4, label="truncated-normal fit")
    mis = top[top.misleading > cut]
    mis_slope = np.polyfit(np.log10(mis.eta), np.log10(mis.misleading), 1)[0]
    ax.plot([cut], [mis_slope], marker="D", color=INK, linestyle="none", markersize=5,
            label="misleading notes, > 400 (ordinary)")
    ax.set_xscale("log")
    ax.set_ylim(0, 0.7)
    ax.set_xlabel("cut c (keep notes with > c Helpful)")
    ax.set_ylabel("slope of log Helpful on log $\\eta$")
    ax.set_title("(c) Cut imposed on not-misleading\nnotes", loc="left")
    ax.legend(loc="lower left", fontsize=6.5)

    # (d) model-free: share just above the cut, by reach
    ax = axes[1, 0]
    for column, c, color, line, marker, label in (
            ("misleading", 400, INK, "-", "D", "misleading, > 400"),
            ("not_misleading", 400, CATEGORY_COLORS[0], "--", "o", "not-misleading, > 400"),
            ("not_misleading", 100, CATEGORY_COLORS[1], ":", "s", "not-misleading, > 100")):
        frame = top[top[column] > c]
        bins = pd.qcut(frame.eta, 5)
        near = (frame[column] <= 1.5 * c).groupby(bins, observed=True).mean()
        xs = frame.eta.groupby(bins, observed=True).median()
        ax.plot(xs, near, color=color, linestyle=line, marker=marker, markersize=3.5,
                linewidth=1.4, label=f"{label} ({len(frame):,})")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("impressions $\\eta$ (median of fifth)")
    ax.set_ylabel("share within (c, 1.5c]")
    ax.set_title("(d) Posts just above the cut", loc="left")
    ax.legend(loc="lower left", fontsize=6.5)

    # (e, f) distributions in the lowest and highest fifth of reach
    for ax, column, letter, name in ((axes[1, 1], "misleading", "e", "misleading"),
                                     (axes[1, 2], "not_misleading", "f", "not-misleading")):
        frame = top[top[column] > 0]
        fifth = pd.qcut(frame.eta, 5, labels=False)
        bins = np.logspace(0, 4.5, 46)
        for k, color, line, label in ((0, CATEGORY_COLORS[1], "-", "lowest fifth of reach"),
                                      (4, CATEGORY_COLORS[2], "--", "highest fifth of reach")):
            values = frame[column][fifth == k]
            ax.hist(values, bins=bins, histtype="step", density=False, color=color,
                    linestyle=line, linewidth=1.4,
                    label=f"{label}: median {values.median():.0f}")
        ax.axvline(cut, **REFERENCE)
        ax.set_xscale("log")
        ax.set_xlabel("Helpful ratings of the top note")
        ax.set_ylabel("number of posts")
        ax.set_title(f"({letter}) Top {name} note by reach", loc="left")
        ax.legend(loc="upper left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "14_truncation")
    return slopes


def fig_selection_paths(ours, table):
    """Top-note Helpful counts against reach, split by which note selected the post and by sign."""
    top = top_helpful(ours, table).fillna(0)
    top["beta"] = table.set_index("id").beta.reindex(top.index)
    via_not_misleading = top.not_misleading > top.misleading
    positive = top.beta > 0
    cut = 400
    panels = (
        ("not_misleading", via_not_misleading, "a", "Top not-misleading note",
         ("selected via a misleading note", "selected via a not-misleading note")),
        ("not_misleading", positive, "b", "Top not-misleading note",
         ("$\\beta \\leq 0$", "$\\beta > 0$")),
        ("misleading", via_not_misleading, "c", "Top misleading note",
         ("selected via a misleading note", "selected via a not-misleading note")),
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.9), sharey=True)
    for ax, (column, split, letter, title, names) in zip(axes, panels):
        frame = top[top[column] > 0]
        flag = split.reindex(frame.index).to_numpy()
        x = np.log10(frame.eta.to_numpy(float))
        y = np.log10(frame[column].to_numpy(float))
        grid = np.linspace(x.min(), x.max(), 50)
        for value, color, line, name in ((False, INK_SECONDARY, "--", names[0]),
                                         (True, CATEGORY_COLORS[2], "-", names[1])):
            keep = flag == value
            ax.scatter(10 ** x[keep], 10 ** y[keep], s=3 if not value else 5,
                       alpha=0.25 if not value else 0.6, color=color, linewidths=0,
                       rasterized=True)
            slope, intercept = np.polyfit(x[keep], y[keep], 1)
            ax.plot(10 ** grid, 10 ** (intercept + slope * grid), color=color if value else INK,
                    linestyle=line, linewidth=1.6, path_effects=HALO,
                    label=f"{name} ({keep.sum():,}): slope {slope:.2f}")
        ax.axhline(cut, **REFERENCE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.8, 3e4)
        ax.set_xlabel("impressions $\\eta$")
        ax.set_title(f"({letter}) {title}", loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7)
    axes[0].set_ylabel("Helpful ratings of the top note")
    fig.tight_layout()
    save(fig, "15_selection_paths")


def hill_index(values, k):
    """Hill estimator of the tail index from the k largest values."""
    values = np.sort(np.asarray(values, float))[::-1]
    return 1 / np.mean(np.log(values[:k] / values[k]))


def fig_tails(all_notes, post_frame, table):
    """Is the upper tail of the misleading rating mass kept by the selection?"""
    scored = note_scores(all_notes.copy())
    scored = scored[scored.sign.notna()]
    grouped = scored.groupby("tweetId")
    population = pd.DataFrame({
        "mass": grouped.score_nb.apply(lambda x: -x[x < 0].sum()),
        "top": grouped.n_h.max()})
    population = population[population.mass > 0]
    in_data = population.index.isin(post_frame.id)
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.4))
    ax = axes[0]
    for values, color, line, label in (
            (population.mass, CATEGORY_COLORS[3], "--", "all noted posts"),
            (population.mass[in_data], CATEGORY_COLORS[0], "-", "our posts")):
        ccdf(ax, values, color=color, linestyle=line, linewidth=1.4,
             label=f"{label} ({len(values):,}); tail index {hill_index(values, 2000):.2f}")
    x = np.array([1600, 12000])
    ax.plot(x, 2e-3 * (x / 1600) ** -2.0, **REFERENCE, label="slope $-2$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("misleading rating mass $-\\beta^-$")
    ax.set_ylabel("share of posts with at least")
    ax.set_title("(a) Tail of the misleading rating mass", loc="left")
    ax.legend(loc="lower left", fontsize=6.5)
    ax = axes[1]
    thresholds = np.logspace(1.5, 4, 25)
    selected = [(population.top[population.mass >= x] > 400).mean() for x in thresholds]
    kept = [in_data[population.mass.to_numpy() >= x].mean() for x in thresholds]
    ax.plot(thresholds, selected, color=CATEGORY_COLORS[1], linestyle="--", linewidth=1.4,
            label="meets the rule (> 400 Helpful)")
    ax.plot(thresholds, kept, color=CATEGORY_COLORS[0], linewidth=1.4,
            label="in our data (also fetched)")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("misleading rating mass at least x")
    ax.set_ylabel("share of those posts")
    ax.set_title("(b) Share of the tail we kept", loc="left")
    ax.legend(loc="lower right", fontsize=6.5)
    ax = axes[2]
    v = (table.action_count / table.impression_count).to_numpy(float)
    mass = -table.beta_minus.to_numpy(float)
    top_decile = mass >= np.quantile(mass, 0.9)
    for keep, color, line, label in ((~top_decile, CATEGORY_COLORS[3], "--", "other 90%"),
                                     (top_decile, CATEGORY_COLORS[2], "-",
                                      "top 10% by misleading mass")):
        ccdf(ax, v[keep], color=color, linestyle=line, linewidth=1.4,
             label=f"{label}; tail index {hill_index(v[keep], 100):.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("actions per view $\\alpha/\\eta$")
    ax.set_ylabel("share of posts with at least")
    ax.set_title("(c) Valuation tail, our posts", loc="left")
    ax.legend(loc="lower left", fontsize=6.5)
    fig.tight_layout()
    save(fig, "16_tails")


# ---------------------------------------------------------------- our posts vs all noted posts

GROUPS = (("ours", "our posts", "-"), ("unfetched", "> 400 Helpful, not fetched", "--"),
          ("near", "100-400 Helpful", ":"), ("rest", "< 100 Helpful", "-."))
MISLEADING_TAGS = ("misleadingFactualError", "misleadingMissingImportantContext",
                   "misleadingUnverifiedClaimAsFact", "misleadingManipulatedMedia",
                   "misleadingOutdatedInformation", "misleadingSatire", "misleadingOther")
NOT_MISLEADING_TAGS = ("notMisleadingFactuallyCorrect", "notMisleadingPersonalOpinion",
                       "notMisleadingClearlySatire", "notMisleadingOutdatedButNotWhenWritten",
                       "notMisleadingOther")
HELPFUL_TAGS = ("helpfulGoodSources", "helpfulClear", "helpfulAddressesClaim",
                "helpfulImportantContext", "helpfulUnbiasedLanguage", "helpfulOther")
NOT_HELPFUL_TAGS = ("notHelpfulNoteNotNeeded", "notHelpfulOpinionSpeculation",
                    "notHelpfulMissingKeyPoints", "notHelpfulIncorrect",
                    "notHelpfulSourcesMissingOrUnreliable", "notHelpfulArgumentativeOrBiased",
                    "notHelpfulIrrelevantSources", "notHelpfulHardToUnderstand",
                    "notHelpfulSpamHarassmentOrAbuse", "notHelpfulOther")
SNOWFLAKE_EPOCH_MS = 1288834974657


def tag_label(tag):
    """'notHelpfulNoteNotNeeded' -> 'note not needed'."""
    for prefix in ("notMisleading", "misleading", "notHelpful", "helpful"):
        if tag.startswith(prefix):
            tag = tag[len(prefix):]
            break
    words = "".join(f" {c.lower()}" if c.isupper() else c for c in tag).strip()
    return words


def note_extras():
    """Note reason fields and per-note rating-tag sums (raw_data/*.tsv written by mariadb)."""
    fields = cached("note_fields", lambda: pd.read_csv(ROOT / "raw_data" / "note_fields.tsv",
                                                       sep="\t", na_values=["NULL"]))
    tags = cached("note_tags", lambda: pd.read_csv(ROOT / "raw_data" / "note_tags.tsv",
                                                   sep="\t", na_values=["NULL"]))
    return fields.merge(tags, on="noteId", how="left")


def post_groups(all_notes, post_ids):
    """Every post with notes, in four groups by its most-helpful note."""
    frame = all_notes.assign(misleading=all_notes.classification == MISLEADING)
    per = frame.groupby("tweetId").agg(
        max_helpful=("n_h", "max"), n_notes=("noteId", "size"), n_ratings=("n_all", "sum"),
        first_note_ms=("note_ms", "min"), share_misleading=("misleading", "mean"))
    group = np.select([per.index.isin(post_ids), per.max_helpful > 400, per.max_helpful >= 100],
                      ["ours", "unfetched", "near"], "rest")
    per["group"] = group
    ids = per.index.to_numpy(np.int64)
    post_ms = np.where(ids > 2**41, (ids >> 22) + SNOWFLAKE_EPOCH_MS, np.nan)
    per["post_ms"] = post_ms
    per["note_delay_days"] = (per.first_note_ms - per.post_ms) / DAY
    return per


def group_color(index):
    return CATEGORY_COLORS[index]


def ccdf(ax, values, **kwargs):
    values = np.sort(np.asarray(values, float))
    values = values[values > 0]
    share = 1 - np.arange(len(values)) / len(values)
    ax.step(values, share, where="post", **kwargs)


def fig_groups_counts(per, notes_grouped):
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.6))
    for i, (key, label, line) in enumerate(GROUPS):
        posts = per[per.group == key]
        notes = notes_grouped[notes_grouped.group == key]
        kwargs = {"color": group_color(i), "linestyle": line, "linewidth": 1.4,
                  "label": f"{label} ({len(posts):,} posts)"}
        ccdf(axes[0, 0], posts.n_notes, **kwargs)
        ccdf(axes[0, 1], notes.n_all, **kwargs)
        axes[1, 0].hist(posts.share_misleading, bins=np.linspace(0, 1, 21), density=True,
                        histtype="step", **kwargs)
        rated = notes[(notes.classification == MISLEADING) & (notes.n_all >= 20)]
        axes[1, 1].hist(rated.helpful_share, bins=np.linspace(0, 1, 26), density=True,
                        histtype="step", **kwargs)
    for ax in axes[0]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel("share of at least")
    axes[0, 0].set_xlabel("notes on the post")
    axes[0, 0].set_title("(a) Notes per post", loc="left")
    axes[0, 1].set_xlabel("ratings of the note")
    axes[0, 1].set_title("(b) Ratings per note", loc="left")
    axes[1, 0].set_xlabel("share of the post's notes that are misleading")
    axes[1, 0].set_ylabel("density")
    axes[1, 0].set_title("(c) Note classification mix", loc="left")
    axes[1, 1].set_xlabel("helpful share of the note's ratings")
    axes[1, 1].set_title("(d) Contention, misleading notes\nwith at least 20 ratings", loc="left")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "8_groups_notes_and_ratings")


def fig_groups_timing(per, notes_grouped):
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.2))
    days = np.array(LAG_DAYS)
    for i, (key, label, line) in enumerate(GROUPS):
        posts = per[per.group == key]
        notes = notes_grouped[notes_grouped.group == key]
        kwargs = {"color": group_color(i), "linestyle": line, "linewidth": 1.4, "label": label}
        delay = posts.note_delay_days.dropna().clip(lower=1 / 1440)
        values = np.sort(delay)
        axes[0].plot(values, np.arange(1, len(values) + 1) / len(values), **kwargs)
        total = notes.n_all.sum()
        axes[1].plot(days, [notes[f"n_{d}d"].sum() / total for d in LAG_DAYS], marker="o",
                     markersize=3, **kwargs)
        months = pd.to_datetime(posts.post_ms.dropna(), unit="ms")
        months = months[months >= "2022-10-01"].dt.to_period("M")
        share = months.value_counts(normalize=True).sort_index()
        axes[2].plot(share.index.to_timestamp(), share.values, **kwargs)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("days from post to its first note")
    axes[0].set_ylabel("share of posts")
    axes[0].set_title("(a) Note delay", loc="left")
    axes[1].set_xscale("log")
    axes[1].set_ylim(0.5, 1)
    axes[1].set_xlabel("days after the note was written")
    axes[1].set_ylabel("share of ratings so far")
    axes[1].set_title("(b) Rating timing", loc="left")
    axes[2].set_xlabel("post month")
    axes[2].set_ylabel("share of the group's posts")
    axes[2].set_title("(c) Post dates", loc="left")
    axes[2].tick_params(axis="x", labelrotation=30)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "9_groups_timing")


def grouped_bars(ax, rows, values, title, xlabel):
    """Horizontal bars: rows = tags, one bar per group."""
    height = 0.8 / len(GROUPS)
    y = np.arange(len(rows))
    for i, (key, label, _) in enumerate(GROUPS):
        ax.barh(y + (i - 1.5) * height, values[key], height=height, color=group_color(i),
                alpha=0.85, label=label, hatch=("", "///", "...", "xx")[i],
                edgecolor="white", linewidth=0.3)
    ax.set_yticks(y, [tag_label(r) for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")


def fig_groups_note_reasons(notes_grouped):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    for ax, classification, tags, letter, name in (
            (axes[0], MISLEADING, MISLEADING_TAGS + ("trustworthySources",), "a", "misleading"),
            (axes[1], NOT_MISLEADING, NOT_MISLEADING_TAGS + ("trustworthySources",), "b",
             "not-misleading")):
        values = {}
        for key, _, _ in GROUPS:
            notes = notes_grouped[(notes_grouped.group == key)
                                  & (notes_grouped.classification == classification)]
            values[key] = [notes[t].mean() for t in tags]
        grouped_bars(ax, tags, values, f"({letter}) Writer's reasons, {name} notes",
                     "share of notes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "10_groups_note_reasons")


def fig_groups_rating_reasons(notes_grouped):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.4))
    misleading = notes_grouped[notes_grouped.classification == MISLEADING]
    for ax, tags, denominator, letter, name in (
            (axes[0], HELPFUL_TAGS, ("n_h", "n_s"), "a", "Helpful and Somewhat"),
            (axes[1], NOT_HELPFUL_TAGS, ("n_nh", "n_s"), "b", "Not helpful and Somewhat")):
        values = {}
        for key, _, _ in GROUPS:
            notes = misleading[misleading.group == key]
            total = notes[list(denominator)].sum().sum()
            values[key] = [notes[t].sum() / total for t in tags]
        grouped_bars(ax, tags, values, f"({letter}) Raters' reasons\n({name} ratings)",
                     "tags per rating")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "11_groups_rating_reasons")


def fig_not_helpful_reasons_vs_reach(table, notes_grouped):
    """Within our posts: why misleading notes get Not-helpful ratings, by reach."""
    misleading = notes_grouped[(notes_grouped.classification == MISLEADING)
                               & notes_grouped.tweetId.isin(table.id)]
    misleading = misleading.assign(top=misleading.groupby("tweetId").n_h.rank(
        ascending=False, method="first") == 1)
    eta = table.set_index("id").impression_count
    bins = pd.qcut(eta, 6)
    shown = ("notHelpfulNoteNotNeeded", "notHelpfulOpinionSpeculation",
             "notHelpfulMissingKeyPoints", "notHelpfulIncorrect",
             "notHelpfulSourcesMissingOrUnreliable")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, top, letter, name in ((axes[0], True, "a", "top misleading note"),
                                  (axes[1], False, "b", "other misleading notes")):
        notes = misleading[misleading.top == top]
        per_post = notes.groupby("tweetId")[list(shown) + ["n_nh", "n_s"]].sum()
        per_post["bin"] = bins.reindex(per_post.index)
        grouped = per_post.groupby("bin", observed=True).sum()
        x = [interval.mid for interval in grouped.index]
        x = eta.groupby(bins, observed=True).median().loc[grouped.index].to_numpy()
        for (tag, color, line) in zip(shown, CATEGORY_COLORS + (INK,),
                                      ("-", "--", ":", "-.", (0, (5, 1, 1, 1)))):
            ax.plot(x, grouped[tag] / (grouped.n_nh + grouped.n_s), color=color,
                    linestyle=line, marker="o", markersize=3, linewidth=1.4,
                    label=tag_label(tag))
        ax.set_xscale("log")
        ax.set_xlabel("impressions $\\eta$ (median of sixth)")
        ax.set_title(f"({letter}) {name}", loc="left")
    axes[0].set_ylabel("tags per Not-helpful\nor Somewhat rating")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    save(fig, "12_not_helpful_reasons_vs_reach")


# ---------------------------------------------------------------- report

def summary(all_notes, table, ours, per_post, in_data, csv):
    merged = table.merge(csv[["id", "agg_post_rating"]], on="id", how="inner")
    rows = {
        "ratings in database": int(all_notes.n_all.sum()),
        "notes in database": len(all_notes),
        "posts with notes in database": len(per_post),
        "posts in tweets table": None,
        "posts after notebook filters (this script)": len(table),
        "rows of g_full_tweets.csv": len(csv),
        "posts matched to the CSV": len(merged),
        "max |beta (script) - agg_post_rating (CSV)|":
            float((merged.beta - merged.agg_post_rating).abs().max()),
        "latest rating in database": str(pd.to_datetime(all_notes.last_ms.max(), unit="ms")),
        "posts with notes and most-rated note >= 400 ratings":
            int((per_post.max_note >= 400).sum()),
        "  of which in our data": int(((per_post.max_note >= 400) & in_data).sum()),
        "our posts with most-rated note < 400 ratings": int((table.max_note_ratings < 400).sum()),
        "posts with a note with > 400 Helpful ratings": int((per_post.max_helpful > 400).sum()),
        "  of which in tweets table": None,
        "posts in tweets table with no note > 400 Helpful": None,
        "share of our ratings without helpfulness level (dropped by notebook)":
            float(table.n_old.sum() / table.n_ratings.sum()),
        "our posts whose beta changes by > 10% when old ratings are scored":
            float(((table.beta_all - table.beta).abs() > 0.1 * table.beta.abs()).mean()),
        "share of our ratings given on a different post (media notes)":
            float(table.n_other_post.sum() / table.n_ratings.sum()),
        "share of our posts with both misleading and not-misleading notes":
            float(((table.n_misleading > 0) & (table.n_misleading < table.n_notes)).mean()),
        "median notes per post": float(table.n_notes.median()),
        "median ratings per post": float(table.n_ratings.median()),
        "share of our ratings within 14 days of the note": float(
            ours.n_14d.sum() / ours.n_all.sum()),
        "Spearman(impressions, ratings)": rho(table.impression_count, table.n_ratings),
        "Spearman(impressions, notes)": rho(table.impression_count, table.n_notes),
        "Spearman(notes, |beta|)": rho(table.n_notes, table.beta.abs()),
        "Spearman(ratings, |beta|)": rho(table.n_ratings, table.beta.abs()),
    }
    return rows


FIGURES = {
    "1_selection": "Selection of posts. (a) Posts with notes by the number of Helpful ratings of "
    "their most-helpful note, all posts in the database (gray) and posts in our data (ochre). "
    "(b) Share of those posts in our data by that number: 0 at or below 400, about 87% above "
    "400 at every level.",
    "2_ratings_vs_reach": "Ratings, notes and aggregate rating against reach. Each point is a "
    "post of our data; black line: median in 12 bins of equal size.",
    "3_rating_timing": "Timing of ratings. Share of a note's ratings given within d days after "
    "the note was written, pooled over notes.",
    "4_beta_composition": "Composition of the aggregate rating. (a) |beta| by number of notes; "
    "(b) |beta| against the number of ratings (dashed: equality); (c) helpful share of a note's "
    "ratings (Helpful 1, Somewhat 0.5, Not helpful 0), notes on posts of our data.",
    "5_externality_normalizations": "Candidate externality per view, absolute value on a log "
    "axis: gray = posts with beta < 0 (harmful), magenta = beta > 0 (beneficial); posts with "
    "beta = 0 are not shown. Top: against impressions (black: median of harmful posts in 12 "
    "bins); bottom: against actions per view (valuation per view at theta = 1). Spearman "
    "correlations use signed values.",
    "6_post_age": "Post age. (a) Post dates; (b) delay from post to its first note; "
    "(c) impressions against age in months at the pull date 2025-02-02.",
    "7_ratings_by_sign_and_level": "Which ratings grow with reach. (a) Median ratings on a "
    "post's misleading notes by helpfulness level, against impressions, in 12 bins of equal "
    "size; slope = log-log regression slope over posts with a positive count. (b) The same for "
    "not-misleading notes, over posts with at least one. (c) Rating mass of misleading notes; "
    "(d) of not-misleading notes.",
    "13_beta_vs_reach_by_rating_volume": "Aggregate rating against reach, our posts. Columns: "
    "(a, d) signed beta on a linear axis (dashed: 0; axis cut at the 0.2% and 99.8% "
    "quantiles); (b, e) rating mass of misleading notes; (c, f) of not-misleading notes (posts "
    "with such notes). Top row colored by all ratings on the post's notes, bottom row by "
    "Helpful ratings; lines: median of each color group in 6 bins of equal size.",
    "14_truncation": "Does the > 400 Helpful selection flatten the reach slope? (a) Helpful "
    "ratings of each post's top not-misleading note against impressions; ochre: above 400; "
    "lines: log-log least-squares fits to all posts and to those above 400 only. (b) The same "
    "for the top misleading note (almost all above 400 by selection). (c) Slope of the "
    "not-misleading notes after keeping only notes with more than c Helpful ratings: "
    "ordinary least squares (ochre) and a truncated-normal maximum-likelihood fit (purple); "
    "dashed: slope without a cut; diamond: misleading notes above 400. (d) Share of posts "
    "with a top note at most 1.5 times the cut, by fifth of reach. (e, f) Distribution of the "
    "top note's Helpful ratings in the lowest and highest fifth of reach.",
    "15_selection_paths": "Which note selected the post. A post enters our data through its "
    "most-Helpful note (> 400 Helpful). (a) Helpful ratings of the top not-misleading note "
    "against impressions; magenta: posts selected through that not-misleading note; gray: "
    "posts selected through a misleading note. (b) The same, split by the sign of beta. "
    "(c) Helpful ratings of the top misleading note, split as in (a). Lines: log-log "
    "least-squares fits per group; dashed horizontal: 400.",
    "16_tails": "Upper tail of the misleading rating mass (Helpful 1, Somewhat 0.5, +1 per note, "
    "summed over a post's misleading notes; computable for every noted post without "
    "impressions). (a) Share of posts with at least that mass, all noted posts and our posts; "
    "tail index = Hill estimator from the 2,000 largest values; dashed: slope -2. (b) Among "
    "all noted posts with mass at least x: share meeting the selection rule and share in our "
    "data. (c) Our posts: tail of actions per view (valuation per view at theta = 1) for the "
    "10% with the largest misleading mass and the rest; Hill estimator from the 100 largest.",
    "8_groups_notes_and_ratings": "Our posts against all noted posts. Groups by the number of "
    "Helpful ratings of the post's most-helpful note: our posts; > 400 but not in our data; "
    "100-400; < 100. (a) Notes per post and (b) ratings per note, as the share of posts or "
    "notes with at least that many; (c) share of a post's notes classified misleading; "
    "(d) helpful share of the ratings of misleading notes with at least 20 ratings.",
    "9_groups_timing": "Timing by group. (a) Delay from post to its first note (post time "
    "from the post ID); (b) share of ratings within d days after the note; (c) post months "
    "from October 2022 (earlier posts: 0.2% of our posts, 2.8% of the < 100 Helpful group).",
    "10_groups_note_reasons": "Reasons the note writer selected, share of notes by group. "
    "(a) Misleading notes; (b) not-misleading notes.",
    "11_groups_rating_reasons": "Reasons raters selected on misleading notes, tags per rating "
    "by group. (a) Helpful tags per Helpful or Somewhat rating; (b) Not-helpful tags per "
    "Not-helpful or Somewhat rating. A rating can carry several tags. Tags of the 2021 "
    "rating form (unused since) are omitted.",
    "12_not_helpful_reasons_vs_reach": "Our posts: Not-helpful reasons on misleading notes by "
    "reach (sixths of impressions). (a) The post's top misleading note (most Helpful "
    "ratings); (b) its other misleading notes.",
}


def main():
    style()
    all_notes, post_frame = load()
    table, ours = post_table(all_notes, post_frame)
    csv = pd.read_csv(ROOT / "data" / "samples" / "g_full_tweets.csv", usecols=["id",
                                                                             "agg_post_rating"])
    per_post, in_data = fig_selection(all_notes, table)
    fig_reach(table)
    fig_timing(ours, all_notes)
    fig_beta_parts(table, ours)
    fig_normalizations(table)
    fig_age(table)
    fig_signs(table, ours)
    fig_beta_colored(table)
    fig_truncation(ours, table)
    fig_selection_paths(ours, table)
    fig_tails(all_notes, post_frame, table)
    per = post_groups(all_notes, post_frame.id)
    notes_grouped = note_scores(all_notes.merge(note_extras(), on="noteId", how="left")).merge(
        per[["group"]], left_on="tweetId", right_index=True)
    fig_groups_counts(per, notes_grouped)
    fig_groups_timing(per, notes_grouped)
    fig_groups_note_reasons(notes_grouped)
    fig_groups_rating_reasons(notes_grouped)
    fig_not_helpful_reasons_vs_reach(table, notes_grouped)
    rows = summary(all_notes, table, ours, per_post, in_data, csv)
    rows["posts in tweets table"] = len(post_frame)
    for key, label, _ in GROUPS:
        rows[f"posts with notes: {label}"] = int((per.group == key).sum())
    selected = per_post.index[per_post.max_helpful > 400]
    rows["  of which in tweets table"] = int(selected.isin(post_frame.id).sum())
    rows["posts in tweets table with no note > 400 Helpful"] = int(
        (~post_frame.id.isin(selected)).sum())
    table.to_pickle(CACHE / "post_table.pkl")
    lines = ["| statistic | value |", "|---|---|"]
    for key, value in rows.items():
        text = f"{value:.4g}" if isinstance(value, float) else str(value)
        lines.append(f"| {key} | {text} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    readme = ["# Raw data exploration", "",
              "Written by `scripts/explore_raw_data.py` from the MariaDB copy of the raw dump.", "",
              "## Summary statistics", "", *lines, ""]
    for name, caption in FIGURES.items():
        readme += [f"## {caption.split('.')[0]}", "", f"![{name}]({name}.png)", "",
                   caption, ""]
    (OUT / "README.md").write_text("\n".join(readme))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
