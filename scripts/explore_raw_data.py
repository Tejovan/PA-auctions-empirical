"""Exploration of the raw Community Notes / X data behind data/samples/g_full_tweets.csv.

Reads the MariaDB database `tweets` (tables note_ratings, notes, tweets; loaded from the
untracked dump tweets.sql.tar.gz), caches per-note and per-post aggregates in raw_data/cache/,
and writes figures and a README with summary statistics to output/raw_data_exploration/.

    mariadb tweets < <(tar -xzOf tweets.sql.tar.gz)      # once
    .venv/bin/python scripts/explore_raw_data.py

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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from audited_auctions.figures import CATEGORY_COLORS, INK, INK_SECONDARY, REFERENCE, STYLE  # noqa: E402

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
    eta = table.impression_count.to_numpy(float)
    misleading = ours[ours.sign < 0].groupby("tweetId").agg(
        helpful=("n_h", "sum"), somewhat=("n_s", "sum"), not_helpful=("n_nh", "sum"),
        top=("n_h", "max"), notes=("noteId", "size"))
    frame = table.set_index("id").join(misleading).fillna(0)
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 2.9))
    ax = axes[0]
    series = [("helpful", "Helpful", "-", "o"), ("somewhat", "Somewhat helpful", "--", "s"),
              ("not_helpful", "Not helpful", ":", "^"), ("top", "Helpful, top note", "-.", "D")]
    for (column, label, line, marker), color in zip(series, CATEGORY_COLORS):
        y = frame[column].to_numpy(float)
        edges = np.quantile(eta, np.linspace(0, 1, 13))
        index = np.clip(np.digitize(eta, edges[1:-1]), 0, 11)
        x = [np.median(eta[index == b]) for b in range(12)]
        m = [np.median(y[index == b]) for b in range(12)]
        slope = np.polyfit(np.log(eta[y > 0]), np.log(y[y > 0]), 1)[0]
        ax.plot(x, m, color=color, linestyle=line, marker=marker, markersize=3.5,
                linewidth=1.4, label=f"{label} (slope {slope:.2f})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("impressions $\\eta$")
    ax.set_ylabel("median ratings")
    ax.set_title("(a) Ratings on misleading notes", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25))
    for ax, column, letter, label in ((axes[1], "beta_minus", "b", "$-\\beta^-$ (misleading notes)"),
                                      (axes[2], "beta_plus", "c", "$\\beta^+$ (not-misleading notes)")):
        y = table[column].abs().to_numpy(float)
        keep = y > 0
        scatter(ax, eta[keep], y[keep])
        binned_median(ax, eta[keep], y[keep])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("impressions $\\eta$")
        ax.set_title(f"({letter}) {label}\nSpearman with $\\eta$ {rho(eta, y):.2f}; "
                     f"> 0 in {keep.mean():.0%}", loc="left")
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
    fig, axes = plt.subplots(2, 3, figsize=(9.3, 5.2))
    for column, (label, e) in enumerate(candidates):
        ax = axes[0, column]
        scatter(ax, eta, -e)
        neg = e < 0
        binned_median(ax, eta[neg], -e[neg])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("impressions $\\eta$")
        ax.set_ylabel(f"$-$({label.split(' (')[0]})")
        ax.set_title(f"({'abc'[column]}) {label}\nSpearman with $\\eta$: {rho(eta, e):.2f}",
                     loc="left")
        ax = axes[1, column]
        scatter(ax, v, -e)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("actions per view $\\alpha/\\eta$")
        ax.set_ylabel(f"$-$({label.split(' (')[0]})")
        ax.set_title(f"({'def'[column]}) vs valuation per view\nSpearman: {rho(v, e):.2f}",
                     loc="left")
    fig.tight_layout()
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
    "5_externality_normalizations": "Candidate externality per view. Top: against impressions; "
    "bottom: against actions per view (valuation per view at theta = 1). Sign flipped so "
    "harmful posts are positive; posts with beta >= 0 are not shown on the log axes.",
    "6_post_age": "Post age. (a) Post dates; (b) delay from post to its first note; "
    "(c) impressions against age in months at the pull date 2025-02-02.",
    "7_ratings_by_sign_and_level": "Which ratings grow with reach. (a) Median ratings on a "
    "post's misleading notes by helpfulness level, against impressions; slope = log-log "
    "regression slope. (b) Rating mass of misleading notes; (c) of not-misleading notes.",
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
    rows = summary(all_notes, table, ours, per_post, in_data, csv)
    rows["posts in tweets table"] = len(post_frame)
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
