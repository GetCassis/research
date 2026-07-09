"""Analysis module for the derivation-distance study.

Reads results/runs/<run>.jsonl, produces figures in results/figures/ and a
plain-language results/summary.md. Tests hypotheses H1-H4 (see README).

The derivation-distance axis is config.DERIVATION_LEVELS (L1..L6), all complete;
the headline accuracy line and every per-level line plot cover that axis only.
config.SIDE_ARMS (L0 bare schema) is deliberately incomplete and is reported
separately (a distinct marker on the accuracy figure, its own summary section).

Usage:
    python src/analyze.py --run pilot
    python src/analyze.py --run smoke   # small dev/debug set

Dependencies: pandas + numpy + matplotlib only (no seaborn).
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

DPI = 150

# The derivation-distance axis (L1..L6), all complete. Completeness held constant
# across this axis, so accuracy differences here are derivation distance.
DERIVATION_LEVELS = list(config.DERIVATION_LEVELS)
# Side arm(s): bare schema with NO required facts (deliberately incomplete).
# Reported SEPARATELY; never plotted as a point on the derivation-distance line.
SIDE_ARMS = list(config.SIDE_ARMS)
# Full set of levels that may appear in a run (side arms + derivation axis).
LEVEL_ORDER = list(config.LEVELS)
LEVEL_SHORT = {name: name.split("_")[0] for name in LEVEL_ORDER}  # "L1_schema_descriptions" -> "L1"

# Baseline of the derivation axis (max derivation distance, still complete).
BASELINE = DERIVATION_LEVELS[0]              # "L1_schema_descriptions"
BASELINE_SHORT = LEVEL_SHORT[BASELINE]       # "L1"
TOP = DERIVATION_LEVELS[-1]                  # "L6_answer_templates"
TOP_SHORT = LEVEL_SHORT[TOP]                 # "L6"
ANCHOR = config.ANCHOR_LEVEL                 # "L4_metric_formulas" (canonical ontology)
ANCHOR_SHORT = LEVEL_SHORT[ANCHOR]           # "L4"

# Model display order (capability-ascending) and stable colours.
MODEL_ORDER = ["haiku", "sonnet", "opus"]
MODEL_COLOR = {"haiku": "#d62728", "sonnet": "#ff7f0e", "opus": "#1f77b4"}
MODEL_MARKER = {"haiku": "o", "sonnet": "s", "opus": "^"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_runs(run_name):
    """Load the jsonl run file into a DataFrame, with level ordering applied."""
    path = os.path.join(config.BASE, "results", "runs", f"{run_name}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run file not found: {path}")
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Run file is empty: {path}")
    df = pd.DataFrame(rows)

    # Ensure expected columns exist even if some runs omitted them.
    for col, default in [
        ("thinking_tokens", 0),
        ("context_tokens", np.nan),
        ("input_tokens", np.nan),
        ("output_tokens", np.nan),
        ("cost_usd", np.nan),
        ("correct", False),
        ("exec_ok", False),
        ("error", None),
        ("sample", 0),
    ]:
        if col not in df.columns:
            df[col] = default

    # Normalise types.
    df["correct"] = df["correct"].fillna(False).astype(bool)
    df["error"] = df["error"].where(df["error"].notna(), None)
    df["thinking_tokens"] = pd.to_numeric(df["thinking_tokens"], errors="coerce").fillna(0)

    # Order levels as a categorical for clean grouping / plotting.
    present_levels = [lv for lv in LEVEL_ORDER if lv in set(df["level"])]
    df["level"] = pd.Categorical(df["level"], categories=LEVEL_ORDER, ordered=True)
    df.attrs["present_levels"] = present_levels
    return df


def levels_in(df):
    """All levels actually present (side arms + derivation), in config order."""
    return [lv for lv in LEVEL_ORDER if (df["level"] == lv).any()]


def derivation_levels_in(df):
    """Derivation-axis levels actually present, in config order (L1..L6). No side arms."""
    return [lv for lv in DERIVATION_LEVELS if (df["level"] == lv).any()]


def side_arms_in(df):
    """Side-arm levels actually present (e.g. L0), in config order."""
    return [lv for lv in SIDE_ARMS if (df["level"] == lv).any()]


def models_in(df):
    """Models actually present, in capability-ascending order (unknowns appended)."""
    present = set(df["model"].dropna())
    ordered = [m for m in MODEL_ORDER if m in present]
    extras = sorted(present - set(MODEL_ORDER))
    return ordered + extras


def _has_sql(df):
    return df.get("sql").fillna("").astype(str).str.strip().ne("") if "sql" in df else (df["error"].isna())


def is_infra_fail(df):
    """True infrastructure/API failure: errored AND produced no SQL. These are dropped."""
    return df["error"].notna() & ~_has_sql(df)


def clean_for_accuracy(df):
    """Rows that represent a completed model attempt. A row counts whenever the
    model produced SQL — even if that SQL failed to execute (binder/function
    errors): those are genuine WRONG answers and must count as incorrect, not be
    dropped. Only true infra failures (no SQL emitted) are excluded."""
    keep = df[~is_infra_fail(df)].copy()
    # An emitted-SQL row that carries an execution error is incorrect by definition.
    keep.loc[keep["error"].notna(), "correct"] = False
    keep["correct"] = keep["correct"].fillna(False).astype(bool)
    return keep


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def per_model_level(df_clean):
    """Aggregate clean rows to (model, level) -> stats. Returns long DataFrame."""
    recs = []
    for model in models_in(df_clean):
        for level in LEVEL_ORDER:
            sub = df_clean[(df_clean["model"] == model) & (df_clean["level"] == level)]
            if len(sub) == 0:
                continue
            # Reliability: per-question pass_rate = mean(correct) over that
            # question's N repeated samples, then aggregate across questions.
            #   always = fraction of questions solved in ALL N samples (reliable)
            #   any    = fraction of questions solved in AT LEAST ONE sample
            pass_rate = sub.groupby("id")["correct"].mean()
            n_q = len(pass_rate)
            reliable_always = float((pass_rate >= 1.0 - 1e-9).mean()) if n_q else np.nan
            reliable_any = float((pass_rate > 1e-9).mean()) if n_q else np.nan
            recs.append(
                {
                    "model": model,
                    "level": level,
                    "level_short": LEVEL_SHORT[level],
                    "n": len(sub),
                    "n_questions": n_q,
                    "accuracy": sub["correct"].mean(),
                    "reliable_always": reliable_always,
                    "reliable_any": reliable_any,
                    "mean_cost_usd": sub["cost_usd"].mean(),
                    "mean_context_tokens": sub["context_tokens"].mean(),
                    "mean_thinking_tokens": sub["thinking_tokens"].mean(),
                    "n_correct": int(sub["correct"].sum()),
                    "cost_sum": sub["cost_usd"].sum(),
                }
            )
    out = pd.DataFrame(recs)
    if not out.empty:
        out["level"] = pd.Categorical(out["level"], categories=LEVEL_ORDER, ordered=True)
        out = out.sort_values(["model", "level"]).reset_index(drop=True)
    return out


def model_color(model):
    return MODEL_COLOR.get(model, None)


def model_marker(model):
    return MODEL_MARKER.get(model, "o")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _x_axis_setup(ax, levels):
    xs = list(range(len(levels)))
    ax.set_xticks(xs)
    ax.set_xticklabels([LEVEL_SHORT[lv] for lv in levels])
    return {lv: i for i, lv in enumerate(levels)}


def fig_accuracy_vs_level(agg, deriv_levels, side_arms, out_path):
    """HEADLINE: accuracy vs derivation level, one line per model. Tests H1 (knee) & H2 (fan-out).

    The connected line covers the DERIVATION axis only (L1..L6). The L0 bare-schema
    side arm is deliberately INCOMPLETE and is NOT a point on this line: it is drawn
    as a separate marker to the LEFT of L1, with its own legend entry.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5))

    # x positions: derivation levels at 0..N-1; side arm(s) plotted to the LEFT at
    # negative positions, visually detached from the connected line.
    xmap = {lv: i for i, lv in enumerate(deriv_levels)}
    n_arms = len(side_arms)
    side_x = {sa: -1 - (n_arms - 1 - i) for i, sa in enumerate(side_arms)}  # e.g. L0 at -1

    # x ticks: side arms (left) + derivation levels.
    tick_pos = [side_x[sa] for sa in side_arms] + [xmap[lv] for lv in deriv_levels]
    tick_lab = [LEVEL_SHORT[sa] for sa in side_arms] + [LEVEL_SHORT[lv] for lv in deriv_levels]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab)

    # Connected line: derivation axis only.
    for model in models_in(agg):
        sub = agg[(agg["model"] == model) & (agg["level"].isin(deriv_levels))].sort_values("level")
        if sub.empty:
            continue
        xs = [xmap[lv] for lv in sub["level"]]
        ys = sub["accuracy"].values
        ax.plot(
            xs, ys, marker=model_marker(model), color=model_color(model),
            label=model, linewidth=2, markersize=7,
        )

    # Side arm(s): separate marker per model, NOT connected to the line.
    arm_label_done = False
    for sa in side_arms:
        for model in models_in(agg):
            r = agg[(agg["model"] == model) & (agg["level"] == sa)]
            if r.empty:
                continue
            label = None
            if not arm_label_done:
                label = f"{LEVEL_SHORT[sa]} bare schema (incomplete)"
                arm_label_done = True
            ax.scatter(
                [side_x[sa]], [float(r["accuracy"].iloc[0])],
                marker="*", s=170, color=model_color(model),
                edgecolors="black", linewidths=0.6, zorder=5, label=label,
            )
    # Visual separator between the side arm region and the derivation line.
    if side_arms:
        ax.axvline(-0.5, color="grey", linestyle=":", linewidth=1, alpha=0.6)

    # Annotate the canonical-ontology anchor level.
    if ANCHOR in xmap:
        ax.axvline(xmap[ANCHOR], color="grey", linestyle="--", linewidth=1, alpha=0.5)
        ax.annotate(
            "canonical ontology",
            xy=(xmap[ANCHOR], 0.02), xycoords=("data", "axes fraction"),
            rotation=90, va="bottom", ha="right", fontsize=8, color="grey",
        )

    ax.set_xlabel(
        f"Derivation distance: HIGH ({BASELINE_SHORT} shared ontology, max distance)  ->  "
        f"LOW ({TOP_SHORT} templated gold)"
    )
    ax.set_ylabel("Execution accuracy (fraction correct)")
    ax.set_title("Accuracy vs derivation distance, per model  (H1 knee, H2 fan-out)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(title="model", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def fig_cost_frontier(agg, deriv_levels, out_path):
    """Cost frontier: x = $ per correct answer, y = accuracy. One line per model. Tests H3.

    Derivation axis only (L1..L6); the incomplete L0 side arm is excluded.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for model in models_in(agg):
        sub = agg[(agg["model"] == model) & (agg["level"].isin(deriv_levels))].sort_values("level")
        # $ per correct answer; fall back to mean cost when no correct answers.
        xs, ys, labels = [], [], []
        for _, r in sub.iterrows():
            if r["n_correct"] > 0:
                cost_per_correct = r["cost_sum"] / r["n_correct"]
            else:
                cost_per_correct = np.nan  # undefined: cannot buy a correct answer here
            xs.append(cost_per_correct)
            ys.append(r["accuracy"])
            labels.append(r["level_short"])
        ax.plot(xs, ys, marker=model_marker(model), color=model_color(model),
                label=model, linewidth=1.5, markersize=7, alpha=0.9)
        for x, y, lab in zip(xs, ys, labels):
            if not np.isnan(x):
                ax.annotate(lab, (x, y), textcoords="offset points",
                            xytext=(4, 4), fontsize=7, color=model_color(model))

    ax.set_xlabel("Mean cost per correct answer (USD, = total cost / # correct)")
    ax.set_ylabel("Execution accuracy (fraction correct)")
    ax.set_title("Cost frontier: $ per correct answer vs accuracy, per model  (H3)")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(title="model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def fig_reliability_vs_level(agg, deriv_levels, out_path):
    """Answer reliability vs derivation level, one colour per model.

    For each (model, level) we compute, over its questions:
      always = fraction solved in ALL N samples  -> SOLID line + markers
      any    = fraction solved in AT LEAST ONE   -> FAINT DASHED line, same colour
    Derivation axis only (L1..L6); the incomplete L0 side arm is excluded.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5))
    xmap = _x_axis_setup(ax, deriv_levels)

    for model in models_in(agg):
        sub = agg[(agg["model"] == model) & (agg["level"].isin(deriv_levels))].sort_values("level")
        if sub.empty:
            continue
        xs = [xmap[lv] for lv in sub["level"]]
        color = model_color(model)
        # "always": reliable (solved N/N) — solid line with markers.
        ax.plot(xs, sub["reliable_always"].values, marker=model_marker(model),
                color=color, label=model, linewidth=2, markersize=7)
        # "any": solved >=1/N — faint dashed line, same colour, no marker.
        ax.plot(xs, sub["reliable_any"].values, color=color,
                linewidth=1.5, linestyle="--", alpha=0.45)

    # Linestyle legend note: build a second legend so colours stay = models.
    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color="grey", linewidth=2, linestyle="-", label="solid = solved in all N samples"),
        Line2D([0], [0], color="grey", linewidth=1.5, linestyle="--", alpha=0.6, label="dashed = solved in any sample"),
    ]

    ax.set_xlabel(
        f"Derivation distance: HIGH ({BASELINE_SHORT} shared ontology, max distance)  ->  "
        f"LOW ({TOP_SHORT} templated gold)"
    )
    ax.set_ylabel("Fraction of questions")
    ax.set_title(
        "Answer reliability vs derivation distance\n"
        "(solid = solved in all N samples; dashed = in any)"
    )
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    model_legend = ax.legend(title="model", fontsize=8, loc="lower right")
    ax.add_artist(model_legend)
    ax.legend(handles=style_handles, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def fig_context_tokens_vs_level(agg, deriv_levels, out_path):
    """Context tokens vs derivation level (length/cost covariate, roughly model-independent).

    Derivation axis only (L1..L6); the incomplete L0 side arm is excluded.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    xmap = _x_axis_setup(ax, deriv_levels)

    for model in models_in(agg):
        sub = agg[(agg["model"] == model) & (agg["level"].isin(deriv_levels))].sort_values("level")
        xs = [xmap[lv] for lv in sub["level"]]
        ys = sub["mean_context_tokens"].values
        ax.plot(xs, ys, marker=model_marker(model), color=model_color(model),
                label=model, linewidth=2, markersize=7)

    ax.set_xlabel(
        f"Derivation distance: HIGH ({BASELINE_SHORT} shared ontology, max distance)  ->  "
        f"LOW ({TOP_SHORT} templated gold)"
    )
    ax.set_ylabel("Mean context tokens")
    ax.set_title("Context length vs level  (cost covariate, ~model-independent)")
    ax.grid(True, alpha=0.3)
    ax.legend(title="model")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def knee_level(sub):
    """Lowest derivation level reaching >=90% of this model's max accuracy.

    Computed over the DERIVATION axis only (L1..L6); the incomplete L0 side arm
    is excluded. "Lowest level" here means the highest-derivation-distance rung (L1
    first), so the knee is the cheapest rung that already gets you ~there.
    Returns a short label or None.
    """
    sub = sub[sub["level"].isin(DERIVATION_LEVELS)]
    if sub.empty:
        return None
    max_acc = sub["accuracy"].max()
    if max_acc <= 0:
        return None
    threshold = 0.9 * max_acc
    sub_sorted = sub.sort_values("level")
    for _, r in sub_sorted.iterrows():
        if r["accuracy"] >= threshold:
            return r["level_short"]
    return None


def acc_at(sub, level):
    """Accuracy at a given full level name, or None if absent."""
    r = sub[sub["level"] == level]
    if r.empty:
        return None
    return float(r["accuracy"].iloc[0])


def fmt_pct(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f}%"


def fmt_num(x, nd=0):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:.{nd}f}"


def fmt_usd(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"${x:.5f}"


def build_summary(df, df_clean, agg, run_name):
    """Return the full summary.md text."""
    levels = levels_in(df)
    models = models_in(df)
    lines = []
    A = lines.append

    A(f"# Derivation-distance results — `{run_name}`\n")
    n_total = len(df)
    n_infra = int(is_infra_fail(df).sum())
    n_exec = int((df["error"].notna() & ~is_infra_fail(df)).sum())
    A(f"Total run records: **{n_total}**. Infra/API failures (no SQL emitted, excluded): "
      f"**{n_infra}**. SQL-execution failures (model wrote invalid SQL — **counted as incorrect**): "
      f"**{n_exec}**.\n")
    A("Accuracy = fraction of completed attempts (model emitted SQL) whose result matches the gold set "
      "(order-insensitive, float-rounded).\n")

    # -- Master table -------------------------------------------------------
    A("## Per (model x level) stats\n")
    A("Accuracy = mean correctness over all samples. reliable (N/N) = fraction of "
      "questions solved in *every* one of their N repeated samples (the stricter "
      "consistency number).\n")
    A("| model | level | n | accuracy | reliable (N/N) | mean cost (USD) | mean context tok |")
    A("|---|---|---:|---:|---:|---:|---:|")
    for model in models:
        sub = agg[agg["model"] == model]
        for level in LEVEL_ORDER:
            r = sub[sub["level"] == level]
            if r.empty:
                continue
            r = r.iloc[0]
            A(f"| {model} | {LEVEL_SHORT[level]} | {int(r['n'])} | {fmt_pct(r['accuracy'])} "
              f"| {fmt_pct(r['reliable_always'])} | {fmt_usd(r['mean_cost_usd'])} "
              f"| {fmt_num(r['mean_context_tokens'])} |")
    A("")

    # -- Per-model knee + landmark accuracies -------------------------------
    A("## Per-model: knee level and landmark accuracies\n")
    A(f"Knee = lowest derivation level (highest effort, {BASELINE_SHORT} first) reaching "
      f">=90% of that model's own max accuracy, over the {BASELINE_SHORT}..{TOP_SHORT} axis. "
      f"{ANCHOR_SHORT} = `{ANCHOR}` = the canonical published ontology. "
      f"All three landmark columns are derivation-axis levels (complete); the incomplete "
      f"{LEVEL_SHORT[SIDE_ARMS[0]]} side arm is reported separately below.\n")
    A(f"| model | knee | acc @ {BASELINE_SHORT} (L1) | "
      f"acc @ {ANCHOR_SHORT} (canonical) | acc @ {TOP_SHORT} | max acc |")
    A("|---|---|---:|---:|---:|---:|")
    landmark = {}
    for model in models:
        sub = agg[agg["model"] == model]
        sub_deriv = sub[sub["level"].isin(DERIVATION_LEVELS)]
        knee = knee_level(sub) or "n/a"
        a_base = acc_at(sub, BASELINE)
        a_anchor = acc_at(sub, ANCHOR)
        a_top = acc_at(sub, TOP)
        amax = float(sub_deriv["accuracy"].max()) if not sub_deriv.empty else None
        landmark[model] = {"knee": knee, "base": a_base, "anchor": a_anchor,
                           "top": a_top, "max": amax}
        A(f"| {model} | {knee} | {fmt_pct(a_base)} | {fmt_pct(a_anchor)} | "
          f"{fmt_pct(a_top)} | {fmt_pct(amax)} |")
    A("")

    # -- H2 read: L1 -> L4 gain --------------------------------------------
    A("## H2 read — capability interaction (the headline)\n")
    A(f"H2 predicts weaker models gain *more* accuracy as derivation distance drops "
      f"(steeper curve). We measure the accuracy gain from {BASELINE_SHORT} (shared ontology, max effort, "
      f"max effort) to {ANCHOR_SHORT} (canonical) per model. Both rungs are complete, so this "
      f"gain is pure derivation distance. If the pattern holds, the gain should be ordered "
      f"haiku > sonnet > opus.\n")
    A(f"| model | acc @ {BASELINE_SHORT} | acc @ {ANCHOR_SHORT} | "
      f"gain ({ANCHOR_SHORT} - {BASELINE_SHORT}) |")
    A("|---|---:|---:|---:|")
    gains = {}
    for model in models:
        lm = landmark[model]
        if lm["base"] is not None and lm["anchor"] is not None:
            gain = lm["anchor"] - lm["base"]
            gains[model] = gain
            A(f"| {model} | {fmt_pct(lm['base'])} | {fmt_pct(lm['anchor'])} | "
              f"{gain * 100:+.0f} pts |")
        else:
            A(f"| {model} | {fmt_pct(lm['base'])} | {fmt_pct(lm['anchor'])} | n/a |")
    A("")
    # Verdict on the ordering.
    cap_rank = {m: i for i, m in enumerate(MODEL_ORDER)}  # haiku weakest -> opus strongest
    ranked_models = [m for m in models if m in gains]
    ranked_models.sort(key=lambda m: cap_rank.get(m, 99))  # weak -> strong
    if len(ranked_models) >= 2:
        gseq = [gains[m] for m in ranked_models]
        holds = all(gseq[i] >= gseq[i + 1] - 1e-9 for i in range(len(gseq) - 1))
        ordering = " > ".join(f"{m} ({gains[m] * 100:+.0f} pts)" for m in ranked_models)
        if holds and any(abs(g) > 1e-9 for g in gseq):
            A(f"**H2 pattern holds** in this run: weaker-to-stronger gains are {ordering}, "
              "i.e. the weaker model gains at least as much as the stronger one.\n")
        elif all(abs(g) < 1e-9 for g in gseq):
            A(f"**Inconclusive**: all {BASELINE_SHORT}->{ANCHOR_SHORT} gains are ~0 here "
              f"({ordering}). Likely a ceiling/floor effect on this (small?) set — "
              "nobody has room to move.\n")
        else:
            A(f"**H2 pattern does NOT hold** in this run: weaker-to-stronger gains are {ordering}, "
              "which is not monotonically decreasing with capability. Worth a closer look on the "
              "full set.\n")
    else:
        A(f"_Not enough models with both {BASELINE_SHORT} and {ANCHOR_SHORT} present "
          "to evaluate H2._\n")

    # -- Completeness side-arm (L0) ----------------------------------------
    side_arms = side_arms_in(df)
    if side_arms:
        sa = side_arms[0]
        sa_short = LEVEL_SHORT[sa]
        A(f"## Completeness side-arm ({sa_short} bare schema)\n")
        A(f"| model | n | acc @ {sa_short} (names only) | acc @ {BASELINE_SHORT} "
          f"(full schema+desc) | gap ({BASELINE_SHORT} - {sa_short}) |")
        A("|---|---:|---:|---:|---:|")
        for model in models:
            sub = agg[agg["model"] == model]
            r = sub[sub["level"] == sa]
            n = int(r["n"].iloc[0]) if not r.empty else 0
            a_sa = acc_at(sub, sa)
            a_base = acc_at(sub, BASELINE)
            if a_sa is not None and a_base is not None:
                gap = f"{(a_base - a_sa) * 100:+.0f} pts"
            else:
                gap = "n/a"
            A(f"| {model} | {n} | {fmt_pct(a_sa)} | {fmt_pct(a_base)} | {gap} |")
        A("")
        A(f"`{sa}` is a SIDE ARM, not a rung on the derivation-distance axis: it gives bare "
          f"column names with NONE of the required facts, so it is deliberately incomplete for "
          f"most questions. Its gap vs {BASELINE_SHORT} therefore measures how much a model can "
          f"recover the missing conventions from column names alone — a completeness effect, "
          f"NOT derivation distance.\n")

    # -- Quotable cross-model line -----------------------------------------
    A("## Quotable line (if one exists)\n")
    quote = find_quotable(agg, models)
    if quote:
        A(quote + "\n")
    else:
        A("_No clean cross-model equivalence found in this run (e.g. on the smoke set "
          "everything saturates at ~100%, so there is nothing to quote)._\n")

    # -- Error breakdown ----------------------------------------------------
    A("## Error breakdown per model/level\n")
    A("SQL-exec failures are counted as incorrect (model wrote invalid SQL); infra failures are dropped.\n")
    if int(df["error"].notna().sum()) == 0:
        A("No errored runs in this file.\n")
    else:
        err = df[df["error"].notna()]
        A("| model | level | n errored (exec+infra) |")
        A("|---|---|---:|")
        grp = err.groupby(["model", "level"], observed=True).size().reset_index(name="n_err")
        grp["lv_idx"] = grp["level"].map({lv: i for i, lv in enumerate(LEVEL_ORDER)})
        grp = grp.sort_values(["model", "lv_idx"])
        for _, r in grp.iterrows():
            A(f"| {r['model']} | {LEVEL_SHORT.get(r['level'], r['level'])} | {int(r['n_err'])} |")
        A("")

    A("---")
    A(f"_Generated by `src/analyze.py` from `results/runs/{run_name}.jsonl`. "
      f"Levels present: {', '.join(LEVEL_SHORT[lv] for lv in levels)}. "
      f"Models present: {', '.join(models)}._")
    return "\n".join(lines) + "\n"


def find_quotable(agg, models):
    """Find a line like 'Opus at L<i> matches Haiku at L<j>' (strong needs less effort).

    For each ordered (strong, weak) model pair, find the lowest level i where the
    strong model's accuracy is >= the weak model's *max* accuracy, and report the
    lowest level j where the weak model first reaches that same accuracy. Quote it
    only when the strong model needs an earlier (lower) level than the weak model.
    """
    cap_rank = {m: i for i, m in enumerate(MODEL_ORDER)}
    present = [m for m in models if m in cap_rank]
    present.sort(key=lambda m: cap_rank[m])
    best = None
    for wi in range(len(present)):
        for si in range(wi + 1, len(present)):
            weak, strong = present[wi], present[si]
            # Derivation axis only; the incomplete L0 side arm is excluded.
            ws = agg[(agg["model"] == weak) & (agg["level"].isin(DERIVATION_LEVELS))].sort_values("level")
            ss = agg[(agg["model"] == strong) & (agg["level"].isin(DERIVATION_LEVELS))].sort_values("level")
            if ws.empty or ss.empty:
                continue
            weak_max = ws["accuracy"].max()
            if weak_max <= 0:
                continue
            # lowest level where weak reaches its own max
            weak_lv = None
            for _, r in ws.iterrows():
                if r["accuracy"] >= weak_max - 1e-9:
                    weak_lv = r
                    break
            # lowest level where strong reaches >= weak_max
            strong_lv = None
            for _, r in ss.iterrows():
                if r["accuracy"] >= weak_max - 1e-9:
                    strong_lv = r
                    break
            if weak_lv is None or strong_lv is None:
                continue
            si_idx = LEVEL_ORDER.index(strong_lv["level"])
            wi_idx = LEVEL_ORDER.index(weak_lv["level"])
            if si_idx < wi_idx:  # strong gets there with less context
                gap = wi_idx - si_idx
                cand = (
                    gap,
                    f"**{strong.capitalize()} at {strong_lv['level_short']} matches "
                    f"{weak.capitalize()} at {weak_lv['level_short']}** "
                    f"(both at {fmt_pct(weak_max)} accuracy) — the stronger model reaches the "
                    f"same answer quality with {gap} fewer rung(s) of hand-holding.",
                )
                if best is None or cand[0] > best[0]:
                    best = cand
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze derivation-distance run results.")
    parser.add_argument("--run", required=True, help="run name, e.g. 'pilot' or 'smoke'")
    args = parser.parse_args()

    df = load_runs(args.run)
    df_clean = clean_for_accuracy(df)
    agg = per_model_level(df_clean)

    fig_dir = os.path.join(config.BASE, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    deriv_levels = derivation_levels_in(df)   # L1..L6 (the derivation-distance axis)
    side_arms = side_arms_in(df)              # L0 (incomplete side arm, reported separately)

    if agg.empty:
        print("WARNING: no error-free rows to aggregate; figures will be empty.")
    if not deriv_levels:
        print("WARNING: no derivation-axis levels (L1..L6) present; line figures will be empty.")

    paths = {
        "accuracy": os.path.join(fig_dir, f"{args.run}_accuracy_vs_level.png"),
        "cost": os.path.join(fig_dir, f"{args.run}_cost_frontier.png"),
        "reliability": os.path.join(fig_dir, f"{args.run}_reliability_vs_level.png"),
        "context": os.path.join(fig_dir, f"{args.run}_context_tokens_vs_level.png"),
    }
    fig_accuracy_vs_level(agg, deriv_levels, side_arms, paths["accuracy"])
    fig_cost_frontier(agg, deriv_levels, paths["cost"])
    fig_reliability_vs_level(agg, deriv_levels, paths["reliability"])
    fig_context_tokens_vs_level(agg, deriv_levels, paths["context"])

    summary = build_summary(df, df_clean, agg, args.run)
    summary_path = os.path.join(config.BASE, "results", "summary.md")
    with open(summary_path, "w") as fh:
        fh.write(summary)

    print(f"Loaded {len(df)} records ({len(df_clean)} error-free) from run '{args.run}'.")
    print("Wrote figures:")
    for p in paths.values():
        print(f"  {p}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
