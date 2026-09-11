#!/usr/bin/env python3
"""
AS01 Part A: Dataset datasheet / profiling / feature engineering
Dataset: UCI Drug Review Dataset (Drugs.com), Hugging Face mirror lewtun/drug-reviews

Run:
    python part_a_profile.py

Inputs:
    data/raw/train.jsonl, data/raw/test.jsonl

Outputs (outputs/part_a/):
    tables/*.csv, *.md        descriptive statistics tables
    figures/*.png             visualizations
    summary.json              machine-readable summary of everything below
    data/clean/reviews_clean.parquet   cleaned dataset with engineered features
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "outputs" / "part_a"
TAB = OUT / "tables"
FIG = OUT / "figures"
for p in (CLEAN, TAB, FIG):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
rng = np.random.default_rng(SEED)

# ---- palette (single sequential hue for magnitude; fixed categorical order) ----
BLUE = "#2a78d6"
BLUE_LIGHT = "#86b6ef"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#e34948"
GRAY = "#52514e"
CAT = [BLUE, ORANGE, AQUA, YELLOW]
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.edgecolor": "#c3c2b7", "axes.titleweight": "bold",
})

summary: dict = {}


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> None:
    df.to_csv(TAB / f"{name}.csv", index=index)
    with open(TAB / f"{name}.md", "w") as f:
        f.write(df.to_markdown(index=index))


# =========================================================================
# 1. LOAD
# =========================================================================
train = pd.read_json(RAW / "train.jsonl", lines=True)
test = pd.read_json(RAW / "test.jsonl", lines=True)
train["split"] = "train"
test["split"] = "test"
raw = pd.concat([train, test], ignore_index=True)
raw = raw.rename(columns={"Unnamed: 0": "uniqueID"})  # matches the UCI column name
print(f"Loaded {len(raw):,} rows ({len(train):,} train / {len(test):,} test)")

summary["n_observations"] = int(len(raw))
summary["n_train"] = int(len(train))
summary["n_test"] = int(len(test))
summary["n_features_original"] = int(raw.shape[1] - 1)  # exclude our split col

# =========================================================================
# 2. RAW PROFILE: dtypes, missing, unique
# =========================================================================
orig_cols = ["uniqueID", "drugName", "condition", "review", "rating", "date", "usefulCount"]
prof = pd.DataFrame({
    "dtype": raw[orig_cols].dtypes.astype(str),
    "missing": raw[orig_cols].isna().sum(),
    "missing_pct": (raw[orig_cols].isna().mean() * 100).round(3),
    "unique": raw[orig_cols].nunique(),
})
save_table(prof, "01_feature_profile")
print(prof)
summary["feature_profile"] = prof.to_dict(orient="index")

# =========================================================================
# 3. DATA QUALITY CHECKS
# =========================================================================
span_mask = raw["condition"].astype(str).str.contains("</span>", regex=False)
html_entity_mask = raw["review"].astype(str).str.contains(r"&#?\w+;", regex=True)
QUOTE_RE = re.compile(r'^"(.*)"$', re.S)
quoted_mask = raw["review"].map(lambda t: bool(QUOTE_RE.match(str(t))))
dup_mask = raw.duplicated(subset=["drugName", "condition", "review"], keep="first")
dup_review_only = raw.duplicated(subset=["review"], keep="first")
dq = pd.DataFrame({
    "check": [
        "condition contains scraped HTML '</span>' artifact",
        "review contains HTML entities (&#039; &amp; ...)",
        "review wrapped in literal double quotes",
        "exact duplicate (drugName, condition, review)",
        "duplicate review text (any drug/condition)",
        "condition missing",
        "review shorter than 5 words",
    ],
    "count": [
        int(span_mask.sum()), int(html_entity_mask.sum()), int(quoted_mask.sum()),
        int(dup_mask.sum()), int(dup_review_only.sum()), int(raw["condition"].isna().sum()),
        int((raw["review"].astype(str).str.split().str.len() < 5).sum()),
    ],
})
# how many duplicated review texts are the SAME review listed under different drug names
# (Drugs.com lists one review under both the brand and the generic name)
grp = raw.groupby("review").agg(n=("drugName", "size"), n_drugs=("drugName", "nunique"),
                                n_cond=("condition", "nunique"))
multi = grp[grp["n"] > 1]
dq.loc[len(dq)] = ["review texts appearing >1 times (distinct texts)", int(len(multi))]
dq.loc[len(dq)] = ["  of which listed under >1 drugName (brand/generic)", int((multi["n_drugs"] > 1).sum())]
dq.loc[len(dq)] = ["  of which listed under >1 condition", int((multi["n_cond"] > 1).sum())]
dq["pct"] = (dq["count"] / len(raw) * 100).round(2)
save_table(dq, "02_data_quality_checks", index=False)
print(dq)
summary["data_quality"] = dq.to_dict(orient="records")

# =========================================================================
# 4. NUMERIC DESCRIPTIVE STATISTICS (raw)
# =========================================================================
def describe_numeric(s: pd.Series) -> dict:
    s = s.dropna()
    q1, q2, q3 = s.quantile([0.25, 0.5, 0.75])
    return {
        "count": int(s.count()), "min": float(s.min()), "max": float(s.max()),
        "mean": float(s.mean()), "median": float(q2), "std": float(s.std()),
        "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
        "range": float(s.max() - s.min()),
        "skewness": float(stats.skew(s)), "kurtosis_excess": float(stats.kurtosis(s)),
    }


num_raw = pd.DataFrame({c: describe_numeric(raw[c]) for c in ["rating", "usefulCount"]}).T
save_table(num_raw.round(3), "03_numeric_stats_raw")
print(num_raw.round(3))

# =========================================================================
# 5. CATEGORICAL / TEXT FREQUENCIES (raw)
# =========================================================================
top_drugs = raw["drugName"].value_counts().head(20)
top_conds = raw["condition"].value_counts().head(20)
rating_dist = raw["rating"].value_counts().sort_index()
rating_dist_df = pd.DataFrame({"count": rating_dist, "pct": (rating_dist / len(raw) * 100).round(2)})
save_table(top_drugs.to_frame("count"), "04_top20_drugs")
save_table(top_conds.to_frame("count"), "05_top20_conditions")
save_table(rating_dist_df, "06_rating_distribution")

# long-tail description of categorical features
def tail_stats(s: pd.Series) -> dict:
    vc = s.value_counts()
    return {
        "unique": int(vc.size),
        "top1_share_pct": round(float(vc.iloc[0] / vc.sum() * 100), 2),
        "top10_share_pct": round(float(vc.head(10).sum() / vc.sum() * 100), 2),
        "top50_share_pct": round(float(vc.head(50).sum() / vc.sum() * 100), 2),
        "values_with_<=5_rows": int((vc <= 5).sum()),
        "values_with_1_row": int((vc == 1).sum()),
        "median_rows_per_value": float(vc.median()),
    }


cat_tail = pd.DataFrame({"drugName": tail_stats(raw["drugName"]),
                         "condition": tail_stats(raw["condition"].dropna())}).T
save_table(cat_tail, "07_categorical_longtail")
print(cat_tail)
summary["categorical_longtail"] = cat_tail.to_dict(orient="index")

# =========================================================================
# 6. CLEANING
# =========================================================================
clean = raw.copy()
clean["review"] = clean["review"].astype(str).map(html.unescape).str.strip()
clean["review"] = clean["review"].map(lambda t: QUOTE_RE.sub(r"\1", t).strip())
clean["condition_clean"] = clean["condition"].astype("string")
clean.loc[span_mask, "condition_clean"] = pd.NA          # scraped junk -> missing
if not pd.api.types.is_datetime64_any_dtype(clean["date"]):
    clean["date"] = pd.to_datetime(clean["date"], format="%B %d, %Y", errors="coerce")

n_before = len(clean)
clean = clean[~clean.duplicated(subset=["drugName", "condition", "review"], keep="first")]
n_after_dedup = len(clean)
clean = clean[clean["review"].str.split().str.len() >= 3]
n_after_short = len(clean)
summary["cleaning"] = {
    "rows_raw": n_before,
    "rows_after_exact_dedup": n_after_dedup,
    "rows_after_drop_reviews_under_3_words": n_after_short,
    "condition_set_missing_due_to_html_artifact": int(span_mask.sum()),
}
print(summary["cleaning"])

# =========================================================================
# 7. FEATURE ENGINEERING
# =========================================================================
# (a) text length features
clean["review_char_len"] = clean["review"].str.len()
clean["review_word_len"] = clean["review"].str.split().str.len()
clean["review_sentence_count"] = clean["review"].str.count(r"[.!?]+(?:\s|$)").clip(lower=1)

# (b) rating -> sentiment label (target for future SFT)
def sentiment(r: float) -> str:
    if r <= 4:
        return "NEGATIVE"
    if r <= 6:
        return "NEUTRAL"
    return "POSITIVE"


clean["sentiment_label"] = clean["rating"].map(sentiment)

# (c) date -> year
clean["review_year"] = clean["date"].dt.year

# (d) side-effect mention flag (keyword based, transparent)
SIDE_EFFECT_RE = re.compile(
    r"\b(side[- ]?effects?|nausea|nauseous|dizz(y|iness)|headache|vomit|weight gain|"
    r"insomnia|drowsy|drowsiness|fatigue|rash|constipation|diarrhea|dry mouth|"
    r"anxiety|sweating|cramps?|bleeding|swelling|itch(y|ing))\b", re.I)
clean["side_effect_flag"] = clean["review"].map(lambda t: int(bool(SIDE_EFFECT_RE.search(t))))

# (e) popularity: how many reviews exist for the same drug / condition
clean["drug_review_count"] = clean.groupby("drugName")["review"].transform("size")
clean["condition_review_count"] = clean.groupby("condition_clean")["review"].transform("size")

# (f) chronic-condition flag (target population of Scenario 01)
CHRONIC = {
    "High Blood Pressure", "Diabetes, Type 2", "High Cholesterol", "Asthma",
    "COPD", "Rheumatoid Arthritis", "Osteoarthritis", "Hypothyroidism",
    "Atrial Fibrillation", "Heart Failure", "Chronic Pain", "Epilepsy",
    "Crohn's Disease", "Ulcerative Colitis", "Psoriasis", "Multiple Sclerosis",
    "Osteoporosis", "Diabetes, Type 1", "Fibromyalgia", "Gout", "Angina",
    "Left Ventricular Dysfunction", "Chronic Kidney Disease", "Hepatitis C",
    "Parkinson's Disease", "Alzheimer's Disease", "Migraine", "Bipolar Disorde",
    "Schizophrenia", "Depression", "Anxiety", "Major Depressive Disorde",
}
clean["is_chronic_condition"] = clean["condition_clean"].isin(CHRONIC).astype(int)

# (g) Traffic Light Protocol label (rule based)
RED_RE = re.compile(
    r"\b(suicid\w*|kill(ing)? myself|end(ing)? my life|self[- ]harm|cutting myself|overdos\w*|"
    r"hiv|abortion|miscarriage|raped?|sexual(ly)? (assault|abuse)|"
    r"heroin|cocaine|meth|relapsed?|rehab)\b", re.I)
RED_RE_CS = re.compile(r"\bAIDS\b")   # case-sensitive: avoid matching 'sleep-aids'
RED_CONDITIONS = {
    "HIV Infection", "Opiate Dependence", "Alcohol Dependence", "Abortion",
    "Opiate Withdrawal", "Schizophrenia", "Schizoaffective Disorde", "Cocaine Dependence",
    "Substance Abuse", "Hepatitis C", "Herpes Simplex", "Genital Herpes",
}
AMBER_RE = re.compile(
    r"(\bmy (son|daughter|husband|wife|mom|mother|dad|father|child|kid|kids|boyfriend|girlfriend|"
    r"partner|grandmother|grandfather|grandma|grandpa|niece|nephew|brother|sister)\b"
    r"|\b\d{1,2}[- ]?(years?|yrs?)[- ]old\b"
    r"|\b\d{1,2} ?y/?o\b"
    r"|\bi(?:'m| am) (?:a |an |now )?\d{2}\b"
    r"|\bage(?:d)? \d{1,2}\b"
    r"|\bpregnan\w*|\bbreastfeed\w*"
    r"|\bdr\.? [A-Z][a-z]{2,}\b)", re.I)


def tlp_label(row) -> str:
    text = row["review"]
    if RED_RE.search(text) or RED_RE_CS.search(text) or (row["condition_clean"] in RED_CONDITIONS):
        return "TLP:RED"
    if AMBER_RE.search(text):
        return "TLP:AMBER"
    if row["review_word_len"] <= 8 and not re.search(r"\b(i|my|me)\b", text, re.I):
        return "TLP:CLEAR"
    return "TLP:GREEN"


clean["tlp_label"] = clean.apply(tlp_label, axis=1)

clean.to_parquet(CLEAN / "reviews_clean.parquet", index=False)
print("Saved cleaned dataset:", CLEAN / "reviews_clean.parquet", clean.shape)
summary["n_features_after_engineering"] = int(clean.shape[1])
summary["engineered_features"] = [
    "condition_clean", "review_char_len", "review_word_len", "review_sentence_count",
    "sentiment_label", "review_year", "side_effect_flag", "drug_review_count",
    "condition_review_count", "is_chronic_condition", "tlp_label",
]

# TLP examples table: one shortest-but-informative example per label
tlp_counts = clean["tlp_label"].value_counts()
tlp_rows = []
for lab in ["TLP:CLEAR", "TLP:GREEN", "TLP:AMBER", "TLP:RED"]:
    sub = clean[clean["tlp_label"] == lab]
    sub = sub[(sub["review_word_len"] >= 6) & (sub["review_word_len"] <= 45)]
    ex = sub.sample(1, random_state=SEED).iloc[0] if len(sub) else None
    tlp_rows.append({
        "tlp_label": lab,
        "count": int(tlp_counts.get(lab, 0)),
        "pct": round(float(tlp_counts.get(lab, 0) / len(clean) * 100), 2),
        "example_drug": ex["drugName"] if ex is not None else "",
        "example_condition": ex["condition_clean"] if ex is not None else "",
        "example_review": ex["review"] if ex is not None else "",
    })
tlp_df = pd.DataFrame(tlp_rows)
save_table(tlp_df, "08_tlp_labels", index=False)
print(tlp_df[["tlp_label", "count", "pct"]])

# =========================================================================
# 8. DESCRIPTIVE STATS AFTER CLEANING + NEW FEATURES
# =========================================================================
num_cols = ["rating", "usefulCount", "review_char_len", "review_word_len",
            "review_sentence_count", "drug_review_count", "condition_review_count", "review_year"]
num_clean = pd.DataFrame({c: describe_numeric(clean[c]) for c in num_cols}).T
save_table(num_clean.round(3), "09_numeric_stats_clean_and_engineered")
print(num_clean.round(2))

cat_cols = ["sentiment_label", "side_effect_flag", "is_chronic_condition", "tlp_label", "split"]
cat_freq = []
for c in cat_cols:
    vc = clean[c].value_counts()
    for k, v in vc.items():
        cat_freq.append({"feature": c, "value": k, "count": int(v), "pct": round(v / len(clean) * 100, 2)})
cat_freq_df = pd.DataFrame(cat_freq)
save_table(cat_freq_df, "10_categorical_freq_engineered", index=False)

# comparison: sentiment label vs rating; side_effect_flag vs rating; chronic vs rating
cmp1 = clean.groupby("sentiment_label")["rating"].agg(["count", "min", "max", "mean"]).round(2)
cmp2 = clean.groupby("side_effect_flag")["rating"].agg(["count", "mean", "median", "std"]).round(3)
cmp3 = clean.groupby("is_chronic_condition")["rating"].agg(["count", "mean", "median", "std"]).round(3)
cmp4 = clean.groupby("tlp_label")["rating"].agg(["count", "mean", "median"]).round(3)
cmp5 = pd.crosstab(clean["side_effect_flag"], clean["sentiment_label"], normalize="index").round(3) * 100
save_table(cmp1, "11_cmp_sentiment_vs_rating")
save_table(cmp2, "12_cmp_sideeffect_vs_rating")
save_table(cmp3, "13_cmp_chronic_vs_rating")
save_table(cmp4, "14_cmp_tlp_vs_rating")
save_table(cmp5, "15_cmp_sideeffect_vs_sentiment_pct")
corr = clean[["rating", "usefulCount", "review_word_len", "review_sentence_count",
              "drug_review_count", "side_effect_flag"]].corr(method="spearman").round(3)
save_table(corr, "16_spearman_correlations")
print(corr)
summary["comparisons"] = {
    "sideeffect_vs_rating": cmp2.to_dict(orient="index"),
    "chronic_vs_rating": cmp3.to_dict(orient="index"),
    "spearman": corr.to_dict(),
}

# =========================================================================
# 9. NORMALITY TESTS
# =========================================================================
norm_rows = []
for c in ["rating", "usefulCount", "review_word_len", "review_char_len", "review_sentence_count",
          "drug_review_count"]:
    s = clean[c].dropna().astype(float)
    samp = s.sample(5000, random_state=SEED) if len(s) > 5000 else s
    sh_w, sh_p = stats.shapiro(samp)
    da_k2, da_p = stats.normaltest(s)
    log_s = np.log1p(samp)
    sh_w_log, sh_p_log = stats.shapiro(log_s)
    norm_rows.append({
        "feature": c, "skewness": round(float(stats.skew(s)), 3),
        "kurtosis_excess": round(float(stats.kurtosis(s)), 3),
        "shapiro_W_n5000": round(float(sh_w), 4), "shapiro_p": float(f"{sh_p:.3g}"),
        "dagostino_K2": round(float(da_k2), 1), "dagostino_p": float(f"{da_p:.3g}"),
        "shapiro_W_log1p": round(float(sh_w_log), 4),
        "verdict": "non-normal" if sh_p < 0.05 else "approx. normal",
    })
norm_df = pd.DataFrame(norm_rows)
save_table(norm_df, "17_normality_tests", index=False)
print(norm_df)
summary["normality"] = norm_df.to_dict(orient="records")

# =========================================================================
# 10. FIGURES
# =========================================================================
def finish(fig, name, note=None):
    if note:
        fig.text(0.01, 0.005, note, fontsize=7, color=GRAY)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


# F1 rating histogram
fig, ax = plt.subplots(figsize=(6, 3.4))
ax.bar(rating_dist.index, rating_dist.values, color=BLUE, width=0.7)
ax.set_xticks(range(1, 11))
ax.set_xlabel("rating (1-10)")
ax.set_ylabel("number of reviews")
ax.set_title("F1. Rating distribution (raw, n=%s)" % f"{len(raw):,}")
for x, y in zip(rating_dist.index, rating_dist.values):
    if y > 0.12 * rating_dist.max():
        ax.text(x, y, f"{y/len(raw)*100:.0f}%", ha="center", va="bottom", fontsize=7, color=GRAY)
finish(fig, "F1_rating_hist")

# F2 usefulCount histogram, linear and log1p
fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
axes[0].hist(clean["usefulCount"], bins=60, color=BLUE)
axes[0].set_title("F2a. usefulCount (linear)")
axes[0].set_xlabel("usefulCount")
axes[0].set_ylabel("reviews")
axes[1].hist(np.log1p(clean["usefulCount"]), bins=40, color=BLUE)
axes[1].set_title("F2b. log1p(usefulCount)")
axes[1].set_xlabel("log(1 + usefulCount)")
finish(fig, "F2_usefulcount_hist")

# F3 boxplots of numeric features
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
for ax, c in zip(axes, ["rating", "usefulCount", "review_word_len"]):
    ax.boxplot(clean[c].dropna(), widths=0.5,
               boxprops=dict(color=BLUE), medianprops=dict(color=ORANGE, linewidth=2),
               whiskerprops=dict(color=GRAY), capprops=dict(color=GRAY),
               flierprops=dict(marker=".", markersize=2, markeredgecolor=BLUE_LIGHT, alpha=0.4))
    ax.set_title(f"F3. {c}")
    ax.set_xticks([])
finish(fig, "F3_boxplots")

# F4 top-20 conditions
fig, ax = plt.subplots(figsize=(7, 5))
tc = top_conds[::-1]
ax.barh(tc.index, tc.values, color=BLUE)
ax.set_xlabel("number of reviews")
ax.set_title("F4. Top-20 conditions (raw)")
finish(fig, "F4_top_conditions", note="Note the scraped artifact value '...</span> users found this comment helpful.'")

# F5 top-20 drugs
fig, ax = plt.subplots(figsize=(7, 5))
td = top_drugs[::-1]
ax.barh(td.index, td.values, color=BLUE)
ax.set_xlabel("number of reviews")
ax.set_title("F5. Top-20 drugs (raw)")
finish(fig, "F5_top_drugs")

# F6 review length density
fig, ax = plt.subplots(figsize=(6, 3.4))
wl = clean["review_word_len"]
ax.hist(wl, bins=80, range=(0, 400), color=BLUE, density=True)
ax.axvline(wl.median(), color=ORANGE, linewidth=1.5)
ax.text(wl.median() + 5, ax.get_ylim()[1] * 0.9, f"median = {wl.median():.0f} words", color=ORANGE, fontsize=8)
ax.set_xlabel("review length (words)")
ax.set_ylabel("density")
ax.set_title("F6. Review length distribution")
finish(fig, "F6_review_length_density")

# F7 reviews per year
yr = clean["review_year"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(yr.index.astype(int), yr.values, color=BLUE)
ax.set_xlabel("year")
ax.set_ylabel("reviews")
ax.set_title("F7. Reviews per year")
finish(fig, "F7_reviews_per_year")

# F8 rating by top-8 conditions (boxplot)
top8 = clean["condition_clean"].value_counts().head(8).index.tolist()
fig, ax = plt.subplots(figsize=(8, 3.6))
data = [clean.loc[clean["condition_clean"] == c, "rating"] for c in top8]
ax.boxplot(data, widths=0.5, boxprops=dict(color=BLUE), medianprops=dict(color=ORANGE, linewidth=2),
           whiskerprops=dict(color=GRAY), capprops=dict(color=GRAY), flierprops=dict(marker=""))
ax.set_xticks(range(1, len(top8) + 1))
ax.set_xticklabels([c[:18] for c in top8], rotation=25, ha="right")
ax.set_ylabel("rating")
ax.set_title("F8. Rating by top-8 conditions")
finish(fig, "F8_rating_by_condition")

# F9 engineered: sentiment label counts and TLP counts
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
sl = clean["sentiment_label"].value_counts().reindex(["NEGATIVE", "NEUTRAL", "POSITIVE"])
axes[0].bar(sl.index, sl.values, color=BLUE)
axes[0].set_title("F9a. sentiment_label (from rating)")
for i, v in enumerate(sl.values):
    axes[0].text(i, v, f"{v/len(clean)*100:.1f}%", ha="center", va="bottom", fontsize=8, color=GRAY)
tl = clean["tlp_label"].value_counts().reindex(["TLP:CLEAR", "TLP:GREEN", "TLP:AMBER", "TLP:RED"])
axes[1].bar(tl.index, tl.values, color=BLUE)
axes[1].set_title("F9b. tlp_label (rule based)")
for i, v in enumerate(tl.values):
    axes[1].text(i, v, f"{v/len(clean)*100:.1f}%", ha="center", va="bottom", fontsize=8, color=GRAY)
axes[1].tick_params(axis="x", labelsize=8)
finish(fig, "F9_engineered_labels")

# F10 rating distribution split by side_effect_flag (normalized)
fig, ax = plt.subplots(figsize=(6, 3.4))
for i, (flag, lab) in enumerate([(0, "no side-effect keyword"), (1, "side-effect keyword")]):
    sub = clean.loc[clean["side_effect_flag"] == flag, "rating"].value_counts(normalize=True).sort_index()
    ax.bar(sub.index + (i - 0.5) * 0.38, sub.values * 100, width=0.38, color=CAT[i], label=lab)
ax.set_xticks(range(1, 11))
ax.set_xlabel("rating")
ax.set_ylabel("% of reviews in group")
ax.set_title("F10. Rating distribution by side_effect_flag")
ax.legend(frameon=False)
finish(fig, "F10_rating_by_sideeffect_flag")

# F11 chronic vs non-chronic rating distribution
fig, ax = plt.subplots(figsize=(6, 3.4))
for i, (flag, lab) in enumerate([(0, "other conditions"), (1, "chronic conditions")]):
    sub = clean.loc[clean["is_chronic_condition"] == flag, "rating"].value_counts(normalize=True).sort_index()
    ax.bar(sub.index + (i - 0.5) * 0.38, sub.values * 100, width=0.38, color=CAT[i], label=lab)
ax.set_xticks(range(1, 11))
ax.set_xlabel("rating")
ax.set_ylabel("% of reviews in group")
ax.set_title("F11. Rating distribution: chronic vs other conditions")
ax.legend(frameon=False)
finish(fig, "F11_rating_chronic_vs_other")

# F12 QQ plots for normality evidence
fig, axes = plt.subplots(1, 3, figsize=(9, 3.1))
for ax, c in zip(axes, ["rating", "usefulCount", "review_word_len"]):
    s = clean[c].dropna().sample(5000, random_state=SEED)
    (osm, osr), (slope, intercept, r) = stats.probplot(s, dist="norm")
    ax.plot(osm, osr, ".", markersize=2, color=BLUE)
    ax.plot(osm, slope * np.array(osm) + intercept, color=ORANGE, linewidth=1.2)
    ax.set_title(f"F12. Q-Q: {c}")
    ax.set_xlabel("theoretical quantiles")
    ax.set_ylabel("sample quantiles")
finish(fig, "F12_qq_plots")

# F13 rating vs usefulCount (median usefulCount per rating)
fig, ax = plt.subplots(figsize=(6, 3.2))
g = clean.groupby("rating")["usefulCount"].median()
ax.bar(g.index, g.values, color=BLUE)
ax.set_xticks(range(1, 11))
ax.set_xlabel("rating")
ax.set_ylabel("median usefulCount")
ax.set_title("F13. Median usefulCount by rating")
finish(fig, "F13_usefulcount_by_rating")

with open(OUT / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("\nAll Part A outputs written to", OUT)
