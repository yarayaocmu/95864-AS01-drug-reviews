#!/usr/bin/env python3
"""
Assemble every Part A / B / C result table and figure into one Markdown file that can be
pasted into the Google Doc memo appendix. The written-analysis sections are left as
headed placeholders with EVIDENCE NOTES (facts pulled from the results) so that the
narrative is written by the team in its own words, as the assignment requires.

Run after part_a_profile.py, part_b_experiments.py, prepare_sft_data.py:
    python build_memo_appendix.py
Output:
    outputs/AS01_memo_appendix.md
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
A = ROOT / "outputs" / "part_a"
B = ROOT / "outputs" / "part_b"
C = ROOT / "outputs" / "part_c"
OUT = ROOT / "outputs" / "AS01_memo_appendix.md"

sa = json.loads((A / "summary.json").read_text())
sb = json.loads((B / "results.json").read_text())
ev = pd.read_csv(B / "tables" / "05_similarity_evaluation.csv", index_col=0)
norm = pd.read_csv(A / "tables" / "17_normality_tests.csv")
dq = pd.read_csv(A / "tables" / "02_data_quality_checks.csv")
tlp = pd.read_csv(A / "tables" / "08_tlp_labels.csv")
bal = pd.read_csv(C / "sft_split_balance.csv")


def md(path: Path) -> str:
    return path.read_text().strip()


def fig(name: str, part: Path, caption: str) -> str:
    rel = (part / "figures" / f"{name}.png").relative_to(ROOT)
    return f"![{caption}]({rel})\n\n*{caption}*"


def pct(x):
    return f"{x:.1f}%"


L: list[str] = []
add = L.append

add("# AS01 Appendix: Datasheet, Experiments, and Research-Plan Evidence")
add("Dataset: UCI Drug Review Dataset (Drugs.com), Hugging Face mirror `lewtun/drug-reviews`. "
    "All numbers below are produced by the scripts in this folder (see README, Reproducibility).")
add("")
add("> Sections marked **[WRITE IN YOUR OWN WORDS]** must be written by the team. The bullet "
    "points under *Evidence notes* are facts from the results to build the narrative on.")

# --------------------------------------------------------------------------- Part A
add("\n---\n## A. Dataset Datasheet\n")
add("### A.1 Dataset identification\n")
add("| Item | Value |\n|---|---|")
add("| Name | Drug Review Dataset (Drugs.com) |")
add("| Original source | UCI Machine Learning Repository, donated 2018 (Gräßer et al.) |")
add("| Copy used | Hugging Face `lewtun/drug-reviews` (train.jsonl, test.jsonl), downloaded 2026-09-11 |")
add(f"| Observations | {sa['n_observations']:,} ({sa['n_train']:,} train / {sa['n_test']:,} test, original split) |")
add(f"| Original features | {sa['n_features_original']} (uniqueID, drugName, condition, review, rating, date, usefulCount) |")
add(f"| Features after engineering | {sa['n_features_after_engineering']} |")
add("| Access restriction | Public web-scraped reviews; per course policy an HF/UCI dataset may not be used on CMU computing resources without approval, so all processing ran on a personal laptop |")
add("| Language | English |")
add("")
add("### A.2 Feature codebook **[WRITE IN YOUR OWN WORDS: descriptions + utility column]**\n")
add("| Feature | Data type | Description (your words) | Utility for Scenario 01 (your words) |\n|---|---|---|---|")
for f_, t in [("uniqueID", "integer"), ("drugName", "string / categorical"), ("condition", "string / categorical"),
              ("review", "string / free text"), ("rating", "integer 1-10 (ordinal)"), ("date", "date"),
              ("usefulCount", "integer count")]:
    add(f"| {f_} | {t} |  |  |")
add("")
add("### A.3 Feature profile (types, missing values, unique values)\n")
add(md(A / "tables" / "01_feature_profile.md"))
add("\n### A.4 Data quality checks\n")
add(md(A / "tables" / "02_data_quality_checks.md"))
add("\n*Evidence notes:*")
for _, r in dq.iterrows():
    add(f"- {r['check']}: {int(r['count']):,} rows ({r['pct']}%)")
add(f"- Cleaning kept {sa['cleaning']['rows_after_drop_reviews_under_3_words']:,} of "
    f"{sa['cleaning']['rows_raw']:,} rows (exact duplicates and reviews under 3 words removed; "
    f"HTML entities unescaped; wrapping quotes stripped; scraped `</span>` conditions set to missing).")
add("- The 40% 'duplicate review text' is almost entirely the SAME review listed under a brand name and a generic name. "
    "This must be de-duplicated before any train/test split, otherwise the test set leaks training text.")

add("\n### A.5 Traffic Light Protocol labels (engineered feature `tlp_label`)\n")
add("| TLP label | Rule used (this dataset) | Meaning for sharing | Count | Example from dataset |\n|---|---|---|---|---|")
rules = {
    "TLP:CLEAR": "<= 8 words and no first-person pronoun (generic product remark)",
    "TLP:GREEN": "default: a patient's own experience, no extra identifiers",
    "TLP:AMBER": "mentions third parties (my son/wife...), explicit age, pregnancy, or a named doctor",
    "TLP:RED": "crisis or highly sensitive content (suicidal ideation, overdose, HIV, abuse, substance dependence) or a sensitive condition",
}
meaning = {"TLP:CLEAR": "may be shared publicly", "TLP:GREEN": "share within the organisation / partner clinics",
           "TLP:AMBER": "restricted; de-identify before use", "TLP:RED": "do not use for training without review; exclude by default"}
for _, r in tlp.iterrows():
    ex_ = str(r["example_review"]).replace("\n", " ").replace("|", "/")[:220]
    add(f"| {r['tlp_label']} | {rules[r['tlp_label']]} | {meaning[r['tlp_label']]} | {int(r['count']):,} ({r['pct']}%) | "
        f"*{r['example_drug']} / {r['example_condition']}*: \"{ex_}\" |")
add("")
add(fig("F9_engineered_labels", A, "Figure A-1. Engineered label distributions: sentiment_label and tlp_label."))

add("\n### A.6 Descriptive statistics\n")
add("**Numeric features (raw)**\n")
add(md(A / "tables" / "03_numeric_stats_raw.md"))
add("\n**Numeric features after cleaning, including engineered features**\n")
add(md(A / "tables" / "09_numeric_stats_clean_and_engineered.md"))
add("\n**Rating distribution**\n")
add(md(A / "tables" / "06_rating_distribution.md"))
add("\n**Categorical features: long-tail summary**\n")
add(md(A / "tables" / "07_categorical_longtail.md"))
add("\n**Top-20 drugs**\n")
add(md(A / "tables" / "04_top20_drugs.md"))
add("\n**Top-20 conditions**\n")
add(md(A / "tables" / "05_top20_conditions.md"))
add("\n**Engineered categorical features**\n")
add(md(A / "tables" / "10_categorical_freq_engineered.md"))
add("")
for name, cap in [("F1_rating_hist", "Figure A-2. Rating histogram: bimodal, mass at 10 and 1."),
                  ("F2_usefulcount_hist", "Figure A-3. usefulCount, linear and log scale."),
                  ("F3_boxplots", "Figure A-4. Boxplots of rating, usefulCount, review length."),
                  ("F4_top_conditions", "Figure A-5. Top-20 conditions."),
                  ("F5_top_drugs", "Figure A-6. Top-20 drugs."),
                  ("F6_review_length_density", "Figure A-7. Review length density."),
                  ("F7_reviews_per_year", "Figure A-8. Reviews per year."),
                  ("F8_rating_by_condition", "Figure A-9. Rating by top-8 conditions."),
                  ("F13_usefulcount_by_rating", "Figure A-10. Median usefulCount by rating.")]:
    add(fig(name, A, cap))
    add("")

add("\n### A.7 Normal vs non-normal distributions **[WRITE IN YOUR OWN WORDS]**\n")
add(md(A / "tables" / "17_normality_tests.md"))
add("")
add(fig("F12_qq_plots", A, "Figure A-11. Q-Q plots against a normal distribution."))
add("\n*Evidence notes:*")
for _, r in norm.iterrows():
    add(f"- `{r['feature']}`: skew {r['skewness']}, excess kurtosis {r['kurtosis_excess']}, Shapiro W={r['shapiro_W_n5000']} "
        f"(p={r['shapiro_p']:.1e}) -> {r['verdict']}")
add("- rating is bimodal (32% at 10, 13% at 1), so it is not even unimodal; mean 6.98 vs median 8.")
add("- usefulCount is right-skewed with a very long tail (kurtosis > 50): median 16, max 1,291.")
add("- usefulCount becomes close to normal after a log1p transform (Shapiro W 0.64 -> 0.98), i.e. it is approximately log-normal. "
    "review_word_len does NOT improve under the log transform (W 0.95 -> 0.88): it is right-skewed with heavy tails but not log-normal.")
add("- drugName and condition are long-tailed categoricals: top-10 conditions cover 46% of rows; "
    f"{int(sa['categorical_longtail']['drugName']['values_with_<=5_rows'])} of 3,671 drugs have 5 or fewer reviews.")

add("\n### A.8 Impact of these distributions on model development **[WRITE IN YOUR OWN WORDS]**\n")
add("*Evidence notes to build on:* class imbalance (66% POSITIVE / 25% NEGATIVE / 9% NEUTRAL after mapping); "
    "long-tail drugs and conditions (chronic conditions are 22.7% of rows); usefulCount outliers; "
    "brand/generic duplication; Birth Control alone is 18% of rows (the target population of Scenario 01 is chronic disease).")

add("\n### A.9 Feature engineering\n")
add("| New feature | Source feature(s) | How created | Data type | Why useful for Scenario 01 |\n|---|---|---|---|---|")
add("| condition_clean | condition | scraped `</span>` artifacts set to missing | string | removes 1,171 fake condition values that would become fake classes |")
add("| review_char_len / review_word_len / review_sentence_count | review | character, whitespace-token, sentence-boundary counts after HTML unescape | integer | length is a proxy for information content; needed for chunking and sequence-length decisions |")
add("| sentiment_label | rating | 1-4 NEGATIVE, 5-6 NEUTRAL, 7-10 POSITIVE | categorical (3) | the SFT target: patient-reported experience class |")
add("| review_year | date | year component | integer | drift check: drugs and language change over 2008-2017 |")
add("| side_effect_flag | review | regex over 20 common side-effect terms | binary | flags reviews that carry the safety information the tool must surface |")
add("| drug_review_count / condition_review_count | drugName / condition_clean | group size | integer | exposure of each drug/condition in training data (long-tail risk) |")
add("| is_chronic_condition | condition_clean | membership in a curated chronic-condition list | binary | identifies Scenario 01's target population for sub-group evaluation |")
add("| tlp_label | review, condition_clean | rule-based Traffic Light Protocol classifier | categorical (4) | data-protection filter before training; governance evidence |")
add("\n**Evaluation of the new features**\n")
add("Side-effect flag vs rating:\n")
add(md(A / "tables" / "12_cmp_sideeffect_vs_rating.md"))
add("\nSide-effect flag vs sentiment label (% within flag group):\n")
add(md(A / "tables" / "15_cmp_sideeffect_vs_sentiment_pct.md"))
add("\nChronic-condition flag vs rating:\n")
add(md(A / "tables" / "13_cmp_chronic_vs_rating.md"))
add("\nTLP label vs rating:\n")
add(md(A / "tables" / "14_cmp_tlp_vs_rating.md"))
add("\nSpearman correlations:\n")
add(md(A / "tables" / "16_spearman_correlations.md"))
add("")
add(fig("F10_rating_by_sideeffect_flag", A, "Figure A-12. Rating distribution by side_effect_flag."))
add(fig("F11_rating_chronic_vs_other", A, "Figure A-13. Rating distribution, chronic vs other conditions."))
add("\n*Evidence notes:*")
se = sa["comparisons"]["sideeffect_vs_rating"]
add(f"- Reviews with a side-effect keyword rate the drug lower on average ({se['1']['mean']:.2f} vs {se['0']['mean']:.2f}) "
    "but the flag alone barely separates sentiment (65% vs 68% POSITIVE): patients report side effects even when satisfied, "
    "so the keyword flag is a weak label and the model must read context.")
ch = sa["comparisons"]["chronic_vs_rating"]
add(f"- Chronic-condition reviews are slightly MORE positive ({ch['1']['mean']:.2f} vs {ch['0']['mean']:.2f}) and are 22.7% of rows.")
add("- usefulCount correlates with rating (Spearman 0.28): positive reviews get more 'useful' votes, so usefulCount is not a neutral quality signal.")
add("- review length is uncorrelated with rating (0.01) and correlated with side_effect_flag (0.26): longer reviews carry more safety content.")

# --------------------------------------------------------------------------- Part B
add("\n---\n## B. Tokenization and Embedding Experiments\n")
add(f"Sample: {sb['n_sample']} de-duplicated reviews, {sb['config']['per_condition']} per condition from "
    f"{len(sb['conditions'])} conditions (4 high-frequency: Birth Control, Depression, Pain, Acne; "
    "4 chronic: Diabetes Type 2, High Blood Pressure, High Cholesterol, Insomnia), 20-220 words, seed 42. "
    f"Sentiment mix in sample: {sb['sample_sentiment_dist']}.")
add("\n### B.1 Tokenization **[explain your choice in your own words]**\n")
add(md(B / "tables" / "01_tokenization_stats.md"))
add("\nHow the dense model's WordPiece tokenizer splits common chronic-disease drug names:\n")
add(md(B / "tables" / "02_drug_name_subword_splits.md"))
add("")
add(fig("G4_subword_stats", B, "Figure B-1. Subword tokens per word and per review, with the model's 256-token limit."))
add("\n*Evidence notes:*")
add(f"- Stopwords are {sb['stopword_share_mean']*100:.0f}% of word tokens in patient reviews.")
add(f"- WordPiece produces {sb['subword_per_word_ratio_mean']:.2f} subword tokens per word; "
    f"{sb['reviews_truncated_by_dense_model']} of {sb['n_sample']} sampled reviews exceed the 256-token limit "
    "(sample was capped at 220 words; in the full data 1.6% of reviews are longer than 220 words).")
add("- Drug names are almost never single tokens (metformin -> met ##form ##in; hydrochlorothiazide -> 7 pieces); "
    "only 'insulin' is in the vocabulary. The general-domain tokenizer has no lexical unit for the entities the scenario cares about.")
add("\n### B.2 Embedding models **[explain your choice in your own words]**\n")
add(md(B / "tables" / "04_embedding_shapes.md"))
add(f"\nDense model used: `{sb['config']['dense_model']}` (EmbeddingGemma is gated on Hugging Face and needs "
    "licence approval; the script accepts `--dense-model models/embedders/google/embeddinggemma-300m` once downloaded).")
add("\nTop TF-IDF terms per condition (what the sparse representation 'sees'):\n")
add(md(B / "tables" / "03_top_tfidf_terms_per_condition.md"))
add("\n### B.3 Similarity results (cosine)\n")
add(md(B / "tables" / "05_similarity_evaluation.md"))
add("\nPer-condition 1-NN accuracy:\n")
add(md(B / "tables" / "07_per_condition_nn1.md"))
add("")
add(fig("G1_nn_match_rates", B, "Figure B-2. Leave-one-out 1-NN match rates by method."))
add(fig("G2_similarity_distributions", B, "Figure B-3. Pairwise cosine, same vs different condition."))
add(fig("G3_condition_heatmaps", B, "Figure B-4. Mean cosine between conditions, TF-IDF vs dense."))
add(fig("G5_per_condition_nn", B, "Figure B-5. Per-condition 1-NN accuracy."))
add("\nMost / least similar pairs per method (truncated):\n")
pairs = pd.read_csv(B / "tables" / "06_most_least_similar_pairs.csv")
pairs = pairs[pairs["method"].isin(["bow", "tfidf", "dense"]) & (pairs["rank"] == 1)]
add("| method | kind | cosine | cond_i | cond_j | review_i | review_j |\n|---|---|---|---|---|---|---|")
for _, r in pairs.iterrows():
    add(f"| {r['method']} | {r['kind']} | {r['cosine']} | {r['cond_i']} | {r['cond_j']} | "
        f"{str(r['review_i'])[:110].replace('|','/')}... | {str(r['review_j'])[:110].replace('|','/')}... |")
add("\nClinical query retrieval (top-3, TF-IDF vs dense): see `outputs/part_b/tables/09_clinical_query_retrieval.csv`.\n")
q = pd.read_csv(B / "tables" / "09_clinical_query_retrieval.csv")
add("| query | method | rank | cosine | drug | condition | rating | review |\n|---|---|---|---|---|---|---|---|")
for _, r in q.iterrows():
    add(f"| {r['query'][:60]} | {r['method']} | {r['rank']} | {r['cosine']} | {r['drug']} | {r['condition']} | {r['rating']} | "
        f"{str(r['review'])[:120].replace('|','/')}... |")
add("\n*Evidence notes:*")
for m in ["bow", "tfidf", "tfidf_nostop", "dense", "dense_chunked"]:
    add(f"- `{m}`: 1-NN same-condition {ev.loc[m,'cond_nn1_match']:.3f} (chance 0.125), "
        f"1-NN same-sentiment {ev.loc[m,'sent_nn1_match']:.3f} (majority-class baseline {ev.loc['chance_level','sent_nn1_match']:.3f}), "
        f"within-minus-between condition gap {ev.loc[m,'cond_gap']:.3f} (Cohen d {ev.loc[m,'cond_effect_d']:.2f}), "
        f"sentiment d {ev.loc[m,'sent_effect_d']:.3f}")
add("- Every representation groups reviews by CONDITION far above chance, and NONE groups them by SENTIMENT "
    "(all sentiment 1-NN rates are at or below the majority-class baseline). Generic embeddings encode topic, not patient experience. "
    "This is the direct motivation for supervised fine-tuning on the rating-derived label.")
add("- Removing stopwords raises BoW from 0.39 to 0.63 and TF-IDF from 0.65 to 0.72: half of the tokens are noise for the sparse methods.")
add("- The dense model is best overall (0.86) and, importantly, closes the gap on chronic conditions where TF-IDF is weakest "
    "(High Blood Pressure 0.43 -> 0.83, High Cholesterol 0.54 -> 0.87).")
add("- Chunking (40-word chunks, mean-pooled) LOWERS dense accuracy (0.86 -> 0.80): patient reviews are short enough to embed whole; chunking splits context.")
add("- BoW's most-similar pair joins a Pain review and a High Cholesterol review because both are long and share function words; "
    "dense's most-similar pair is two statin users describing muscle pain, i.e. a genuinely clinically related pair.")
add("- In the dense heatmap Depression and Insomnia are the closest cross-condition pair (0.42), and High Blood Pressure sits close to "
    "Depression (0.38): the representation reflects symptom overlap (sleep, fatigue), which is useful for the tool but also a source of confusion.")
add("- Query retrieval: for 'dizziness and fatigue after starting blood pressure medication' TF-IDF returns reviews that merely repeat "
    "'blood pressure'; dense returns a clonidine review that literally describes dizzy spells and fatigue.")

add("\n### B.4 Interpretation **[WRITE IN YOUR OWN WORDS: performance, generalisation, bias & representation, data quality, future decisions]**\n")

# --------------------------------------------------------------------------- Part C
add("\n---\n## C. Research Plan Evidence: SFT data preparation (no training yet)\n")
add(md(C / "sft_split_balance.md"))
add("\nExample SFT record:\n")
add("```json\n" + (C / "sft_example_record.json").read_text() + "\n```")
add("\nPreparation log:\n")
add("```json\n" + (C / "sft_prep_log.json").read_text() + "\n```")
add("\n*Evidence notes:*")
add("- TLP:RED rows are excluded before any training (patient-data protection plan, Scenario 01 optional requirement).")
add("- Brand/generic duplicates are collapsed so the unseen-drug test cannot leak text from training.")
add("- Evaluation splits keep the natural 66/25/9 imbalance; only the training set is capped per class.")

add("\n---\n## Reproducibility\n")
add("```bash\ncd as01_v02\npython3 -m venv .venv && source .venv/bin/activate\npip install -r requirements.txt\n"
    "# data: data/raw/train.jsonl, data/raw/test.jsonl from https://huggingface.co/datasets/lewtun/drug-reviews\n"
    "python part_a_profile.py        # outputs/part_a\npython part_b_experiments.py    # outputs/part_b\n"
    "python prepare_sft_data.py      # outputs/part_c, data/sft\npython build_memo_appendix.py   # this file\n"
    "python main_test.py             # starter-kit self-test (17 pass, 1 skip)\n```")
add("\nSeeds: 42 everywhere. Hardware used: Apple M4 Pro, 48 GB, CPU/MPS. Python 3.13.7; package versions in requirements.txt.")

OUT.write_text("\n".join(L))
print("Wrote", OUT, f"({OUT.stat().st_size/1024:.0f} KB)")
