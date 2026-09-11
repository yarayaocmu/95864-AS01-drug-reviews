#!/usr/bin/env python3
"""
AS01 Part B: Tokenization and embedding experiments on the Drugs.com review dataset.

Pipeline:  raw review -> tokenization/chunking -> embedding -> cosine similarity -> evaluation

Tokenization strategies compared
    word            starter word tokenizer (lowercase, strip punctuation)
    word_nostop     word tokenizer + stopword removal
    subword         the dense model's WordPiece tokenizer (reported for token statistics)
    fixed_chunk     40-word chunks, embedded separately and mean-pooled (dense model only)

Embedding strategies compared
    bow             Bag-of-Words term counts (sparse)             [starter]
    tfidf           TF-IDF (sparse)                               [starter]
    dense           sentence-transformer dense embedding          [default all-MiniLM-L6-v2,
                                                                   swap with --dense-model
                                                                   models/embedders/google/embeddinggemma-300m]
    dense_chunked   dense embedding of 40-word chunks, mean-pooled

Similarity metric: cosine similarity (sparse and dense), as in the starter kit.

Evaluation (all leave-one-out, on a stratified sample)
    * 1-NN / 5-NN condition match rate      (does the representation group reviews by condition?)
    * 1-NN sentiment match rate             (does it group reviews by patient experience?)
    * within-vs-between condition similarity separation
    * most / least similar pairs, per method
    * clinical query retrieval (qualitative)

Run:
    python part_b_experiments.py [--per-condition 100] [--dense-model all-MiniLM-L6-v2]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
# NOTE: src/tokenizers.py (starter kit) shadows the Hugging Face `tokenizers` package if
# src/ is put on sys.path, so load our helper module by file path instead.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("kr_pipeline", ROOT / "src" / "kr_pipeline.py")
kr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kr)

OUT = ROOT / "outputs" / "part_b"
TAB, FIG = OUT / "tables", OUT / "figures"
for p in (TAB, FIG):
    p.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e"
CAT = [BLUE, ORANGE, AQUA, YELLOW]
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.edgecolor": "#c3c2b7", "axes.titleweight": "bold",
})

parser = argparse.ArgumentParser()
parser.add_argument("--per-condition", type=int, default=100)
parser.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
parser.add_argument("--chunk-size", type=int, default=40)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
SEED = args.seed

CONDITIONS = [  # 4 high-frequency + 4 chronic (Scenario 01 target population)
    "Birth Control", "Depression", "Pain", "Acne",
    "Diabetes, Type 2", "High Blood Pressure", "High Cholesterol", "Insomnia",
]


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> None:
    df.to_csv(TAB / f"{name}.csv", index=index)
    (TAB / f"{name}.md").write_text(df.to_markdown(index=index))


results: dict = {"config": vars(args), "conditions": CONDITIONS}

# =========================================================================
# 1. SAMPLE
# =========================================================================
df = pd.read_parquet(ROOT / "data" / "clean" / "reviews_clean.parquet")
# the same review is listed under brand and generic names: keep one copy so that
# nearest-neighbour evaluation is not inflated by identical texts
df = df.drop_duplicates(subset=["review"], keep="first")
df = df[df["condition_clean"].isin(CONDITIONS)]
df = df[(df["review_word_len"] >= 20) & (df["review_word_len"] <= 220)]
parts = []
for c in CONDITIONS:
    g = df[df["condition_clean"] == c]
    parts.append(g.sample(min(len(g), args.per_condition), random_state=SEED))
sample = pd.concat(parts).reset_index(drop=True)
texts = sample["review"].tolist()
cond = sample["condition_clean"].astype(str).to_numpy()
sent = sample["sentiment_label"].to_numpy()
n = len(sample)
print(f"Sample: {n} reviews across {len(CONDITIONS)} conditions")
results["n_sample"] = int(n)
results["sample_sentiment_dist"] = {k: int(v) for k, v in Counter(sent).items()}
sample[["uniqueID", "drugName", "condition_clean", "rating", "sentiment_label", "review_word_len"]] \
    .to_csv(TAB / "00_sample_manifest.csv", index=False)

# =========================================================================
# 2. TOKENIZATION
# =========================================================================
tok_word = [kr.tokenize_words(t) for t in texts]
tok_nostop = [kr.tokenize_words_nostop(t) for t in texts]
chunks = [kr.chunk_fixed_size(t, args.chunk_size) for t in texts]

from sentence_transformers import SentenceTransformer  # noqa: E402

model = SentenceTransformer(args.dense_model)
hf_tok = model.tokenizer
max_seq = model.max_seq_length
tok_sub = [hf_tok.tokenize(t) for t in texts]

tok_stats = pd.DataFrame({
    "strategy": ["word", "word_nostop", f"subword ({Path(args.dense_model).name})", f"fixed_chunk({args.chunk_size})"],
    "units_per_review_mean": [np.mean([len(t) for t in tok_word]), np.mean([len(t) for t in tok_nostop]),
                              np.mean([len(t) for t in tok_sub]), np.mean([len(c) for c in chunks])],
    "units_per_review_median": [np.median([len(t) for t in tok_word]), np.median([len(t) for t in tok_nostop]),
                                np.median([len(t) for t in tok_sub]), np.median([len(c) for c in chunks])],
    "vocab_size": [len(kr.build_vocabulary(tok_word)), len(kr.build_vocabulary(tok_nostop)),
                   len(set(x for t in tok_sub for x in t)), "-"],
    "note": ["starter tokenizer", "stopwords removed", f"model max_seq_length={max_seq}",
             "chunks of words, mean-pooled"],
}).round(2)
save_table(tok_stats, "01_tokenization_stats", index=False)
print(tok_stats)

# subword vs word ratio and truncation
ratio = np.array([len(s) / max(1, len(w)) for s, w in zip(tok_sub, tok_word)])
truncated = int(np.sum(np.array([len(s) for s in tok_sub]) + 2 > max_seq))
results["subword_per_word_ratio_mean"] = float(ratio.mean())
results["reviews_truncated_by_dense_model"] = truncated
results["reviews_truncated_pct"] = round(100 * truncated / n, 2)

# how drug names are split into subwords (domain-vocabulary evidence)
drug_examples = ["metformin", "lisinopril", "atorvastatin", "amlodipine", "sertraline", "gabapentin",
                 "levonorgestrel", "hydrochlorothiazide", "ibuprofen", "insulin", "aspirin", "warfarin"]
split_tab = pd.DataFrame({
    "drug": drug_examples,
    "wordpiece_tokens": [" ".join(hf_tok.tokenize(d)) for d in drug_examples],
    "n_pieces": [len(hf_tok.tokenize(d)) for d in drug_examples],
})
save_table(split_tab, "02_drug_name_subword_splits", index=False)
print(split_tab)

# stopword share in reviews
stop_share = 1 - np.array([len(b) for b in tok_nostop]) / np.array([len(a) for a in tok_word])
results["stopword_share_mean"] = float(stop_share.mean())

# =========================================================================
# 3. EMBEDDINGS
# =========================================================================
methods: dict[str, np.ndarray] = {}

for name, toks in [("bow", tok_word), ("tfidf", tok_word), ("bow_nostop", tok_nostop), ("tfidf_nostop", tok_nostop)]:
    strat = "bow" if name.startswith("bow") else "tfidf"
    emb, vocab, extra = kr.embed_corpus(toks, strat)
    methods[name] = kr.sparse_to_dense_matrix(emb, len(vocab))
    if name == "tfidf":
        idf = extra["idf"]
        inv = {i: t for t, i in vocab.items()}
        # top TF-IDF terms per condition (what the representation "thinks" each condition is about)
        rows = []
        for c in CONDITIONS:
            idx = np.where(cond == c)[0]
            mean_vec = methods[name][idx].mean(axis=0)
            top = np.argsort(-mean_vec)[:12]
            rows.append({"condition": c, "top_tfidf_terms": ", ".join(inv[i] for i in top)})
        save_table(pd.DataFrame(rows), "03_top_tfidf_terms_per_condition", index=False)

dense = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
methods["dense"] = np.asarray(dense, dtype=np.float32)

# chunked dense: embed each chunk, mean-pool per review
flat_chunks, owner = [], []
for i, cs in enumerate(chunks):
    for c in cs:
        flat_chunks.append(" ".join(c))
        owner.append(i)
chunk_emb = model.encode(flat_chunks, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
pooled = np.zeros((n, chunk_emb.shape[1]), dtype=np.float32)
for e, o in zip(chunk_emb, owner):
    pooled[o] += e
pooled /= np.bincount(owner, minlength=n)[:, None]
methods["dense_chunked"] = pooled

sparsity = {k: float((v == 0).mean()) for k, v in methods.items()}
dims = {k: int(v.shape[1]) for k, v in methods.items()}
emb_tab = pd.DataFrame({"dimension": dims, "fraction_zero_entries": sparsity}).round(4)
save_table(emb_tab, "04_embedding_shapes")
print(emb_tab)

# =========================================================================
# 4. SIMILARITY + EVALUATION
# =========================================================================
def evaluate(sim: np.ndarray, labels: np.ndarray, k: int = 5) -> dict:
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)
    nn1 = np.argmax(s, axis=1)
    top_k = np.argsort(-s, axis=1)[:, :k]
    acc1 = float(np.mean(labels[nn1] == labels))
    # k-NN majority vote
    maj = []
    for i in range(len(labels)):
        votes = Counter(labels[top_k[i]])
        maj.append(votes.most_common(1)[0][0])
    acck = float(np.mean(np.array(maj) == labels))
    return {"nn1_match": acc1, f"nn{k}_majority_match": acck}


def separation(sim: np.ndarray, labels: np.ndarray) -> dict:
    iu = np.triu_indices_from(sim, k=1)
    same = labels[iu[0]] == labels[iu[1]]
    vals = sim[iu]
    within, between = vals[same], vals[~same]
    return {
        "within_mean": float(within.mean()), "between_mean": float(between.mean()),
        "gap": float(within.mean() - between.mean()),
        "effect_size_d": float((within.mean() - between.mean()) / np.sqrt((within.var() + between.var()) / 2)),
        "all_pairs_mean": float(vals.mean()), "all_pairs_median": float(np.median(vals)),
        "all_pairs_min": float(vals.min()), "all_pairs_max": float(vals.max()),
    }


sims: dict[str, np.ndarray] = {}
eval_rows, pair_rows = [], []
chance_cond = float(np.mean([(cond == c).mean() ** 2 for c in CONDITIONS]) * 0 + 1 / len(CONDITIONS))
chance_sent = float(max(Counter(sent).values()) / n)
for name, m in methods.items():
    sim = kr.cosine_matrix(m)
    sims[name] = sim
    ev_c = evaluate(sim, cond)
    ev_s = evaluate(sim, sent)
    sep_c = separation(sim, cond)
    sep_s = separation(sim, sent)
    eval_rows.append({
        "method": name,
        "cond_nn1_match": ev_c["nn1_match"], "cond_nn5_majority": ev_c["nn5_majority_match"],
        "sent_nn1_match": ev_s["nn1_match"], "sent_nn5_majority": ev_s["nn5_majority_match"],
        "cond_within_mean": sep_c["within_mean"], "cond_between_mean": sep_c["between_mean"],
        "cond_gap": sep_c["gap"], "cond_effect_d": sep_c["effect_size_d"],
        "sent_gap": sep_s["gap"], "sent_effect_d": sep_s["effect_size_d"],
        "pairs_mean": sep_c["all_pairs_mean"], "pairs_median": sep_c["all_pairs_median"],
        "pairs_min": sep_c["all_pairs_min"], "pairs_max": sep_c["all_pairs_max"],
    })
    # most / least similar pairs
    s = sim.copy()
    np.fill_diagonal(s, np.nan)
    iu = np.triu_indices(n, k=1)
    order = np.argsort(s[iu])
    for rank, idx in enumerate(order[-3:][::-1]):
        i, j = iu[0][idx], iu[1][idx]
        pair_rows.append({"method": name, "kind": "most_similar", "rank": rank + 1, "cosine": round(float(s[i, j]), 4),
                          "cond_i": cond[i], "cond_j": cond[j], "sent_i": sent[i], "sent_j": sent[j],
                          "review_i": texts[i][:160], "review_j": texts[j][:160]})
    for rank, idx in enumerate(order[:3]):
        i, j = iu[0][idx], iu[1][idx]
        pair_rows.append({"method": name, "kind": "least_similar", "rank": rank + 1, "cosine": round(float(s[i, j]), 4),
                          "cond_i": cond[i], "cond_j": cond[j], "sent_i": sent[i], "sent_j": sent[j],
                          "review_i": texts[i][:160], "review_j": texts[j][:160]})

eval_df = pd.DataFrame(eval_rows).set_index("method").round(4)
eval_df.loc["chance_level"] = np.nan
eval_df.loc["chance_level", ["cond_nn1_match", "cond_nn5_majority"]] = round(chance_cond, 4)
eval_df.loc["chance_level", ["sent_nn1_match", "sent_nn5_majority"]] = round(chance_sent, 4)
save_table(eval_df, "05_similarity_evaluation")
save_table(pd.DataFrame(pair_rows), "06_most_least_similar_pairs", index=False)
print(eval_df[["cond_nn1_match", "cond_nn5_majority", "sent_nn1_match", "cond_gap", "cond_effect_d", "sent_effect_d"]])
results["evaluation"] = eval_df.reset_index().to_dict(orient="records")

# per-condition 1-NN accuracy (chronic vs high-frequency)
rows = []
for name in ["tfidf", "dense"]:
    s = sims[name].copy()
    np.fill_diagonal(s, -np.inf)
    nn1 = np.argmax(s, axis=1)
    for c in CONDITIONS:
        idx = cond == c
        rows.append({"method": name, "condition": c, "n": int(idx.sum()),
                     "nn1_condition_match": float(np.mean(cond[nn1[idx]] == c))})
percond = pd.DataFrame(rows).pivot(index="condition", columns="method", values="nn1_condition_match").round(3)
save_table(percond, "07_per_condition_nn1")
print(percond)

# condition x condition mean similarity (tfidf vs dense)
for name in ["tfidf", "dense"]:
    mat = np.zeros((len(CONDITIONS), len(CONDITIONS)))
    for a, ca in enumerate(CONDITIONS):
        for b, cb in enumerate(CONDITIONS):
            ia, ib = np.where(cond == ca)[0], np.where(cond == cb)[0]
            block = sims[name][np.ix_(ia, ib)]
            if a == b:
                iu = np.triu_indices(len(ia), k=1)
                mat[a, b] = block[iu].mean()
            else:
                mat[a, b] = block.mean()
    save_table(pd.DataFrame(mat, index=CONDITIONS, columns=CONDITIONS).round(3), f"08_condition_similarity_{name}")

# =========================================================================
# 5. CLINICAL QUERY RETRIEVAL (qualitative)
# =========================================================================
QUERIES = [
    "patient reports dizziness and fatigue after starting blood pressure medication",
    "stomach cramps and diarrhea in the first weeks of taking metformin",
    "medication stopped working and symptoms came back",
    "no side effects and cholesterol numbers improved",
]
q_rows = []
# sparse: build tf-idf for queries using the corpus vocab/idf
emb_t, vocab_t, extra_t = kr.embed_corpus(tok_word, "tfidf")
corpus_tfidf = methods["tfidf"]
q_dense = model.encode(QUERIES, normalize_embeddings=True)
for qi, q in enumerate(QUERIES):
    q_vec = kr.embed_tfidf(kr.tokenize_words(q), vocab_t, extra_t["idf"])
    q_arr = kr.sparse_to_dense_matrix([q_vec], len(vocab_t))[0]
    qn = np.linalg.norm(q_arr) or 1.0
    cn = np.linalg.norm(corpus_tfidf, axis=1)
    cn[cn == 0] = 1.0
    s_tfidf = (corpus_tfidf @ q_arr) / (cn * qn)
    s_dense = methods["dense"] @ q_dense[qi]
    for name, s in [("tfidf", s_tfidf), ("dense", s_dense)]:
        for rank, i in enumerate(np.argsort(-s)[:3]):
            q_rows.append({"query": q, "method": name, "rank": rank + 1, "cosine": round(float(s[i]), 4),
                           "drug": sample["drugName"].iloc[i], "condition": cond[i], "rating": int(sample["rating"].iloc[i]),
                           "review": texts[i][:200]})
save_table(pd.DataFrame(q_rows), "09_clinical_query_retrieval", index=False)

# =========================================================================
# 6. FIGURES
# =========================================================================
def finish(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


order = ["bow", "bow_nostop", "tfidf", "tfidf_nostop", "dense", "dense_chunked"]
# G1 NN match rates
fig, ax = plt.subplots(figsize=(7.5, 3.6))
x = np.arange(len(order))
w = 0.38
ax.bar(x - w / 2, [eval_df.loc[m, "cond_nn1_match"] for m in order], w, color=BLUE, label="1-NN same condition")
ax.bar(x + w / 2, [eval_df.loc[m, "sent_nn1_match"] for m in order], w, color=ORANGE, label="1-NN same sentiment")
ax.axhline(chance_cond, color=BLUE, linestyle=":", linewidth=1)
ax.axhline(chance_sent, color=ORANGE, linestyle=":", linewidth=1)
ax.text(len(order) - 0.5, chance_cond + 0.01, "chance (condition)", color=BLUE, fontsize=7, ha="right")
ax.text(len(order) - 0.5, chance_sent + 0.01, "chance (sentiment)", color=ORANGE, fontsize=7, ha="right")
ax.set_xticks(x)
ax.set_xticklabels(order, rotation=15)
ax.set_ylim(0, 1)
ax.set_ylabel("leave-one-out 1-NN match rate")
ax.set_title("G1. Does the nearest neighbour share the condition / sentiment?")
ax.legend(frameon=False, fontsize=8)
finish(fig, "G1_nn_match_rates")

# G2 within vs between similarity distributions
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=False)
for ax, name in zip(axes, ["bow", "tfidf", "dense"]):
    sim = sims[name]
    iu = np.triu_indices_from(sim, k=1)
    same = cond[iu[0]] == cond[iu[1]]
    vals = sim[iu]
    bins = np.linspace(min(vals.min(), 0), 1, 50)
    ax.hist(vals[~same], bins=bins, density=True, color=BLUE, alpha=0.8, label="different condition")
    ax.hist(vals[same], bins=bins, density=True, color=ORANGE, alpha=0.65, label="same condition")
    ax.set_title(f"G2. pairwise cosine: {name}")
    ax.set_xlabel("cosine similarity")
axes[0].set_ylabel("density")
axes[0].legend(frameon=False, fontsize=7)
finish(fig, "G2_similarity_distributions")

# G3 condition x condition heatmaps (single sequential hue)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, name in zip(axes, ["tfidf", "dense"]):
    mat = pd.read_csv(TAB / f"08_condition_similarity_{name}.csv", index_col=0).to_numpy()
    im = ax.imshow(mat, cmap="Blues", vmin=mat.min(), vmax=mat.max())
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_yticks(range(len(CONDITIONS)))
    ax.set_xticklabels([c[:14] for c in CONDITIONS], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([c[:14] for c in CONDITIONS], fontsize=7)
    ax.grid(False)
    for a in range(len(CONDITIONS)):
        for b in range(len(CONDITIONS)):
            ax.text(b, a, f"{mat[a, b]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if mat[a, b] > (mat.min() + mat.max()) / 2 else "#0b0b0b")
    ax.set_title(f"G3. mean cosine between conditions: {name}")
finish(fig, "G3_condition_heatmaps")

# G4 subword/word ratio and truncation
fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
axes[0].hist(ratio, bins=30, color=BLUE)
axes[0].set_title("G4a. subword tokens per word")
axes[0].set_xlabel("WordPiece tokens / words")
lens = np.array([len(s) for s in tok_sub])
axes[1].hist(lens, bins=40, color=BLUE)
axes[1].axvline(max_seq, color=ORANGE, linewidth=1.5)
axes[1].text(max_seq + 3, axes[1].get_ylim()[1] * 0.85, f"model limit = {max_seq}", color=ORANGE, fontsize=8)
axes[1].set_title("G4b. subword tokens per review")
axes[1].set_xlabel("tokens")
finish(fig, "G4_subword_stats")

# G5 per-condition NN accuracy tfidf vs dense
fig, ax = plt.subplots(figsize=(7.5, 3.4))
x = np.arange(len(CONDITIONS))
ax.bar(x - w / 2, percond["tfidf"].reindex(CONDITIONS), w, color=BLUE, label="tfidf")
ax.bar(x + w / 2, percond["dense"].reindex(CONDITIONS), w, color=ORANGE, label="dense")
ax.set_xticks(x)
ax.set_xticklabels([c[:16] for c in CONDITIONS], rotation=20, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("1-NN same-condition rate")
ax.set_title("G5. Per-condition 1-NN accuracy (left 4 = high-frequency, right 4 = chronic)")
ax.legend(frameon=False)
finish(fig, "G5_per_condition_nn")

results["tokenization_stats"] = tok_stats.to_dict(orient="records")
(OUT / "results.json").write_text(json.dumps(results, indent=2, default=str))
print("\nPart B outputs written to", OUT)
