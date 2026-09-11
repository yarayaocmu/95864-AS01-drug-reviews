#!/usr/bin/env python3
"""
AS01 Part C support: turn the cleaned Drugs.com reviews into SFT-ready chat JSONL
for the research plan (AS02 / final project). Nothing is trained here; this script
only documents and executes the data plan so it is reproducible.

Steps
    1. drop TLP:RED reviews (patient-data protection plan) and keep AMBER reviews as-is
       (they are public but flagged for future de-identification)
    2. de-duplicate the brand/generic double listing (same review text under 2+ drugNames)
    3. hold out 10% of DRUGS entirely  -> test_unseen_drugs (generalisation test)
    4. remaining rows: stratified 80/10/10 train/valid/test_seen split
    5. cap the majority class in TRAIN only (class balance), keep eval splits natural
    6. write chat-format JSONL: system + user(drug, condition, review) -> assistant(label)

Run:
    python prepare_sft_data.py [--cap-per-class 8000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "part_c"
SFT = ROOT / "data" / "sft"
for p in (OUT, SFT):
    p.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--cap-per-class", type=int, default=8000)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
rng = np.random.default_rng(args.seed)

SYSTEM = ("You are a clinical decision-support assistant for clinicians and pharmacists. "
          "Read a patient's self-reported medication experience and classify it as POSITIVE, "
          "NEUTRAL, or NEGATIVE. Answer with the label only. This is not medical advice and "
          "does not replace a clinician's judgement.")

df = pd.read_parquet(ROOT / "data" / "clean" / "reviews_clean.parquet")
log = {"rows_clean": int(len(df))}

# 1. patient-data protection: remove RED
df = df[df["tlp_label"] != "TLP:RED"]
log["rows_after_drop_TLP_RED"] = int(len(df))

# 2. brand/generic double listing
df = df.drop_duplicates(subset=["review"], keep="first")
log["rows_after_text_dedup"] = int(len(df))

df = df.dropna(subset=["condition_clean"])
log["rows_after_drop_missing_condition"] = int(len(df))

# 3. unseen-drug hold-out
drugs = df["drugName"].unique()
unseen = set(rng.choice(drugs, size=int(0.10 * len(drugs)), replace=False))
test_unseen = df[df["drugName"].isin(unseen)]
rest = df[~df["drugName"].isin(unseen)]
log["n_drugs_total"] = int(len(drugs))
log["n_drugs_held_out"] = int(len(unseen))

# 4. stratified split of the rest by sentiment label
rest = rest.sample(frac=1.0, random_state=args.seed)
parts = {"train": [], "valid": [], "test_seen": []}
for lab, g in rest.groupby("sentiment_label"):
    n = len(g)
    a, b = int(0.8 * n), int(0.9 * n)
    parts["train"].append(g.iloc[:a])
    parts["valid"].append(g.iloc[a:b])
    parts["test_seen"].append(g.iloc[b:])
splits = {k: pd.concat(v) for k, v in parts.items()}
splits["test_unseen"] = test_unseen

# 5. cap majority class in train only
train_bal = pd.concat([g.sample(min(len(g), args.cap_per_class), random_state=args.seed)
                       for _, g in splits["train"].groupby("sentiment_label")])
splits["train_balanced"] = train_bal.sample(frac=1.0, random_state=args.seed)

# class balance table
rows = []
for name, part in splits.items():
    vc = part["sentiment_label"].value_counts()
    rows.append({"split": name, "rows": int(len(part)),
                 "NEGATIVE": int(vc.get("NEGATIVE", 0)), "NEUTRAL": int(vc.get("NEUTRAL", 0)),
                 "POSITIVE": int(vc.get("POSITIVE", 0)),
                 "chronic_rows": int(part["is_chronic_condition"].sum()),
                 "n_drugs": int(part["drugName"].nunique())})
bal = pd.DataFrame(rows)
bal["POSITIVE_pct"] = (bal["POSITIVE"] / bal["rows"] * 100).round(1)
bal.to_csv(OUT / "sft_split_balance.csv", index=False)
(OUT / "sft_split_balance.md").write_text(bal.to_markdown(index=False))
print(bal)


# 6. write JSONL
def to_record(row) -> dict:
    user = (f"Drug: {row.drugName}\nCondition: {row.condition_clean}\n"
            f"Patient report: \"{row.review}\"")
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user},
                         {"role": "assistant", "content": row.sentiment_label}],
            "meta": {"uniqueID": int(row.uniqueID), "rating": int(row.rating),
                     "condition": row.condition_clean, "is_chronic": int(row.is_chronic_condition),
                     "tlp": row.tlp_label}}


for name, part in splits.items():
    with open(SFT / f"sft_{name}.jsonl", "w") as f:
        for row in part.itertuples():
            f.write(json.dumps(to_record(row)) + "\n")
    log[f"written_{name}"] = int(len(part))

# example record for the memo
ex = to_record(next(splits["train_balanced"].itertuples()))
(OUT / "sft_example_record.json").write_text(json.dumps(ex, indent=2))
(OUT / "sft_prep_log.json").write_text(json.dumps(log, indent=2))
print(json.dumps(log, indent=2))
print("SFT files written to", SFT)
