# AS01 — Team project files (Scenario 01, Drugs.com reviews)

Repository: https://github.com/yarayaocmu/95864-AS01-drug-reviews  
Memo: `report/AS01_memo.pdf`

This folder is the course starter kit (`as01_v02`) plus our own scripts. The starter files are untouched
so that `main_test.py` still passes; our code lives in the files listed below.

## Our scripts

| File | What it does | Output |
|---|---|---|
| `part_a_profile.py` | Part A: load raw data, quality checks, cleaning, descriptive statistics, normality tests, feature engineering (incl. TLP labels), 13 figures | `outputs/part_a/`, `data/clean/reviews_clean.parquet` |
| `src/kr_pipeline.py` | Clean importable versions of the starter tokenizer / BoW / TF-IDF / cosine helpers (same algorithms, no `self` bug) | – |
| `part_b_experiments.py` | Part B: stratified sample, 4 tokenization strategies, 6 embedding variants (BoW, TF-IDF, ±stopwords, dense, dense-chunked), cosine similarity, 1-NN / 5-NN evaluation, query retrieval, 5 figures | `outputs/part_b/` |
| `prepare_sft_data.py` | Part C: research-plan data pipeline (TLP:RED filter, brand/generic de-dup, unseen-drug hold-out, balanced train) -> chat JSONL | `outputs/part_c/`, `data/sft/` |
| `build_memo_appendix.py` | Assembles all tables / figures into one Markdown appendix with evidence notes | `outputs/AS01_memo_appendix.md` |

## Run order

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put train.jsonl / test.jsonl from https://huggingface.co/datasets/lewtun/drug-reviews in data/raw/
python part_a_profile.py
python part_b_experiments.py            # add --dense-model models/embedders/google/embeddinggemma-300m if you have Gemma
python prepare_sft_data.py
python build_memo_appendix.py
python main_test.py                     # starter self-test
```

Seed 42 everywhere. Runs on a laptop (no GPU needed; MiniLM embeds 800 reviews in seconds).

## Policy notes

- Dataset comes from Hugging Face / UCI: per the course FAQ it must not be processed on CMU machines without approval.
  Everything here ran on a personal laptop.
- EmbeddingGemma is a gated model: accept the licence on Hugging Face, put `HF_TOKEN=...` in `.env`, run `downloader.py`
  (edit `repo_id` to `google/embeddinggemma-300m`), then pass `--dense-model` as above.
- `data/raw`, `data/clean`, `data/sft`, `models/`, `.venv`, `.env` should stay out of any shared repo.
