# AS01 Appendix: Datasheet, Experiments, and Research-Plan Evidence
Dataset: UCI Drug Review Dataset (Drugs.com), Hugging Face mirror `lewtun/drug-reviews`. All numbers below are produced by the scripts in this folder (see README, Reproducibility).

> Sections marked **[WRITE IN YOUR OWN WORDS]** must be written by the team. The bullet points under *Evidence notes* are facts from the results to build the narrative on.

---
## A. Dataset Datasheet

### A.1 Dataset identification

| Item | Value |
|---|---|
| Name | Drug Review Dataset (Drugs.com) |
| Original source | UCI Machine Learning Repository, donated 2018 (Gräßer et al.) |
| Copy used | Hugging Face `lewtun/drug-reviews` (train.jsonl, test.jsonl), downloaded 2026-09-11 |
| Observations | 215,063 (161,297 train / 53,766 test, original split) |
| Original features | 7 (uniqueID, drugName, condition, review, rating, date, usefulCount) |
| Features after engineering | 19 |
| Access restriction | Public web-scraped reviews; per course policy an HF/UCI dataset may not be used on CMU computing resources without approval, so all processing ran on a personal laptop |
| Language | English |

### A.2 Feature codebook **[WRITE IN YOUR OWN WORDS: descriptions + utility column]**

| Feature | Data type | Description (your words) | Utility for Scenario 01 (your words) |
|---|---|---|---|
| uniqueID | integer |  |  |
| drugName | string / categorical |  |  |
| condition | string / categorical |  |  |
| review | string / free text |  |  |
| rating | integer 1-10 (ordinal) |  |  |
| date | date |  |  |
| usefulCount | integer count |  |  |

### A.3 Feature profile (types, missing values, unique values)

|             | dtype          |   missing |   missing_pct |   unique |
|:------------|:---------------|----------:|--------------:|---------:|
| uniqueID    | int64          |         0 |         0     |   215063 |
| drugName    | str            |         0 |         0     |     3671 |
| condition   | str            |      1194 |         0.555 |      916 |
| review      | str            |         0 |         0     |   128478 |
| rating      | int64          |         0 |         0     |       10 |
| date        | datetime64[us] |         0 |         0     |     3579 |
| usefulCount | int64          |         0 |         0     |      397 |

### A.4 Data quality checks

| check                                              |   count |    pct |
|:---------------------------------------------------|--------:|-------:|
| condition contains scraped HTML '</span>' artifact |    1171 |   0.54 |
| review contains HTML entities (&#039; &amp; ...)   |  140533 |  65.35 |
| review wrapped in literal double quotes            |  215063 | 100    |
| exact duplicate (drugName, condition, review)      |      83 |   0.04 |
| duplicate review text (any drug/condition)         |   86585 |  40.26 |
| condition missing                                  |    1194 |   0.56 |
| review shorter than 5 words                        |    2210 |   1.03 |
| review texts appearing >1 times (distinct texts)   |   86068 |  40.02 |
| of which listed under >1 drugName (brand/generic)  |   86049 |  40.01 |
| of which listed under >1 condition                 |     485 |   0.23 |

*Evidence notes:*
- condition contains scraped HTML '</span>' artifact: 1,171 rows (0.54%)
- review contains HTML entities (&#039; &amp; ...): 140,533 rows (65.35%)
- review wrapped in literal double quotes: 215,063 rows (100.0%)
- exact duplicate (drugName, condition, review): 83 rows (0.04%)
- duplicate review text (any drug/condition): 86,585 rows (40.26%)
- condition missing: 1,194 rows (0.56%)
- review shorter than 5 words: 2,210 rows (1.03%)
- review texts appearing >1 times (distinct texts): 86,068 rows (40.02%)
-   of which listed under >1 drugName (brand/generic): 86,049 rows (40.01%)
-   of which listed under >1 condition: 485 rows (0.23%)
- Cleaning kept 214,078 of 215,063 rows (exact duplicates and reviews under 3 words removed; HTML entities unescaped; wrapping quotes stripped; scraped `</span>` conditions set to missing).
- The 40% 'duplicate review text' is almost entirely the SAME review listed under a brand name and a generic name. This must be de-duplicated before any train/test split, otherwise the test set leaks training text.

### A.5 Traffic Light Protocol labels (engineered feature `tlp_label`)

| TLP label | Rule used (this dataset) | Meaning for sharing | Count | Example from dataset |
|---|---|---|---|---|
| TLP:CLEAR | <= 8 words and no first-person pronoun (generic product remark) | may be shared publicly | 2,967 (1.39%) | *Topiramate / Bipolar Disorde*: "Stabilizing when used with Lamictal and Lexapro." |
| TLP:GREEN | default: a patient's own experience, no extra identifiers | share within the organisation / partner clinics | 157,803 (73.71%) | *Ibandronate / Osteoporosis*: "Taken one dose so far and do not believe that I will take no more. I took the pill on Sunday and still having severe pains in my stomach I don't like the way it makes me feel." |
| TLP:AMBER | mentions third parties (my son/wife...), explicit age, pregnancy, or a named doctor | restricted; de-identify before use | 43,076 (20.12%) | *Phentermine / Weight Loss*: "I am 56 years old, female weighed in today at 160.  Just started phentermine 37.5 mg today, 1/6/2017.  I also got a B12 shot.  I'm hoping to lose at least 15 pounds!  I appreciate everyone that takes time to post here be" |
| TLP:RED | crisis or highly sensitive content (suicidal ideation, overdose, HIV, abuse, substance dependence) or a sensitive condition | do not use for training without review; exclude by default | 10,232 (4.78%) | *Aripiprazole / Schizophrenia*: "I started abilify in January of this year, 10 mg. I'm now taking 15 mg. This medication has given me back my life.  No voices, no visual hallucinations. I still struggle with mild paranoia though. I can get through the" |

![Figure A-1. Engineered label distributions: sentiment_label and tlp_label.](outputs/part_a/figures/F9_engineered_labels.png)

*Figure A-1. Engineered label distributions: sentiment_label and tlp_label.*

### A.6 Descriptive statistics

**Numeric features (raw)**

|             |   count |   min |   max |   mean |   median |    std |   q1 |   q3 |   iqr |   range |   skewness |   kurtosis_excess |
|:------------|--------:|------:|------:|-------:|---------:|-------:|-----:|-----:|------:|--------:|-----------:|------------------:|
| rating      |  215063 |     1 |    10 |  6.99  |        8 |  3.276 |    5 |   10 |     5 |       9 |     -0.796 |            -0.896 |
| usefulCount |  215063 |     0 |  1291 | 28.001 |       16 | 36.346 |    6 |   36 |    30 |    1291 |      4.495 |            56.805 |

**Numeric features after cleaning, including engineered features**

|                        |   count |   min |   max |     mean |   median |       std |   q1 |   q3 |   iqr |   range |   skewness |   kurtosis_excess |
|:-----------------------|--------:|------:|------:|---------:|---------:|----------:|-----:|-----:|------:|--------:|-----------:|------------------:|
| rating                 |  214078 |     1 |    10 |    6.984 |        8 |     3.276 |    5 |   10 |     5 |       9 |     -0.792 |            -0.901 |
| usefulCount            |  214078 |     0 |  1291 |   28.062 |       16 |    36.404 |    6 |   36 |    30 |    1291 |      4.49  |            56.665 |
| review_char_len        |  214078 |     8 | 10431 |  449.116 |      446 |   234.455 |  257 |  676 |   419 |   10423 |      1.016 |            26.469 |
| review_word_len        |  214078 |     3 |  1894 |   84.98  |       85 |    44.633 |   49 |  126 |    77 |    1891 |      0.943 |            22.964 |
| review_sentence_count  |  214078 |     1 |    91 |    5.952 |        6 |     3.329 |    3 |    8 |     5 |      90 |      0.967 |             6.951 |
| drug_review_count      |  214078 |     1 |  4920 |  833.473 |      438 |  1114.69  |  129 | 1072 |   943 |    4919 |      2.238 |             4.629 |
| condition_review_count |  211720 |     1 | 38412 | 9780.82  |     3093 | 13837.3   |  813 | 8176 |  7363 |   38411 |      1.473 |             0.41  |
| review_year            |  214078 |  2008 |  2017 | 2013.95  |     2015 |     2.71  | 2012 | 2016 |     4 |       9 |     -0.701 |            -0.759 |

**Rating distribution**

|   rating |   count |   pct |
|---------:|--------:|------:|
|        1 |   28918 | 13.45 |
|        2 |    9265 |  4.31 |
|        3 |    8718 |  4.05 |
|        4 |    6671 |  3.1  |
|        5 |   10723 |  4.99 |
|        6 |    8462 |  3.93 |
|        7 |   12547 |  5.83 |
|        8 |   25046 | 11.65 |
|        9 |   36708 | 17.07 |
|       10 |   68005 | 31.62 |

**Categorical features: long-tail summary**

|           |   unique |   top1_share_pct |   top10_share_pct |   top50_share_pct |   values_with_<=5_rows |   values_with_1_row |   median_rows_per_value |
|:----------|---------:|-----------------:|------------------:|------------------:|-----------------------:|--------------------:|------------------------:|
| drugName  |     3671 |             2.29 |             13.33 |             32.99 |                   1783 |                 798 |                       6 |
| condition |      916 |            17.97 |             46.16 |             76.88 |                    341 |                 113 |                      11 |

**Top-20 drugs**

| drugName                           |   count |
|:-----------------------------------|--------:|
| Levonorgestrel                     |    4930 |
| Etonogestrel                       |    4421 |
| Ethinyl estradiol / norethindrone  |    3753 |
| Nexplanon                          |    2892 |
| Ethinyl estradiol / norgestimate   |    2790 |
| Ethinyl estradiol / levonorgestrel |    2503 |
| Phentermine                        |    2085 |
| Sertraline                         |    1868 |
| Escitalopram                       |    1747 |
| Mirena                             |    1673 |
| Implanon                           |    1506 |
| Gabapentin                         |    1415 |
| Bupropion                          |    1369 |
| Miconazole                         |    1344 |
| Venlafaxine                        |    1338 |
| Medroxyprogesterone                |    1308 |
| Citalopram                         |    1308 |
| Duloxetine                         |    1256 |
| Lexapro                            |    1250 |
| Bupropion / naltrexone             |    1249 |

**Top-20 conditions**

| condition                 |   count |
|:--------------------------|--------:|
| Birth Control             |   38436 |
| Depression                |   12164 |
| Pain                      |    8245 |
| Anxiety                   |    7812 |
| Acne                      |    7435 |
| Bipolar Disorde           |    5604 |
| Insomnia                  |    4904 |
| Weight Loss               |    4857 |
| Obesity                   |    4757 |
| ADHD                      |    4509 |
| Diabetes, Type 2          |    3362 |
| Emergency Contraception   |    3290 |
| High Blood Pressure       |    3104 |
| Vaginal Yeast Infection   |    3085 |
| Abnormal Uterine Bleeding |    2744 |
| Bowel Preparation         |    2498 |
| Smoking Cessation         |    2440 |
| ibromyalgia               |    2370 |
| Migraine                  |    2277 |
| Anxiety and Stress        |    2236 |

**Engineered categorical features**

| feature              | value     |   count |   pct |
|:---------------------|:----------|--------:|------:|
| sentiment_label      | POSITIVE  |  141490 | 66.09 |
| sentiment_label      | NEGATIVE  |   53450 | 24.97 |
| sentiment_label      | NEUTRAL   |   19138 |  8.94 |
| side_effect_flag     | 1         |  117930 | 55.09 |
| side_effect_flag     | 0         |   96148 | 44.91 |
| is_chronic_condition | 0         |  165581 | 77.35 |
| is_chronic_condition | 1         |   48497 | 22.65 |
| tlp_label            | TLP:GREEN |  157803 | 73.71 |
| tlp_label            | TLP:AMBER |   43076 | 20.12 |
| tlp_label            | TLP:RED   |   10232 |  4.78 |
| tlp_label            | TLP:CLEAR |    2967 |  1.39 |
| split                | train     |  160578 | 75.01 |
| split                | test      |   53500 | 24.99 |

![Figure A-2. Rating histogram: bimodal, mass at 10 and 1.](outputs/part_a/figures/F1_rating_hist.png)

*Figure A-2. Rating histogram: bimodal, mass at 10 and 1.*

![Figure A-3. usefulCount, linear and log scale.](outputs/part_a/figures/F2_usefulcount_hist.png)

*Figure A-3. usefulCount, linear and log scale.*

![Figure A-4. Boxplots of rating, usefulCount, review length.](outputs/part_a/figures/F3_boxplots.png)

*Figure A-4. Boxplots of rating, usefulCount, review length.*

![Figure A-5. Top-20 conditions.](outputs/part_a/figures/F4_top_conditions.png)

*Figure A-5. Top-20 conditions.*

![Figure A-6. Top-20 drugs.](outputs/part_a/figures/F5_top_drugs.png)

*Figure A-6. Top-20 drugs.*

![Figure A-7. Review length density.](outputs/part_a/figures/F6_review_length_density.png)

*Figure A-7. Review length density.*

![Figure A-8. Reviews per year.](outputs/part_a/figures/F7_reviews_per_year.png)

*Figure A-8. Reviews per year.*

![Figure A-9. Rating by top-8 conditions.](outputs/part_a/figures/F8_rating_by_condition.png)

*Figure A-9. Rating by top-8 conditions.*

![Figure A-10. Median usefulCount by rating.](outputs/part_a/figures/F13_usefulcount_by_rating.png)

*Figure A-10. Median usefulCount by rating.*


### A.7 Normal vs non-normal distributions **[WRITE IN YOUR OWN WORDS]**

| feature               |   skewness |   kurtosis_excess |   shapiro_W_n5000 |   shapiro_p |   dagostino_K2 |   dagostino_p |   shapiro_W_log1p | verdict    |
|:----------------------|-----------:|------------------:|------------------:|------------:|---------------:|--------------:|------------------:|:-----------|
| rating                |     -0.792 |            -0.901 |            0.8033 |    2.83e-61 |        47197.2 |             0 |            0.7316 | non-normal |
| usefulCount           |      4.49  |            56.665 |            0.6417 |    4.13e-73 |       210815   |             0 |            0.9842 | non-normal |
| review_word_len       |      0.943 |            22.964 |            0.9495 |    2.07e-38 |        78031.9 |             0 |            0.8772 | non-normal |
| review_char_len       |      1.016 |            26.469 |            0.9344 |    2.18e-42 |        83798.1 |             0 |            0.8674 | non-normal |
| review_sentence_count |      0.967 |             6.951 |            0.9619 |    2.41e-34 |        53776.6 |             0 |            0.9516 | non-normal |
| drug_review_count     |      2.238 |             4.629 |            0.6967 |    1.02e-69 |        93497.6 |             0 |            0.9666 | non-normal |

![Figure A-11. Q-Q plots against a normal distribution.](outputs/part_a/figures/F12_qq_plots.png)

*Figure A-11. Q-Q plots against a normal distribution.*

*Evidence notes:*
- `rating`: skew -0.792, excess kurtosis -0.901, Shapiro W=0.8033 (p=2.8e-61) -> non-normal
- `usefulCount`: skew 4.49, excess kurtosis 56.665, Shapiro W=0.6417 (p=4.1e-73) -> non-normal
- `review_word_len`: skew 0.943, excess kurtosis 22.964, Shapiro W=0.9495 (p=2.1e-38) -> non-normal
- `review_char_len`: skew 1.016, excess kurtosis 26.469, Shapiro W=0.9344 (p=2.2e-42) -> non-normal
- `review_sentence_count`: skew 0.967, excess kurtosis 6.951, Shapiro W=0.9619 (p=2.4e-34) -> non-normal
- `drug_review_count`: skew 2.238, excess kurtosis 4.629, Shapiro W=0.6967 (p=1.0e-69) -> non-normal
- rating is bimodal (32% at 10, 13% at 1), so it is not even unimodal; mean 6.98 vs median 8.
- usefulCount is right-skewed with a very long tail (kurtosis > 50): median 16, max 1,291.
- usefulCount becomes close to normal after a log1p transform (Shapiro W 0.64 -> 0.98), i.e. it is approximately log-normal. review_word_len does NOT improve under the log transform (W 0.95 -> 0.88): it is right-skewed with heavy tails but not log-normal.
- drugName and condition are long-tailed categoricals: top-10 conditions cover 46% of rows; 1783 of 3,671 drugs have 5 or fewer reviews.

### A.8 Impact of these distributions on model development **[WRITE IN YOUR OWN WORDS]**

*Evidence notes to build on:* class imbalance (66% POSITIVE / 25% NEGATIVE / 9% NEUTRAL after mapping); long-tail drugs and conditions (chronic conditions are 22.7% of rows); usefulCount outliers; brand/generic duplication; Birth Control alone is 18% of rows (the target population of Scenario 01 is chronic disease).

### A.9 Feature engineering

| New feature | Source feature(s) | How created | Data type | Why useful for Scenario 01 |
|---|---|---|---|---|
| condition_clean | condition | scraped `</span>` artifacts set to missing | string | removes 1,171 fake condition values that would become fake classes |
| review_char_len / review_word_len / review_sentence_count | review | character, whitespace-token, sentence-boundary counts after HTML unescape | integer | length is a proxy for information content; needed for chunking and sequence-length decisions |
| sentiment_label | rating | 1-4 NEGATIVE, 5-6 NEUTRAL, 7-10 POSITIVE | categorical (3) | the SFT target: patient-reported experience class |
| review_year | date | year component | integer | drift check: drugs and language change over 2008-2017 |
| side_effect_flag | review | regex over 20 common side-effect terms | binary | flags reviews that carry the safety information the tool must surface |
| drug_review_count / condition_review_count | drugName / condition_clean | group size | integer | exposure of each drug/condition in training data (long-tail risk) |
| is_chronic_condition | condition_clean | membership in a curated chronic-condition list | binary | identifies Scenario 01's target population for sub-group evaluation |
| tlp_label | review, condition_clean | rule-based Traffic Light Protocol classifier | categorical (4) | data-protection filter before training; governance evidence |

**Evaluation of the new features**

Side-effect flag vs rating:

|   side_effect_flag |   count |   mean |   median |   std |
|-------------------:|--------:|-------:|---------:|------:|
|                  0 |   96148 |  7.105 |        9 | 3.312 |
|                  1 |  117930 |  6.885 |        8 | 3.244 |

Side-effect flag vs sentiment label (% within flag group):

|   side_effect_flag |   NEGATIVE |   NEUTRAL |   POSITIVE |
|-------------------:|-----------:|----------:|-----------:|
|                  0 |       24.3 |       7.9 |       67.8 |
|                  1 |       25.5 |       9.8 |       64.7 |

Chronic-condition flag vs rating:

|   is_chronic_condition |   count |   mean |   median |   std |
|-----------------------:|--------:|-------:|---------:|------:|
|                      0 |  165581 |  6.939 |        8 | 3.299 |
|                      1 |   48497 |  7.139 |        8 | 3.193 |

TLP label vs rating:

| tlp_label   |   count |   mean |   median |
|:------------|--------:|-------:|---------:|
| TLP:AMBER   |   43076 |  7.017 |        8 |
| TLP:CLEAR   |    2967 |  7.176 |        9 |
| TLP:GREEN   |  157803 |  6.951 |        8 |
| TLP:RED     |   10232 |  7.3   |        9 |

Spearman correlations:

|                       |   rating |   usefulCount |   review_word_len |   review_sentence_count |   drug_review_count |   side_effect_flag |
|:----------------------|---------:|--------------:|------------------:|------------------------:|--------------------:|-------------------:|
| rating                |    1     |         0.283 |             0.008 |                   0.027 |              -0.064 |             -0.051 |
| usefulCount           |    0.283 |         1     |            -0.031 |                   0.011 |              -0.156 |             -0.023 |
| review_word_len       |    0.008 |        -0.031 |             1     |                   0.807 |               0.224 |              0.255 |
| review_sentence_count |    0.027 |         0.011 |             0.807 |                   1     |               0.192 |              0.232 |
| drug_review_count     |   -0.064 |        -0.156 |             0.224 |                   0.192 |               1     |              0.165 |
| side_effect_flag      |   -0.051 |        -0.023 |             0.255 |                   0.232 |               0.165 |              1     |

![Figure A-12. Rating distribution by side_effect_flag.](outputs/part_a/figures/F10_rating_by_sideeffect_flag.png)

*Figure A-12. Rating distribution by side_effect_flag.*
![Figure A-13. Rating distribution, chronic vs other conditions.](outputs/part_a/figures/F11_rating_chronic_vs_other.png)

*Figure A-13. Rating distribution, chronic vs other conditions.*

*Evidence notes:*
- Reviews with a side-effect keyword rate the drug lower on average (6.88 vs 7.11) but the flag alone barely separates sentiment (65% vs 68% POSITIVE): patients report side effects even when satisfied, so the keyword flag is a weak label and the model must read context.
- Chronic-condition reviews are slightly MORE positive (7.14 vs 6.94) and are 22.7% of rows.
- usefulCount correlates with rating (Spearman 0.28): positive reviews get more 'useful' votes, so usefulCount is not a neutral quality signal.
- review length is uncorrelated with rating (0.01) and correlated with side_effect_flag (0.26): longer reviews carry more safety content.

---
## B. Tokenization and Embedding Experiments

Sample: 800 de-duplicated reviews, 100 per condition from 8 conditions (4 high-frequency: Birth Control, Depression, Pain, Acne; 4 chronic: Diabetes Type 2, High Blood Pressure, High Cholesterol, Insomnia), 20-220 words, seed 42. Sentiment mix in sample: {'NEGATIVE': 232, 'NEUTRAL': 92, 'POSITIVE': 476}.

### B.1 Tokenization **[explain your choice in your own words]**

| strategy                   |   units_per_review_mean |   units_per_review_median | vocab_size   | note                         |
|:---------------------------|------------------------:|--------------------------:|:-------------|:-----------------------------|
| word                       |                   85.88 |                      83   | 5092         | starter tokenizer            |
| word_nostop                |                   42.82 |                      42   | 4945         | stopwords removed            |
| subword (all-MiniLM-L6-v2) |                  108    |                     103.5 | 4996         | model max_seq_length=256     |
| fixed_chunk(40)            |                    2.62 |                       3   | -            | chunks of words, mean-pooled |

How the dense model's WordPiece tokenizer splits common chronic-disease drug names:

| drug                | wordpiece_tokens                      |   n_pieces |
|:--------------------|:--------------------------------------|-----------:|
| metformin           | met ##form ##in                       |          3 |
| lisinopril          | li ##sin ##op ##ril                   |          4 |
| atorvastatin        | at ##or ##vas ##tat ##in              |          5 |
| amlodipine          | am ##lo ##di ##pine                   |          4 |
| sertraline          | ser ##tral ##ine                      |          3 |
| gabapentin          | ga ##ba ##pen ##tin                   |          4 |
| levonorgestrel      | lev ##ono ##rge ##strel               |          4 |
| hydrochlorothiazide | hydro ##ch ##lor ##oth ##ia ##zi ##de |          7 |
| ibuprofen           | ib ##up ##ro ##fen                    |          4 |
| insulin             | insulin                               |          1 |
| aspirin             | as ##pi ##rin                         |          3 |
| warfarin            | war ##far ##in                        |          3 |

![Figure B-1. Subword tokens per word and per review, with the model's 256-token limit.](outputs/part_b/figures/G4_subword_stats.png)

*Figure B-1. Subword tokens per word and per review, with the model's 256-token limit.*

*Evidence notes:*
- Stopwords are 49% of word tokens in patient reviews.
- WordPiece produces 1.26 subword tokens per word; 0 of 800 sampled reviews exceed the 256-token limit (sample was capped at 220 words; in the full data 1.6% of reviews are longer than 220 words).
- Drug names are almost never single tokens (metformin -> met ##form ##in; hydrochlorothiazide -> 7 pieces); only 'insulin' is in the vocabulary. The general-domain tokenizer has no lexical unit for the entities the scenario cares about.

### B.2 Embedding models **[explain your choice in your own words]**

|               |   dimension |   fraction_zero_entries |
|:--------------|------------:|------------------------:|
| bow           |        5092 |                  0.9878 |
| tfidf         |        5092 |                  0.9878 |
| bow_nostop    |        4945 |                  0.9924 |
| tfidf_nostop  |        4945 |                  0.9924 |
| dense         |         384 |                  0      |
| dense_chunked |         384 |                  0      |

Dense model used: `sentence-transformers/all-MiniLM-L6-v2` (EmbeddingGemma is gated on Hugging Face and needs licence approval; the script accepts `--dense-model models/embedders/google/embeddinggemma-300m` once downloaded).

Top TF-IDF terms per condition (what the sparse representation 'sees'):

| condition           | top_tfidf_terms                                               |
|:--------------------|:--------------------------------------------------------------|
| Birth Control       | i, the, and, my, it, was, a, to, period, have, pill, this     |
| Depression          | i, and, to, the, it, depression, was, a, my, for, of, anxiety |
| Pain                | pain, i, the, to, and, a, for, my, have, it, was, me          |
| Acne                | i, acne, and, my, it, the, a, skin, to, on, was, but          |
| Diabetes, Type 2    | i, the, to, and, my, have, a, in, was, on, sugar, it          |
| High Blood Pressure | i, pressure, my, and, the, to, blood, it, a, bp, was, of      |
| High Cholesterol    | i, cholesterol, and, to, my, the, a, of, it, for, have, was   |
| Insomnia            | i, sleep, it, and, to, night, the, a, ambien, for, me, have   |

### B.3 Similarity results (cosine)

| method        |   cond_nn1_match |   cond_nn5_majority |   sent_nn1_match |   sent_nn5_majority |   cond_within_mean |   cond_between_mean |   cond_gap |   cond_effect_d |   sent_gap |   sent_effect_d |   pairs_mean |   pairs_median |   pairs_min |   pairs_max |
|:--------------|-----------------:|--------------------:|-----------------:|--------------------:|-------------------:|--------------------:|-----------:|----------------:|-----------:|----------------:|-------------:|---------------:|------------:|------------:|
| bow           |           0.385  |              0.4375 |           0.5    |              0.5525 |             0.3817 |              0.3578 |     0.0239 |          0.1828 |     0.0022 |          0.0169 |       0.3608 |         0.3661 |      0      |      0.7703 |
| tfidf         |           0.6475 |              0.7425 |           0.5688 |              0.62   |             0.1203 |              0.0955 |     0.0249 |          0.4826 |     0.0017 |          0.0354 |       0.0986 |         0.0941 |      0      |      0.4842 |
| bow_nostop    |           0.63   |              0.725  |           0.5362 |              0.605  |             0.1032 |              0.0558 |     0.0474 |          0.752  |     0.0023 |          0.0415 |       0.0617 |         0.0508 |      0      |      0.493  |
| tfidf_nostop  |           0.7188 |              0.8075 |           0.5638 |              0.6025 |             0.0562 |              0.0271 |     0.0292 |          0.781  |     0.0013 |          0.0408 |       0.0307 |         0.0234 |      0      |      0.5059 |
| dense         |           0.8588 |              0.8888 |           0.5712 |              0.615  |             0.4673 |              0.3207 |     0.1465 |          1.2877 |     0.0004 |          0.0034 |       0.3389 |         0.3339 |     -0.1726 |      0.8792 |
| dense_chunked |           0.8012 |              0.8412 |           0.5775 |              0.6312 |             0.496  |              0.3663 |     0.1297 |          1.0736 |    -0.0006 |         -0.0051 |       0.3823 |         0.3786 |     -0.1057 |      0.888  |
| chance_level  |           0.125  |              0.125  |           0.595  |              0.595  |           nan      |            nan      |   nan      |        nan      |   nan      |        nan      |     nan      |       nan      |    nan      |    nan      |

Per-condition 1-NN accuracy:

| condition           |   dense |   tfidf |
|:--------------------|--------:|--------:|
| Acne                |    0.92 |    0.72 |
| Birth Control       |    0.82 |    0.78 |
| Depression          |    0.81 |    0.54 |
| Diabetes, Type 2    |    0.87 |    0.66 |
| High Blood Pressure |    0.83 |    0.43 |
| High Cholesterol    |    0.87 |    0.54 |
| Insomnia            |    0.88 |    0.83 |
| Pain                |    0.87 |    0.68 |

![Figure B-2. Leave-one-out 1-NN match rates by method.](outputs/part_b/figures/G1_nn_match_rates.png)

*Figure B-2. Leave-one-out 1-NN match rates by method.*
![Figure B-3. Pairwise cosine, same vs different condition.](outputs/part_b/figures/G2_similarity_distributions.png)

*Figure B-3. Pairwise cosine, same vs different condition.*
![Figure B-4. Mean cosine between conditions, TF-IDF vs dense.](outputs/part_b/figures/G3_condition_heatmaps.png)

*Figure B-4. Mean cosine between conditions, TF-IDF vs dense.*
![Figure B-5. Per-condition 1-NN accuracy.](outputs/part_b/figures/G5_per_condition_nn.png)

*Figure B-5. Per-condition 1-NN accuracy.*

Most / least similar pairs per method (truncated):

| method | kind | cosine | cond_i | cond_j | review_i | review_j |
|---|---|---|---|---|---|---|
| bow | most_similar | 0.7703 | Pain | High Cholesterol | This medicine was great. Like you all say I got my life back, but then I started to get to where I felt like I... | This medication landed me in the ER after a week. I was having so much back pain that I could not stand or wal... |
| bow | least_similar | 0.0 | Depression | Diabetes, Type 2 | Made me sleepy and I didn't care about anything. However, I find that a small dosage combined with Wellbutrin ... | Been taking 30mg of Pioglitazone daily. Noticed increased anxiety, weight gain, difficulty sleeping at night. ... |
| tfidf | most_similar | 0.4842 | Insomnia | Insomnia | I've tried many sleep aids. I'm currently taking a generic AMBIEN-CR 12.5mg. I had previously used both Brand ... | Due to issues with my insurance I recently have had to switch from Ambien CR to regular Ambien. Although the m... |
| tfidf | least_similar | 0.0 | Acne | Diabetes, Type 2 | Cleared my acne up so fast. Its nice. Other than that everything is going good except I'm turning into a grump... | After two weeks, increase in blood sugar by 50 and back pain from day one, I will finish this month and see.... |
| dense | most_similar | 0.8792 | High Cholesterol | High Cholesterol | I take 10 mg daily. I have muscle, back as well as joint pain. Not sure if it is caused by statin. Will keep m... | I have only been taking this medicine for 1 month, 10 mg every other day.  Already I am experiencing severe jo... |
| dense | least_similar | -0.1726 | Birth Control | High Blood Pressure | Getting it put in didn't hurt much. I would have a continuous light period for months at a time, which made me... | My blood pressure is down, but I've gained almost 10 lbs in 1 year and I don't sleep well. Also always feel ti... |

Clinical query retrieval (top-3, TF-IDF vs dense): see `outputs/part_b/tables/09_clinical_query_retrieval.csv`.

| query | method | rank | cosine | drug | condition | rating | review |
|---|---|---|---|---|---|---|---|
| patient reports dizziness and fatigue after starting blood p | tfidf | 1 | 0.1977 | Norvasc | High Blood Pressure | 1 | I hate Norvasc! It made my blood pressure worse. I was always borderline with my blood pressure. It didn't help that I w... |
| patient reports dizziness and fatigue after starting blood p | tfidf | 2 | 0.1876 | Prazosin | High Blood Pressure | 5 | This medication acts fast, causes me be be jittery, my blood pressure is low and then settles down after a few hours, th... |
| patient reports dizziness and fatigue after starting blood p | tfidf | 3 | 0.1711 | Valsartan | High Blood Pressure | 1 | After about six weeks of starting Diovan 80 mg my blood pressure began to go higher.  My dosage was increased to 160 mg.... |
| patient reports dizziness and fatigue after starting blood p | dense | 1 | 0.6753 | Clonidine | High Blood Pressure | 1 | Experienced sudden dizzy spells and lightheadedness with this medicine.  Also irregular heart beats and fatigue.  Quit i... |
| patient reports dizziness and fatigue after starting blood p | dense | 2 | 0.6163 | Hydralazine | High Blood Pressure | 1 | I have hypertension. I have been taking a combination of Amlodipine / benazepril. While this has been somewhat effective... |
| patient reports dizziness and fatigue after starting blood p | dense | 3 | 0.6053 | Prazosin | High Blood Pressure | 5 | This medication acts fast, causes me be be jittery, my blood pressure is low and then settles down after a few hours, th... |
| stomach cramps and diarrhea in the first weeks of taking met | tfidf | 1 | 0.2511 | Dulaglutide | Diabetes, Type 2 | 9 | Give Trulicity at least 4 weeks. I like many others experienced upset stomach, nausea and fatigue the first month where ... |
| stomach cramps and diarrhea in the first weeks of taking met | tfidf | 2 | 0.2275 | Dulaglutide | Diabetes, Type 2 | 8 | I've been DM Type 2 for 15 years and in the last 2 years taking Levemir, Novolog and Janumet.  My last A1c was 9.1.  I s... |
| stomach cramps and diarrhea in the first weeks of taking met | tfidf | 3 | 0.2168 | Medroxyprogesterone | Birth Control | 4 | I've only had my first shot. The first month was okay. I was always hungry, severe backaches & headaches, and few hot fl... |
| stomach cramps and diarrhea in the first weeks of taking met | dense | 1 | 0.6433 | Metformin | Diabetes, Type 2 | 8 | I have been taking Metformin for about a month and have not had any side effects. I also take glyburide with it.... |
| stomach cramps and diarrhea in the first weeks of taking met | dense | 2 | 0.5965 | Dulaglutide | Diabetes, Type 2 | 9 | Give Trulicity at least 4 weeks. I like many others experienced upset stomach, nausea and fatigue the first month where ... |
| stomach cramps and diarrhea in the first weeks of taking met | dense | 3 | 0.5913 | Sitagliptin | Diabetes, Type 2 | 10 | I couldn't tolerate metformin at all.  Moving to Januvia was wonderful. I've been on it for a year and a half and have h... |
| medication stopped working and symptoms came back | tfidf | 1 | 0.1986 | Praluent | High Cholesterol | 1 | I am a 35 year old male is very good shape with genetically high cholesterol. I have been extremely intolerant of statin... |
| medication stopped working and symptoms came back | tfidf | 2 | 0.1903 | Minocycline | Acne | 6 | I am 19 years old and I was prescribed minocyline for my hormonal acne. I saw results in the first 2 weeks. My skin was ... |
| medication stopped working and symptoms came back | tfidf | 3 | 0.1656 | Spironolactone | High Blood Pressure | 1 | At first this medication worked. Then it all went down hill. Recently went into the ER with a BP of 205/115. This medici... |
| medication stopped working and symptoms came back | dense | 1 | 0.6162 | Simvastatin | High Cholesterol | 1 | I took the medication for three days and quit. I became so ill and could not keep down food for three days. I was in pai... |
| medication stopped working and symptoms came back | dense | 2 | 0.5299 | Fluoxetine | Depression | 10 | I was against taking any kind of medication but after various breakdowns and my job on the line I took Cipralex which di... |
| medication stopped working and symptoms came back | dense | 3 | 0.5266 | Clonidine | High Blood Pressure | 1 | Experienced sudden dizzy spells and lightheadedness with this medicine.  Also irregular heart beats and fatigue.  Quit i... |
| no side effects and cholesterol numbers improved | tfidf | 1 | 0.3153 | Ezetimibe / simvastatin | High Cholesterol | 10 | Vytorin is working wonders for me.   Finding the right cholesterol lowering medicine was the key.  Lipitor gave me the l... |
| no side effects and cholesterol numbers improved | tfidf | 2 | 0.2744 | Livalo | High Cholesterol | 10 | I had a horrible side effect with Lipitor so the Dr put me on Livalo.  No side effects and was able to bring my total ch... |
| no side effects and cholesterol numbers improved | tfidf | 3 | 0.2666 | Glyxambi | Diabetes, Type 2 | 8 | I have been on this medicine for almost a year. It really does bring your diabetes under control. My A1C has remained a ... |
| no side effects and cholesterol numbers improved | dense | 1 | 0.6615 | Zetia | High Cholesterol | 10 | Cholesterol numbers are in the normal range for the first time EVER. I take it with Livalo (the cost for both medication... |
| no side effects and cholesterol numbers improved | dense | 2 | 0.6499 | Slo-Niacin | High Cholesterol | 10 | Two tabs (1000 mg) lowered my total cholesterol by 30 points. Added one more tablet daily (1500 mg) which will hopefully... |
| no side effects and cholesterol numbers improved | dense | 3 | 0.6492 | Zetia | High Cholesterol | 1 | I experienced severe abdominal cramps and pain as well as diarrhea.  I also experienced weakness and fatigue. I could no... |

*Evidence notes:*
- `bow`: 1-NN same-condition 0.385 (chance 0.125), 1-NN same-sentiment 0.500 (majority-class baseline 0.595), within-minus-between condition gap 0.024 (Cohen d 0.18), sentiment d 0.017
- `tfidf`: 1-NN same-condition 0.647 (chance 0.125), 1-NN same-sentiment 0.569 (majority-class baseline 0.595), within-minus-between condition gap 0.025 (Cohen d 0.48), sentiment d 0.035
- `tfidf_nostop`: 1-NN same-condition 0.719 (chance 0.125), 1-NN same-sentiment 0.564 (majority-class baseline 0.595), within-minus-between condition gap 0.029 (Cohen d 0.78), sentiment d 0.041
- `dense`: 1-NN same-condition 0.859 (chance 0.125), 1-NN same-sentiment 0.571 (majority-class baseline 0.595), within-minus-between condition gap 0.146 (Cohen d 1.29), sentiment d 0.003
- `dense_chunked`: 1-NN same-condition 0.801 (chance 0.125), 1-NN same-sentiment 0.578 (majority-class baseline 0.595), within-minus-between condition gap 0.130 (Cohen d 1.07), sentiment d -0.005
- Every representation groups reviews by CONDITION far above chance, and NONE groups them by SENTIMENT (all sentiment 1-NN rates are at or below the majority-class baseline). Generic embeddings encode topic, not patient experience. This is the direct motivation for supervised fine-tuning on the rating-derived label.
- Removing stopwords raises BoW from 0.39 to 0.63 and TF-IDF from 0.65 to 0.72: half of the tokens are noise for the sparse methods.
- The dense model is best overall (0.86) and, importantly, closes the gap on chronic conditions where TF-IDF is weakest (High Blood Pressure 0.43 -> 0.83, High Cholesterol 0.54 -> 0.87).
- Chunking (40-word chunks, mean-pooled) LOWERS dense accuracy (0.86 -> 0.80): patient reviews are short enough to embed whole; chunking splits context.
- BoW's most-similar pair joins a Pain review and a High Cholesterol review because both are long and share function words; dense's most-similar pair is two statin users describing muscle pain, i.e. a genuinely clinically related pair.
- In the dense heatmap Depression and Insomnia are the closest cross-condition pair (0.42), and High Blood Pressure sits close to Depression (0.38): the representation reflects symptom overlap (sleep, fatigue), which is useful for the tool but also a source of confusion.
- Query retrieval: for 'dizziness and fatigue after starting blood pressure medication' TF-IDF returns reviews that merely repeat 'blood pressure'; dense returns a clonidine review that literally describes dizzy spells and fatigue.

### B.4 Interpretation **[WRITE IN YOUR OWN WORDS: performance, generalisation, bias & representation, data quality, future decisions]**


---
## C. Research Plan Evidence: SFT data preparation (no training yet)

| split          |   rows |   NEGATIVE |   NEUTRAL |   POSITIVE |   chronic_rows |   n_drugs |   POSITIVE_pct |
|:---------------|-------:|-----------:|----------:|-----------:|---------------:|----------:|---------------:|
| train          |  87414 |      21726 |      7752 |      57936 |          19723 |      2596 |           66.3 |
| valid          |  10927 |       2716 |       969 |       7242 |           2426 |      1367 |           66.3 |
| test_seen      |  10928 |       2716 |       970 |       7242 |           2416 |      1366 |           66.3 |
| test_unseen    |  11743 |       2983 |      1187 |       7573 |           2352 |       305 |           64.5 |
| train_balanced |  23752 |       8000 |      7752 |       8000 |           5283 |      1748 |           33.7 |

Example SFT record:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a clinical decision-support assistant for clinicians and pharmacists. Read a patient's self-reported medication experience and classify it as POSITIVE, NEUTRAL, or NEGATIVE. Answer with the label only. This is not medical advice and does not replace a clinician's judgement."
    },
    {
      "role": "user",
      "content": "Drug: Pregabalin\nCondition: Diabetic Peripheral Neuropathy\nPatient report: \"Started taking Lyrica yesterday for severe diabetic neuropathy in both feet and lower legs. I am currently on Suboxone for a 15 year addiction to opiates and was concerned that Lyrica wouldn't work because my body is so screwed up from all the years of opiate use and most medications do not work well for me. Thirty minutes after the first dose I noticed a significant improvement in the neuropathic pain. My bedtime dose was another 75 milligram capsule and I slept better than I have slept in months! This medicine has been a God send for me so far. Hoping that it continues to work well for me! Thanks Pfizer!\""
    },
    {
      "role": "assistant",
      "content": "POSITIVE"
    }
  ],
  "meta": {
    "uniqueID": 19525,
    "rating": 10,
    "condition": "Diabetic Peripheral Neuropathy",
    "is_chronic": 0,
    "tlp": "TLP:GREEN"
  }
}
```

Preparation log:

```json
{
  "rows_clean": 214078,
  "rows_after_drop_TLP_RED": 203846,
  "rows_after_text_dedup": 122190,
  "rows_after_drop_missing_condition": 121012,
  "n_drugs_total": 3052,
  "n_drugs_held_out": 305,
  "written_train": 87414,
  "written_valid": 10927,
  "written_test_seen": 10928,
  "written_test_unseen": 11743,
  "written_train_balanced": 23752
}
```

*Evidence notes:*
- TLP:RED rows are excluded before any training (patient-data protection plan, Scenario 01 optional requirement).
- Brand/generic duplicates are collapsed so the unseen-drug test cannot leak text from training.
- Evaluation splits keep the natural 66/25/9 imbalance; only the training set is capped per class.

---
## Reproducibility

```bash
cd as01_v02
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# data: data/raw/train.jsonl, data/raw/test.jsonl from https://huggingface.co/datasets/lewtun/drug-reviews
python part_a_profile.py        # outputs/part_a
python part_b_experiments.py    # outputs/part_b
python prepare_sft_data.py      # outputs/part_c, data/sft
python build_memo_appendix.py   # this file
python main_test.py             # starter-kit self-test (17 pass, 1 skip)
```

Seeds: 42 everywhere. Hardware used: Apple M4 Pro, 48 GB, CPU/MPS. Python 3.13.7; package versions in requirements.txt.