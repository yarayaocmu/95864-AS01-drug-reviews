| strategy                   |   units_per_review_mean |   units_per_review_median | vocab_size   | note                         |
|:---------------------------|------------------------:|--------------------------:|:-------------|:-----------------------------|
| word                       |                   85.88 |                      83   | 5092         | starter tokenizer            |
| word_nostop                |                   42.82 |                      42   | 4945         | stopwords removed            |
| subword (all-MiniLM-L6-v2) |                  108    |                     103.5 | 4996         | model max_seq_length=256     |
| fixed_chunk(40)            |                    2.62 |                       3   | -            | chunks of words, mean-pooled |