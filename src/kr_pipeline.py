"""
Clean, importable versions of the starter-kit knowledge-representation helpers
(src/tokenizers.py, src/embedders_no_ai.py, src/similarity_metrics.py).

The starter files define their helpers with a `self` argument outside a class and
call each other as bare globals, so they can only run through the runtime patches
in main_test.py. This module keeps the SAME algorithms (word tokenizer, sentence
splitter, fixed-size chunking, Bag-of-Words, TF-IDF with +1 smoothing, sparse and
dense cosine similarity) as plain functions so the Part B experiment script can
import them directly. Base Python only (no numpy needed for the sparse path).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence

# ---------------------------------------------------------------- tokenization
STOPWORDS = set("""
a an the and or but if then than so of to in on at by for with from as is are was
were be been being am do does did doing have has had having i me my mine we our
ours you your yours he him his she her hers it its they them their theirs this
that these those there here what which who whom when where why how all any both
each few more most other some such no nor not only own same too very s t can will
just don should now d ll m o re ve y ain aren couldn didn doesn hadn hasn haven
isn ma mightn mustn needn shan shouldn wasn weren won wouldn would could also
into over under again further once because while about against between through
during before after above below up down out off very
""".split())


def tokenize_words(text: str) -> List[str]:
    """Lowercase, strip punctuation (keep apostrophes), split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return text.split()


def tokenize_words_nostop(text: str) -> List[str]:
    """Word tokenizer + stopword removal (our own variant of the starter method)."""
    return [t for t in tokenize_words(text) if t not in STOPWORDS and len(t) > 1]


def tokenize_sentences(text: str) -> List[List[str]]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [tokenize_words(s) for s in sentences if s]


def chunk_fixed_size(text: str, chunk_size: int = 40) -> List[List[str]]:
    tokens = tokenize_words(text)
    return [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]


# ---------------------------------------------------------------- sparse embeddings
def build_vocabulary(list_of_token_lists: Iterable[Sequence[str]]) -> Dict[str, int]:
    vocab = set()
    for tokens in list_of_token_lists:
        vocab.update(tokens)
    return {tok: i for i, tok in enumerate(sorted(vocab))}


def embed_bow(tokens: Sequence[str], vocab: Dict[str, int]) -> Dict[int, float]:
    vec: Dict[int, float] = defaultdict(float)
    for tok in tokens:
        if tok in vocab:
            vec[vocab[tok]] += 1.0
    return dict(vec)


def compute_idf(list_of_token_lists: Sequence[Sequence[str]], vocab: Dict[str, int]) -> Dict[int, float]:
    n_docs = len(list_of_token_lists)
    df = Counter()
    for tokens in list_of_token_lists:
        for tok in set(tokens):
            df[tok] += 1
    return {idx: math.log(n_docs / (1 + df.get(tok, 0))) + 1.0 for tok, idx in vocab.items()}


def embed_tfidf(tokens: Sequence[str], vocab: Dict[str, int], idf: Dict[int, float]) -> Dict[int, float]:
    return {idx: cnt * idf[idx] for idx, cnt in embed_bow(tokens, vocab).items()}


def embed_corpus(list_of_token_lists: Sequence[Sequence[str]], strategy: str):
    vocab = build_vocabulary(list_of_token_lists)
    if strategy == "bow":
        return [embed_bow(t, vocab) for t in list_of_token_lists], vocab, {}
    if strategy == "tfidf":
        idf = compute_idf(list_of_token_lists, vocab)
        return [embed_tfidf(t, vocab, idf) for t in list_of_token_lists], vocab, {"idf": idf}
    raise ValueError(f"Unknown strategy {strategy}")


# ---------------------------------------------------------------- similarity
def cosine_similarity_sparse(a: Dict[int, float], b: Dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(i, 0.0) for i, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def cosine_similarity_dense(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def sparse_to_dense_matrix(embeddings: Sequence[Dict[int, float]], dim: int):
    """Convert list of sparse dicts to a (n, dim) numpy array for fast matrix ops."""
    import numpy as np
    m = np.zeros((len(embeddings), dim), dtype=np.float32)
    for r, vec in enumerate(embeddings):
        for i, v in vec.items():
            m[r, i] = v
    return m


def cosine_matrix(m):
    """Pairwise cosine similarity for a dense (n, d) numpy array."""
    import numpy as np
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    u = m / norms
    return u @ u.T
