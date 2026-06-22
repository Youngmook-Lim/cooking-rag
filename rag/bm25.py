import re
from collections import Counter
from math import log


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        tokenized = [tokenize(d) for d in docs]
        self.doc_len = [len(t) for t in tokenized]
        self.n = len(docs)
        self.avgdl = sum(self.doc_len) / self.n if self.n else 0.0
        # inverted index: term -> [(doc_id, term_freq), ...]
        self.postings: dict[str, list[tuple[int, int]]] = {}
        df: Counter = Counter()
        for doc_id, tokens in enumerate(tokenized):
            for term, freq in Counter(tokens).items():
                self.postings.setdefault(term, []).append((doc_id, freq))
                df[term] += 1
        self.idf = {
            term: log(1 + (self.n - d + 0.5) / (d + 0.5))
            for term, d in df.items()
        }

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = {}
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, freq in self.postings[term]:
                denom = freq + self.k1 * (
                    1 - self.b + self.b * self.doc_len[doc_id] / self.avgdl
                )
                scores[doc_id] = scores.get(doc_id, 0.0) + \
                    idf * (freq * (self.k1 + 1)) / denom
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
