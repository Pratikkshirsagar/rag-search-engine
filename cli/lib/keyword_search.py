import math
import os
import pickle
import string
from collections import Counter

from nltk.stem import PorterStemmer

from .search_utils import (
    BM25_B,
    BM25_K1,
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    Movie,
    load_movies,
    load_stopwords,
)

stemmer = PorterStemmer()


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, Movie] = {}
        self.term_frequencies: dict[int, Counter[str]] = {}
        self.doc_lengths: dict[int, int] = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")
        self.stopwords = {preprocess_text(word) for word in load_stopwords()}

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text, self.stopwords)
        self.doc_lengths[doc_id] = len(tokens)
        self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            self.index.setdefault(token, set()).add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        """Return sorted document IDs for a single preprocessed token."""
        return sorted(self.index.get(term, set()))

    def get_tf(self, doc_id: int, term: str) -> int:
        """Return the count of a preprocessed token, or zero if it is absent."""
        frequencies = self.term_frequencies.get(doc_id)
        if frequencies is None:
            return 0
        return frequencies[term]

    def get_idf(self, term: str) -> float:
        """Return the smoothed inverse document frequency of a preprocessed token."""
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index.get(term, set()))
        return math.log((total_doc_count + 1) / (term_match_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        """Return the BM25 inverse document frequency of a preprocessed token."""
        total_doc_count = len(self.docmap)
        term_match_doc_count = len(self.index.get(term, set()))
        return math.log(
            (total_doc_count - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ) -> float:
        """Return length-normalized BM25 TF for a preprocessed token."""
        if not math.isfinite(k1) or k1 < 0:
            raise ValueError("BM25 k1 must be a finite, non-negative number.")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 b must be a finite number between 0 and 1.")
        tf = self.get_tf(doc_id, term)
        if tf == 0:
            return 0.0
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1.0
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (self.doc_lengths[doc_id] / avg_doc_length)
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        """Return the full BM25 score for one document and preprocessed token."""
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[tuple[int, float]]:
        """Score every document and return the top (document ID, score) pairs."""
        if limit <= 0:
            return []

        query_tokens = tokenize_text(query, self.stopwords)
        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            scores[doc_id] = 0.0
            for token in query_tokens:
                scores[doc_id] += self.bm25(doc_id, token)

        ranked_documents = sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )
        return ranked_documents[:limit]

    def build(self) -> None:
        self.index.clear()
        self.docmap.clear()
        self.term_frequencies.clear()
        self.doc_lengths.clear()
        for movie in load_movies():
            doc_id = movie["id"]
            self.docmap[doc_id] = movie
            self.__add_document(doc_id, f"{movie['title']} {movie['description']}")

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(os.path.join(CACHE_DIR, "index.pkl"), "wb") as file:
            pickle.dump(self.index, file)
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "wb") as file:
            pickle.dump(self.docmap, file)
        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "wb") as file:
            pickle.dump(self.term_frequencies, file)
        with open(self.doc_lengths_path, "wb") as file:
            pickle.dump(self.doc_lengths, file)

    def load(self) -> None:
        """Load the local cache, raising FileNotFoundError if any file is missing."""
        with open(os.path.join(CACHE_DIR, "index.pkl"), "rb") as file:
            index = pickle.load(file)
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "rb") as file:
            docmap = pickle.load(file)
        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "rb") as file:
            term_frequencies = pickle.load(file)
        with open(self.doc_lengths_path, "rb") as file:
            doc_lengths = pickle.load(file)
        self.index = index
        self.docmap = docmap
        self.term_frequencies = term_frequencies
        self.doc_lengths = doc_lengths


def build_command() -> None:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()
    print("Index built and saved.")


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[Movie]:
    if limit <= 0:
        return []

    inverted_index = InvertedIndex()
    inverted_index.load()
    query_tokens = tokenize_text(query, inverted_index.stopwords)

    results = []
    seen_doc_ids: set[int] = set()
    for token in query_tokens:
        for doc_id in inverted_index.get_documents(token):
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            results.append(inverted_index.docmap[doc_id])
            if len(results) >= limit:
                return results

    return results


def tf_command(doc_id: int, term: str) -> int:
    inverted_index = InvertedIndex()
    inverted_index.load()
    token = tokenize_term(term, inverted_index.stopwords)
    return inverted_index.get_tf(doc_id, token)


def idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    token = tokenize_term(term, inverted_index.stopwords)
    return inverted_index.get_idf(token)


def bm25_idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    token = tokenize_term(term, inverted_index.stopwords)
    return inverted_index.get_bm25_idf(token)


def bm25_tf_command(
    doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    token = tokenize_term(term, inverted_index.stopwords)
    return inverted_index.get_bm25_tf(doc_id, token, k1, b)


def tfidf_command(doc_id: int, term: str) -> float:
    inverted_index = InvertedIndex()
    inverted_index.load()
    token = tokenize_term(term, inverted_index.stopwords)
    tf = inverted_index.get_tf(doc_id, token)
    idf = inverted_index.get_idf(token)
    return tf * idf


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenize_text(text: str, stopwords: set[str]) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token and token not in stopwords:
            valid_tokens.append(stemmer.stem(token))
    return valid_tokens


def tokenize_term(term: str, stopwords: set[str]) -> str:
    tokens = tokenize_text(term, stopwords)
    if len(tokens) != 1:
        raise ValueError("Expected exactly one token after preprocessing the term.")
    return tokens[0]
