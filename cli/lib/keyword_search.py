import math
import os
import pickle
import string
from collections import Counter

from nltk.stem import PorterStemmer

from .search_utils import (
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
        self.stopwords = {preprocess_text(word) for word in load_stopwords()}

    def __add_document(self, doc_id: int, text: str) -> None:
        self.term_frequencies[doc_id] = Counter()
        for token in tokenize_text(text, self.stopwords):
            self.index.setdefault(token, set()).add(doc_id)
            self.term_frequencies[doc_id][token] += 1

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

    def build(self) -> None:
        self.index.clear()
        self.docmap.clear()
        self.term_frequencies.clear()
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

    def load(self) -> None:
        """Load the local cache, raising FileNotFoundError if any file is missing."""
        with open(os.path.join(CACHE_DIR, "index.pkl"), "rb") as file:
            index = pickle.load(file)
        with open(os.path.join(CACHE_DIR, "docmap.pkl"), "rb") as file:
            docmap = pickle.load(file)
        with open(os.path.join(CACHE_DIR, "term_frequencies.pkl"), "rb") as file:
            term_frequencies = pickle.load(file)
        self.index = index
        self.docmap = docmap
        self.term_frequencies = term_frequencies


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
