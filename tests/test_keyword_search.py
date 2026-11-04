import io
import math
import pickle
import runpy
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from cli import lib
from cli.lib import keyword_search, search_utils


class InvertedIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.movies = [
            {"id": 10, "title": "The Matrix", "description": "Hackers are running."},
            {
                "id": 3,
                "title": "Runner",
                "description": "A matrix, matrix and reality.",
            },
            {"id": 8, "title": "Reality", "description": "A runner meets a hacker."},
        ]
        stopwords = patch.object(
            keyword_search, "load_stopwords", return_value=["the", "are", "a", "and"]
        )
        movies = patch.object(keyword_search, "load_movies", return_value=self.movies)
        self.mock_stopwords = stopwords.start()
        self.mock_movies = movies.start()
        self.addCleanup(stopwords.stop)
        self.addCleanup(movies.stop)
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.cache_dir = Path(self.tempdir.name) / "cache"
        cache = patch.object(keyword_search, "CACHE_DIR", str(self.cache_dir))
        cache.start()
        self.addCleanup(cache.stop)

    def build_index(self) -> keyword_search.InvertedIndex:
        index = keyword_search.InvertedIndex()
        index.build()
        return index

    def run_cli(self, *args: str) -> None:
        cli_path = Path(__file__).resolve().parents[1] / "cli" / "keyword_search_cli.py"
        with (
            patch.dict(
                "sys.modules",
                {
                    "lib": lib,
                    "lib.keyword_search": keyword_search,
                    "lib.search_utils": search_utils,
                },
            ),
            patch("sys.argv", [str(cli_path), *args]),
        ):
            runpy.run_path(str(cli_path), run_name="__main__")

    def test_build_indexes_titles_and_descriptions_with_existing_ids(self) -> None:
        index = self.build_index()

        self.assertEqual(index.index["matrix"], {3, 10})
        self.assertEqual(index.index["hacker"], {8, 10})
        self.assertEqual(index.index["run"], {10})
        self.assertEqual(index.docmap, {movie["id"]: movie for movie in self.movies})
        self.assertNotIn("the", index.index)
        self.assertNotIn("and", index.index)
        self.assertNotIn("running", index.index)

    def test_build_counts_each_preprocessed_token_per_document(self) -> None:
        index = self.build_index()

        self.assertIsInstance(index.term_frequencies[3], Counter)
        self.assertEqual(
            index.term_frequencies[3], Counter({"matrix": 2, "runner": 1, "realiti": 1})
        )
        self.assertEqual(index.get_tf(10, "matrix"), 1)
        self.assertEqual(index.get_tf(10, "run"), 1)
        self.assertNotIn("and", index.term_frequencies[3])

    def test_doc_lengths_start_empty_with_a_project_cache_path(self) -> None:
        index = keyword_search.InvertedIndex()

        self.assertEqual(index.doc_lengths, {})
        self.assertEqual(index.doc_lengths_path, str(self.cache_dir / "doc_lengths.pkl"))

    def test_build_tracks_document_lengths_after_tokenization(self) -> None:
        index = self.build_index()

        self.assertEqual(index.doc_lengths, {10: 3, 3: 4, 8: 4})
        for doc_id, length in index.doc_lengths.items():
            self.assertEqual(length, sum(index.term_frequencies[doc_id].values()))

    def test_average_doc_length_handles_empty_and_populated_indexes(self) -> None:
        index = keyword_search.InvertedIndex()
        get_average = getattr(index, "_InvertedIndex__get_avg_doc_length")
        self.assertEqual(get_average(), 0.0)

        index.build()

        self.assertAlmostEqual(get_average(), 11 / 3)

    def test_zero_token_documents_have_zero_lengths_and_scores(self) -> None:
        self.mock_movies.return_value = [
            {"id": 1, "title": "The", "description": "And a!"}
        ]

        index = self.build_index()

        self.assertEqual(index.doc_lengths, {1: 0})
        self.assertEqual(getattr(index, "_InvertedIndex__get_avg_doc_length")(), 0.0)
        self.assertEqual(index.get_bm25_tf(1, "matrix", k1=0, b=1), 0.0)

    def test_average_doc_length_includes_zero_token_documents(self) -> None:
        self.mock_movies.return_value = [
            {"id": 1, "title": "The", "description": "And a!"},
            {"id": 2, "title": "Bear", "description": "Forest"},
        ]

        index = self.build_index()

        self.assertEqual(index.doc_lengths, {1: 0, 2: 2})
        self.assertEqual(getattr(index, "_InvertedIndex__get_avg_doc_length")(), 1.0)

    def test_tf_combines_stemmed_occurrences_in_title_and_description(self) -> None:
        self.mock_movies.return_value = [
            {"id": 1, "title": "Running", "description": "Runs running run."}
        ]

        index = self.build_index()

        self.assertEqual(index.get_tf(1, "run"), 4)

    def test_get_tf_returns_zero_for_missing_terms_and_documents(self) -> None:
        index = self.build_index()
        original_frequencies = index.term_frequencies.copy()

        self.assertEqual(index.get_tf(3, "missing"), 0)
        self.assertEqual(index.get_tf(999, "matrix"), 0)
        self.assertNotIn("missing", index.term_frequencies[3])
        self.assertNotIn(999, index.term_frequencies)
        self.assertEqual(index.term_frequencies, original_frequencies)

    def test_tokenize_term_normalizes_and_stems_a_single_token(self) -> None:
        stopwords = keyword_search.InvertedIndex().stopwords

        for term, expected in [
            ("MATRIX!", "matrix"),
            ("Hackers", "hacker"),
            ("running", "run"),
            ("matrix and", "matrix"),
        ]:
            with self.subTest(term=term):
                self.assertEqual(keyword_search.tokenize_term(term, stopwords), expected)

    def test_tokenize_term_rejects_zero_or_multiple_tokens(self) -> None:
        stopwords = keyword_search.InvertedIndex().stopwords

        for term in ["", "!!!", "the", "matrix hacker"]:
            with self.subTest(term=term):
                with self.assertRaisesRegex(ValueError, "exactly one token"):
                    keyword_search.tokenize_term(term, stopwords)

    def test_get_documents_returns_sorted_unique_ids_without_mutating_index(
        self,
    ) -> None:
        index = self.build_index()

        self.assertEqual(index.get_documents("matrix"), [3, 10])
        self.assertEqual(index.get_documents("missing"), [])
        self.assertNotIn("missing", index.index)
        self.assertEqual(index.get_documents("Matrix"), [])

    def test_build_clears_stale_documents(self) -> None:
        index = self.build_index()
        self.mock_movies.return_value = [self.movies[0]]

        index.build()

        self.assertEqual(index.get_documents("matrix"), [10])
        self.assertNotIn("realiti", index.index)
        self.assertEqual(list(index.docmap), [10])
        self.assertEqual(list(index.term_frequencies), [10])
        self.assertEqual(index.get_tf(10, "matrix"), 1)
        self.assertEqual(index.doc_lengths, {10: 3})

    def test_save_creates_cache_directory_and_pickles_all_attributes(self) -> None:
        index = self.build_index()
        self.assertFalse(self.cache_dir.exists())

        index.save()

        with (self.cache_dir / "index.pkl").open("rb") as file:
            self.assertEqual(pickle.load(file), index.index)
        with (self.cache_dir / "docmap.pkl").open("rb") as file:
            self.assertEqual(pickle.load(file), index.docmap)
        with (self.cache_dir / "term_frequencies.pkl").open("rb") as file:
            self.assertEqual(pickle.load(file), index.term_frequencies)
        with (self.cache_dir / "doc_lengths.pkl").open("rb") as file:
            self.assertEqual(pickle.load(file), index.doc_lengths)

    def test_load_restores_saved_index(self) -> None:
        original = self.build_index()
        original.save()
        restored = keyword_search.InvertedIndex()

        restored.load()

        self.assertEqual(restored.index, original.index)
        self.assertEqual(restored.docmap, original.docmap)
        self.assertEqual(restored.term_frequencies, original.term_frequencies)
        self.assertIsInstance(restored.term_frequencies[3], Counter)
        self.assertEqual(restored.doc_lengths, original.doc_lengths)

    def test_load_raises_when_cache_is_missing(self) -> None:
        index = keyword_search.InvertedIndex()

        with self.assertRaises(FileNotFoundError):
            index.load()

    def test_load_raises_if_any_cache_file_is_missing(self) -> None:
        index = self.build_index()
        for filename in [
            "index.pkl",
            "docmap.pkl",
            "term_frequencies.pkl",
            "doc_lengths.pkl",
        ]:
            with self.subTest(filename=filename):
                index.save()
                (self.cache_dir / filename).unlink()

                with self.assertRaises(FileNotFoundError):
                    keyword_search.InvertedIndex().load()

    def test_build_command_saves_index_without_hardcoded_lookup(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            keyword_search.build_command()

        self.assertEqual(output.getvalue(), "Index built and saved.\n")
        self.assertTrue((self.cache_dir / "index.pkl").is_file())
        self.assertTrue((self.cache_dir / "docmap.pkl").is_file())
        self.assertTrue((self.cache_dir / "term_frequencies.pkl").is_file())
        self.assertTrue((self.cache_dir / "doc_lengths.pkl").is_file())

    def test_search_uses_saved_index_without_loading_movies(self) -> None:
        self.build_index().save()
        self.mock_movies.reset_mock()

        results = keyword_search.search_command("MATRIX! hackers")

        self.assertEqual([movie["id"] for movie in results], [3, 10, 8])
        self.mock_movies.assert_not_called()

    def test_search_raises_without_rebuilding_if_cache_is_missing(self) -> None:
        with self.assertRaises(FileNotFoundError):
            keyword_search.search_command("running")

        self.mock_movies.assert_not_called()
        self.assertFalse(self.cache_dir.exists())

    def test_search_raises_without_rebuilding_if_docmap_is_missing(self) -> None:
        self.build_index().save()
        self.mock_movies.reset_mock()
        (self.cache_dir / "docmap.pkl").unlink()

        with self.assertRaises(FileNotFoundError):
            keyword_search.search_command("matrix")

        self.mock_movies.assert_not_called()
        self.assertFalse((self.cache_dir / "docmap.pkl").exists())

    def test_search_matches_whole_tokens_not_substrings(self) -> None:
        self.build_index().save()

        self.assertEqual(keyword_search.search_command("hack"), [])

    def test_search_deduplicates_results_and_respects_limit(self) -> None:
        self.build_index().save()

        results = keyword_search.search_command("matrix matrix reality", limit=2)

        self.assertEqual([movie["id"] for movie in results], [3, 10])

    def test_search_stops_before_next_token_once_five_results_are_found(self) -> None:
        self.mock_movies.return_value = [
            {"id": doc_id, "title": f"Movie {doc_id}", "description": "Matrix"}
            for doc_id in range(8, 0, -1)
        ]
        self.build_index().save()
        original_get_documents = keyword_search.InvertedIndex.get_documents

        with patch.object(
            keyword_search.InvertedIndex,
            "get_documents",
            autospec=True,
            side_effect=original_get_documents,
        ) as get_documents:
            results = keyword_search.search_command("matrix hacker")

        self.assertEqual([movie["id"] for movie in results], [1, 2, 3, 4, 5])
        get_documents.assert_called_once()
        self.assertEqual(get_documents.call_args.args[1], "matrix")

    def test_search_does_not_return_duplicate_documents_across_tokens(self) -> None:
        self.build_index().save()

        results = keyword_search.search_command("matrix matrix reality")

        self.assertEqual([movie["id"] for movie in results], [3, 10, 8])

    def test_search_empty_unknown_and_stopword_only_queries(self) -> None:
        self.build_index().save()

        for query in ["", "!!!", "the and", "unknown"]:
            with self.subTest(query=query):
                self.assertEqual(keyword_search.search_command(query), [])

    def test_search_nonpositive_limits_do_not_build_index(self) -> None:
        for limit in [0, -1]:
            with self.subTest(limit=limit):
                self.assertEqual(keyword_search.search_command("matrix", limit), [])
        self.mock_movies.assert_not_called()

    def test_bm25_multiplies_length_normalized_tf_and_bm25_idf(self) -> None:
        index = self.build_index()

        self.assertAlmostEqual(index.bm25(3, "matrix"), (440 / 317) * math.log(1.6))
        self.assertAlmostEqual(index.bm25(10, "matrix"), (110 / 101) * math.log(1.6))

    def test_bm25_returns_zero_for_missing_terms_and_documents(self) -> None:
        index = self.build_index()

        self.assertEqual(index.bm25(3, "unknown"), 0.0)
        self.assertEqual(index.bm25(999, "matrix"), 0.0)

    def test_bm25_search_sums_query_terms_and_sorts_descending(self) -> None:
        index = self.build_index()

        results = index.bm25_search("matrix hacker", limit=3)

        self.assertEqual([doc_id for doc_id, _ in results], [10, 3, 8])
        expected_scores = {
            10: 2 * (110 / 101) * math.log(1.6),
            3: (440 / 317) * math.log(1.6),
            8: (220 / 229) * math.log(1.6),
        }
        for doc_id, score in results:
            self.assertAlmostEqual(score, expected_scores[doc_id])
        self.assertEqual(
            [doc_id for doc_id, _ in index.bm25_search("matrix", limit=3)],
            [3, 10, 8],
        )

    def test_bm25_search_normalizes_and_stems_the_query(self) -> None:
        index = self.build_index()

        self.assertEqual(
            index.bm25_search("The MATRIX! and hackers"),
            index.bm25_search("matrix hacker"),
        )
        self.assertEqual(index.bm25_search("RUNNING"), index.bm25_search("run"))

    def test_bm25_search_counts_repeated_query_tokens(self) -> None:
        index = self.build_index()

        for doc_id, score in index.bm25_search("matrix matrix"):
            self.assertAlmostEqual(score, 2 * index.bm25(doc_id, "matrix"))

    def test_bm25_search_scores_all_documents_before_applying_limit(self) -> None:
        self.mock_movies.return_value = [
            {"id": doc_id, "title": f"Movie {doc_id}", "description": "matrix " * doc_id}
            for doc_id in range(1, 9)
        ]
        index = self.build_index()

        with patch.object(index, "bm25", wraps=index.bm25) as bm25:
            results = index.bm25_search("matrix", limit=2)

        self.assertEqual([doc_id for doc_id, _ in results], [8, 7])
        self.assertEqual(bm25.call_count, 8)
        self.assertEqual(
            [doc_id for doc_id, _ in index.bm25_search("matrix")], [8, 7, 6, 5, 4]
        )
        self.assertEqual(len(index.bm25_search("matrix", limit=100)), 8)

    def test_bm25_search_preserves_document_order_when_scores_tie(self) -> None:
        self.mock_movies.return_value = [
            {"id": 8, "title": "Bear", "description": ""},
            {"id": 3, "title": "Bear", "description": ""},
        ]
        index = self.build_index()

        results = index.bm25_search("bear")

        self.assertEqual([doc_id for doc_id, _ in results], [8, 3])
        self.assertEqual(results[0][1], results[1][1])

    def test_bm25_search_keeps_zero_scores_when_no_query_tokens_match(self) -> None:
        index = self.build_index()

        for query in ["xyzzy", "", "!!!", "the and"]:
            with self.subTest(query=query):
                self.assertEqual(index.bm25_search(query, limit=2), [(10, 0.0), (3, 0.0)])

    def test_bm25_search_handles_empty_indexes_and_nonpositive_limits(self) -> None:
        self.assertEqual(keyword_search.InvertedIndex().bm25_search("matrix"), [])
        index = self.build_index()

        with patch.object(index, "bm25") as bm25:
            for limit in [0, -1]:
                with self.subTest(limit=limit):
                    self.assertEqual(index.bm25_search("matrix", limit), [])
        bm25.assert_not_called()

    def test_cli_bm25search_uses_cache_and_prints_ranked_ids_titles_and_scores(self) -> None:
        self.build_index().save()
        self.mock_movies.reset_mock()
        output = io.StringIO()

        with redirect_stdout(output):
            self.run_cli("bm25search", "matrix hacker")

        self.assertEqual(
            output.getvalue(),
            "1. (10) The Matrix - Score: 1.02\n"
            "2. (3) Runner - Score: 0.65\n"
            "3. (8) Reality - Score: 0.45\n",
        )
        self.mock_movies.assert_not_called()

    def test_cli_bm25search_accepts_a_limit(self) -> None:
        self.build_index().save()
        output = io.StringIO()

        with redirect_stdout(output):
            self.run_cli("bm25search", "matrix hacker", "--limit", "1")

        self.assertEqual(output.getvalue(), "1. (10) The Matrix - Score: 1.02\n")

    def test_cli_bm25search_defaults_to_five_results(self) -> None:
        self.mock_movies.return_value = [
            {"id": doc_id, "title": f"Movie {doc_id}", "description": "matrix " * doc_id}
            for doc_id in range(1, 9)
        ]
        self.build_index().save()
        output = io.StringIO()

        with redirect_stdout(output):
            self.run_cli("bm25search", "matrix")

        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 5)
        self.assertTrue(lines[0].startswith("1. (8) Movie 8 - Score: "))

    def test_cli_bm25search_reports_missing_cache_without_rebuilding(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as error:
                self.run_cli("bm25search", "matrix")

        self.assertEqual(error.exception.code, 1)
        self.assertIn("uv run cli/keyword_search_cli.py build", error_output.getvalue())
        self.mock_movies.assert_not_called()
        self.assertFalse(self.cache_dir.exists())

    def test_tf_command_uses_saved_counts_and_tokenizes_the_term(self) -> None:
        self.build_index().save()
        self.mock_movies.reset_mock()

        self.assertEqual(keyword_search.tf_command(3, "MATRIX!"), 2)
        self.assertEqual(keyword_search.tf_command(10, "RUNNING"), 1)
        self.assertEqual(keyword_search.tf_command(3, "unknown"), 0)
        self.assertEqual(keyword_search.tf_command(999, "matrix"), 0)
        self.mock_movies.assert_not_called()

    def test_get_bm25_tf_uses_default_length_normalization(self) -> None:
        index = self.build_index()

        self.assertEqual(search_utils.BM25_K1, 1.5)
        self.assertEqual(search_utils.BM25_B, 0.75)
        self.assertAlmostEqual(index.get_bm25_tf(3, "matrix"), 440 / 317)
        self.assertAlmostEqual(index.get_bm25_tf(10, "matrix"), 110 / 101)

    def test_get_bm25_tf_disables_length_normalization_when_b_is_zero(self) -> None:
        index = self.build_index()

        self.assertAlmostEqual(index.get_bm25_tf(3, "matrix", b=0), 10 / 7)
        self.assertEqual(index.get_bm25_tf(10, "matrix", b=0), 1.0)

    def test_get_bm25_tf_boosts_short_and_penalizes_long_documents(self) -> None:
        self.mock_movies.return_value = [
            {"id": 1, "title": "Bear", "description": ""},
            {"id": 2, "title": "Bear", "description": "Forest"},
            {"id": 3, "title": "Bear", "description": "Forest forest"},
        ]
        index = self.build_index()

        for b, short_score, long_score in [
            (0, 1.0, 1.0),
            (0.75, 40 / 31, 40 / 49),
            (1, 10 / 7, 10 / 13),
        ]:
            with self.subTest(b=b):
                self.assertAlmostEqual(index.get_bm25_tf(1, "bear", b=b), short_score)
                self.assertEqual(index.get_bm25_tf(2, "bear", b=b), 1.0)
                self.assertAlmostEqual(index.get_bm25_tf(3, "bear", b=b), long_score)

    def test_get_bm25_tf_accepts_custom_and_zero_k1(self) -> None:
        index = self.build_index()

        self.assertAlmostEqual(index.get_bm25_tf(3, "matrix", k1=3), 352 / 229)
        self.assertEqual(index.get_bm25_tf(3, "matrix", k1=0), 1.0)

    def test_get_bm25_tf_returns_zero_for_missing_terms_and_documents(self) -> None:
        index = self.build_index()

        for doc_id, term in [(3, "missing"), (999, "matrix")]:
            for k1 in [1.5, 0]:
                with self.subTest(doc_id=doc_id, term=term, k1=k1):
                    self.assertEqual(index.get_bm25_tf(doc_id, term, k1), 0.0)

    def test_get_bm25_tf_has_diminishing_returns(self) -> None:
        index = self.build_index()
        scores = []
        for frequency in [1, 2, 3, 1000]:
            index.term_frequencies[3]["matrix"] = frequency
            scores.append(index.get_bm25_tf(3, "matrix", b=0))

        self.assertEqual(scores, sorted(scores))
        self.assertGreater(scores[1] - scores[0], scores[2] - scores[1])
        self.assertLess(scores[-1], search_utils.BM25_K1 + 1)

    def test_get_bm25_tf_rejects_negative_or_nonfinite_k1(self) -> None:
        index = self.build_index()

        for k1 in [-1.0, math.nan, math.inf, -math.inf]:
            with self.subTest(k1=k1):
                with self.assertRaisesRegex(ValueError, "k1"):
                    index.get_bm25_tf(3, "matrix", k1)

    def test_get_bm25_tf_rejects_out_of_range_or_nonfinite_b(self) -> None:
        index = self.build_index()

        for b in [-0.1, 1.1, math.nan, math.inf, -math.inf]:
            with self.subTest(b=b):
                with self.assertRaisesRegex(ValueError, "BM25 b"):
                    index.get_bm25_tf(3, "matrix", b=b)

    def test_bm25_tf_command_loads_saved_index_and_tokenizes_term(self) -> None:
        self.build_index().save()
        self.mock_movies.reset_mock()
        original_load = keyword_search.InvertedIndex.load

        with patch.object(
            keyword_search.InvertedIndex,
            "load",
            autospec=True,
            side_effect=original_load,
        ) as load:
            score = keyword_search.bm25_tf_command(3, "MATRIX!")

        self.assertAlmostEqual(score, 440 / 317)
        load.assert_called_once()
        self.assertAlmostEqual(keyword_search.bm25_tf_command(3, "MATRIX!", 3), 352 / 229)
        self.assertAlmostEqual(keyword_search.bm25_tf_command(3, "MATRIX!", 3, 0), 8 / 5)
        self.assertAlmostEqual(keyword_search.bm25_tf_command(10, "RUNNING"), 110 / 101)
        self.mock_movies.assert_not_called()

    def test_cli_bm25tf_uses_default_or_optional_k1_and_b_and_formats_score(self) -> None:
        self.build_index().save()

        for doc_id, term, k1, b, expected in [
            ("3", "MATRIX!", None, None, "1.39"),
            ("3", "matrix", "3", None, "1.54"),
            ("3", "matrix", "0", None, "1.00"),
            ("3", "matrix", "1.5", "0", "1.43"),
            ("3", "matrix", "3", "0", "1.60"),
            ("3", "matrix", "2", "1", "1.43"),
            ("3", "unknown", None, None, "0.00"),
            ("999", "matrix", None, None, "0.00"),
            ("3", "unknown", "0", "1", "0.00"),
        ]:
            with self.subTest(doc_id=doc_id, term=term, k1=k1, b=b):
                args = ["bm25tf", doc_id, term]
                if k1 is not None:
                    args.append(k1)
                if b is not None:
                    args.append(b)
                output = io.StringIO()
                with redirect_stdout(output):
                    self.run_cli(*args)
                self.assertEqual(
                    output.getvalue(),
                    f"BM25 TF score of '{term}' in document '{doc_id}': {expected}\n",
                )

    def test_cli_bm25tf_rejects_terms_that_do_not_produce_one_token(self) -> None:
        self.build_index().save()

        for term in ["", "!!!", "the", "matrix hacker"]:
            with self.subTest(term=term):
                error_output = io.StringIO()
                with redirect_stderr(error_output):
                    with self.assertRaises(SystemExit) as error:
                        self.run_cli("bm25tf", "3", term)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("exactly one token", error_output.getvalue())

    def test_cli_bm25tf_rejects_invalid_k1(self) -> None:
        self.build_index().save()

        for k1 in ["-1", "nan", "inf"]:
            with self.subTest(k1=k1):
                error_output = io.StringIO()
                with redirect_stderr(error_output):
                    with self.assertRaises(SystemExit) as error:
                        self.run_cli("bm25tf", "3", "matrix", k1)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("k1", error_output.getvalue())

    def test_cli_bm25tf_rejects_invalid_b(self) -> None:
        self.build_index().save()

        for b in ["-0.1", "1.1", "nan", "inf"]:
            with self.subTest(b=b):
                error_output = io.StringIO()
                with redirect_stderr(error_output):
                    with self.assertRaises(SystemExit) as error:
                        self.run_cli("bm25tf", "3", "matrix", "1.5", b)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("BM25 b", error_output.getvalue())

    def test_cli_bm25tf_reports_missing_doc_lengths_cache(self) -> None:
        self.build_index().save()
        self.mock_movies.reset_mock()
        (self.cache_dir / "doc_lengths.pkl").unlink()
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as error:
                self.run_cli("bm25tf", "3", "matrix")

        self.assertEqual(error.exception.code, 1)
        self.assertIn("uv run cli/keyword_search_cli.py build", error_output.getvalue())
        self.mock_movies.assert_not_called()
        self.assertFalse((self.cache_dir / "doc_lengths.pkl").exists())

    def test_cli_bm25tf_reports_missing_cache_without_rebuilding(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as error:
                self.run_cli("bm25tf", "3", "matrix")

        self.assertEqual(error.exception.code, 1)
        self.assertIn("uv run cli/keyword_search_cli.py build", error_output.getvalue())
        self.mock_movies.assert_not_called()
        self.assertFalse(self.cache_dir.exists())

    def test_idf_command_counts_documents_not_repeated_occurrences(self) -> None:
        self.build_index().save()
        self.mock_movies.reset_mock()

        self.assertAlmostEqual(keyword_search.idf_command("matrix"), math.log(4 / 3))
        self.mock_movies.assert_not_called()

    def test_idf_command_normalizes_and_stems_the_term(self) -> None:
        self.build_index().save()

        self.assertAlmostEqual(keyword_search.idf_command("HACKERS!"), math.log(4 / 3))
        self.assertAlmostEqual(keyword_search.idf_command("RUNNING"), math.log(4 / 2))

    def test_idf_command_returns_zero_when_every_document_matches(self) -> None:
        for movie in self.movies:
            movie["description"] += " Galaxy."
        self.build_index().save()

        self.assertEqual(keyword_search.idf_command("galaxy"), 0.0)

    def test_idf_command_smooths_terms_missing_from_all_documents(self) -> None:
        self.build_index().save()

        self.assertAlmostEqual(keyword_search.idf_command("xyzzy"), math.log(4))

    def test_idf_command_handles_an_empty_corpus(self) -> None:
        self.mock_movies.return_value = []
        self.build_index().save()

        self.assertEqual(keyword_search.idf_command("matrix"), 0.0)

    def test_get_bm25_idf_counts_documents_not_repeated_occurrences(self) -> None:
        index = self.build_index()

        self.assertAlmostEqual(index.get_bm25_idf("matrix"), math.log(1.6))
        self.assertAlmostEqual(index.get_bm25_idf("run"), math.log(8 / 3))

    def test_get_bm25_idf_smooths_missing_terms_without_mutating_index(self) -> None:
        index = self.build_index()

        self.assertAlmostEqual(index.get_bm25_idf("xyzzy"), math.log(8))
        self.assertNotIn("xyzzy", index.index)

    def test_get_bm25_idf_is_positive_even_for_a_universal_term(self) -> None:
        for movie in self.movies:
            movie["description"] += " Galaxy."
        index = self.build_index()

        self.assertAlmostEqual(index.get_bm25_idf("galaxi"), math.log(8 / 7))
        self.assertGreater(index.get_bm25_idf("galaxi"), 0.0)
        self.assertEqual(index.get_idf("galaxi"), 0.0)

    def test_get_bm25_idf_handles_an_empty_corpus(self) -> None:
        index = keyword_search.InvertedIndex()

        self.assertAlmostEqual(index.get_bm25_idf("matrix"), math.log(2))

    def test_bm25_idf_command_loads_saved_index_and_tokenizes_term(self) -> None:
        self.build_index().save()
        self.mock_movies.reset_mock()
        original_load = keyword_search.InvertedIndex.load

        with patch.object(
            keyword_search.InvertedIndex,
            "load",
            autospec=True,
            side_effect=original_load,
        ) as load:
            score = keyword_search.bm25_idf_command("HACKERS!")

        self.assertAlmostEqual(score, math.log(1.6))
        load.assert_called_once()
        self.assertAlmostEqual(
            keyword_search.bm25_idf_command("RUNNING"), math.log(8 / 3)
        )
        self.mock_movies.assert_not_called()

    def test_cli_bm25idf_prints_original_term_and_two_decimal_places(self) -> None:
        self.build_index().save()

        for term, expected in [
            ("MATRIX!", "0.47"),
            ("running", "0.98"),
            ("xyzzy", "2.08"),
        ]:
            with self.subTest(term=term):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.run_cli("bm25idf", term)
                self.assertEqual(
                    output.getvalue(), f"BM25 IDF score of '{term}': {expected}\n"
                )

    def test_cli_bm25idf_rejects_terms_that_do_not_produce_one_token(self) -> None:
        self.build_index().save()

        for term in ["", "!!!", "the", "matrix hacker"]:
            with self.subTest(term=term):
                error_output = io.StringIO()
                with redirect_stderr(error_output):
                    with self.assertRaises(SystemExit) as error:
                        self.run_cli("bm25idf", term)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("exactly one token", error_output.getvalue())

    def test_cli_bm25idf_reports_missing_cache_without_rebuilding(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as error:
                self.run_cli("bm25idf", "matrix")

        self.assertEqual(error.exception.code, 1)
        self.assertIn("uv run cli/keyword_search_cli.py build", error_output.getvalue())
        self.mock_movies.assert_not_called()
        self.assertFalse(self.cache_dir.exists())

    def test_tfidf_command_multiplies_tf_and_idf_and_loads_cache_once(self) -> None:
        self.build_index().save()
        self.mock_movies.reset_mock()
        original_load = keyword_search.InvertedIndex.load

        with patch.object(
            keyword_search.InvertedIndex,
            "load",
            autospec=True,
            side_effect=original_load,
        ) as load:
            score = keyword_search.tfidf_command(3, "matrix")

        self.assertAlmostEqual(score, 2 * math.log(4 / 3))
        load.assert_called_once()
        self.mock_movies.assert_not_called()

    def test_tfidf_command_normalizes_and_stems_the_term(self) -> None:
        self.build_index().save()

        self.assertAlmostEqual(
            keyword_search.tfidf_command(3, "MATRIX!"), 2 * math.log(4 / 3)
        )
        self.assertAlmostEqual(keyword_search.tfidf_command(10, "RUNNING"), math.log(2))

    def test_tfidf_command_returns_zero_for_missing_terms_and_documents(self) -> None:
        self.build_index().save()

        for doc_id, term in [(3, "hacker"), (3, "xyzzy"), (999, "matrix")]:
            with self.subTest(doc_id=doc_id, term=term):
                self.assertEqual(keyword_search.tfidf_command(doc_id, term), 0.0)

    def test_tfidf_command_returns_zero_for_a_universal_term(self) -> None:
        for movie in self.movies:
            movie["description"] += " Galaxy galaxy."
        self.build_index().save()

        self.assertEqual(keyword_search.tfidf_command(3, "galaxy"), 0.0)

    def test_tfidf_command_handles_an_empty_corpus(self) -> None:
        self.mock_movies.return_value = []
        self.build_index().save()

        self.assertEqual(keyword_search.tfidf_command(3, "matrix"), 0.0)

    def test_cli_tfidf_prints_original_arguments_and_two_decimal_places(self) -> None:
        self.build_index().save()

        for doc_id, term, expected in [
            ("3", "MATRIX!", "0.58"),
            ("10", "running", "0.69"),
            ("3", "hacker", "0.00"),
            ("3", "xyzzy", "0.00"),
            ("999", "matrix", "0.00"),
        ]:
            with self.subTest(doc_id=doc_id, term=term):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.run_cli("tfidf", doc_id, term)
                self.assertEqual(
                    output.getvalue(),
                    f"TF-IDF score of '{term}' in document '{doc_id}': {expected}\n",
                )

    def test_cli_tfidf_rejects_terms_that_do_not_produce_one_token(self) -> None:
        self.build_index().save()

        for term in ["", "!!!", "the", "matrix hacker"]:
            with self.subTest(term=term):
                error_output = io.StringIO()
                with redirect_stderr(error_output):
                    with self.assertRaises(SystemExit) as error:
                        self.run_cli("tfidf", "3", term)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("exactly one token", error_output.getvalue())

    def test_cli_tfidf_reports_missing_cache_without_rebuilding(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as error:
                self.run_cli("tfidf", "3", "matrix")

        self.assertEqual(error.exception.code, 1)
        self.assertIn("uv run cli/keyword_search_cli.py build", error_output.getvalue())
        self.mock_movies.assert_not_called()
        self.assertFalse(self.cache_dir.exists())

    def test_cli_idf_prints_original_term_and_two_decimal_places(self) -> None:
        self.build_index().save()

        for term, expected in [
            ("MATRIX!", "0.29"),
            ("running", "0.69"),
            ("xyzzy", "1.39"),
        ]:
            with self.subTest(term=term):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.run_cli("idf", term)
                self.assertEqual(
                    output.getvalue(),
                    f"Inverse document frequency of '{term}': {expected}\n",
                )

    def test_cli_idf_rejects_terms_that_do_not_produce_one_token(self) -> None:
        self.build_index().save()

        for term in ["", "!!!", "the", "matrix hacker"]:
            with self.subTest(term=term):
                error_output = io.StringIO()
                with redirect_stderr(error_output):
                    with self.assertRaises(SystemExit) as error:
                        self.run_cli("idf", term)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("exactly one token", error_output.getvalue())

    def test_cli_idf_reports_missing_cache_without_rebuilding(self) -> None:
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as error:
                self.run_cli("idf", "matrix")

        self.assertEqual(error.exception.code, 1)
        self.assertIn("uv run cli/keyword_search_cli.py build", error_output.getvalue())
        self.mock_movies.assert_not_called()
        self.assertFalse(self.cache_dir.exists())

    def test_cli_tf_prints_only_the_frequency(self) -> None:
        self.build_index().save()

        for doc_id, term, expected in [("3", "MATRIX!", "2\n"), ("3", "unknown", "0\n")]:
            with self.subTest(term=term):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.run_cli("tf", doc_id, term)
                self.assertEqual(output.getvalue(), expected)

    def test_cli_tf_rejects_terms_that_do_not_produce_one_token(self) -> None:
        self.build_index().save()

        for term in ["!!!", "the", "matrix hacker"]:
            with self.subTest(term=term):
                error_output = io.StringIO()
                with redirect_stderr(error_output):
                    with self.assertRaises(SystemExit) as error:
                        self.run_cli("tf", "3", term)
                self.assertEqual(error.exception.code, 2)
                self.assertIn("exactly one token", error_output.getvalue())

    def test_cli_tf_reports_missing_term_frequency_cache(self) -> None:
        self.build_index().save()
        self.mock_movies.reset_mock()
        (self.cache_dir / "term_frequencies.pkl").unlink()
        error_output = io.StringIO()

        with redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as error:
                self.run_cli("tf", "3", "matrix")

        self.assertEqual(error.exception.code, 1)
        self.assertIn("uv run cli/keyword_search_cli.py build", error_output.getvalue())
        self.mock_movies.assert_not_called()

    def test_cli_prints_document_title_and_id(self) -> None:
        self.build_index().save()
        output = io.StringIO()

        with redirect_stdout(output):
            self.run_cli("search", "running")

        self.assertIn("1. The Matrix (ID: 10)", output.getvalue())

    def test_cli_reports_missing_cache_and_exits_without_rebuilding(self) -> None:
        error_output = io.StringIO()

        with redirect_stdout(io.StringIO()), redirect_stderr(error_output):
            with self.assertRaises(SystemExit) as error:
                self.run_cli("search", "matrix")

        self.assertEqual(error.exception.code, 1)
        self.assertIn("Error: Search cache is missing.", error_output.getvalue())
        self.assertIn("uv run cli/keyword_search_cli.py build", error_output.getvalue())
        self.mock_movies.assert_not_called()
        self.assertFalse(self.cache_dir.exists())


if __name__ == "__main__":
    unittest.main()
